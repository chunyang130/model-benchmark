#!/usr/bin/env python3
"""課程問答 RAG：先從知識庫檢索相關段落，再讓 Claude 只根據這些段落回答，並由程式驗證引用。

只用 Python 標準函式庫（3.10 以上），不需安裝任何套件。

用法:
  # 看檢索結果（不呼叫 API，免費）
  python rag.py search "多益班多少錢"

  # 問一個問題（需要 API key）
  python rag.py ask "開課後兩週想退費可以退多少？"

  # 跑評估題組；加 --mock 用預先錄好的模型回覆（免費）
  python rag.py eval --mock eval/mock_responses.json
  python rag.py eval

流程：
  1. 切段：知識庫每份文件依「## 小標」切成段落，每段有編號（例：refund#2）
  2. 檢索：先用同義詞表把學員說法換成文件用語，再用 BM25 找出最相關的前 4 段（中文以兩字一組切詞）
  3. 生成：只把這 4 段交給 Claude，用 tool use 強制回傳「能否回答、答案、引用」
  4. 驗證：引用的段落必須在檢索結果中、引用原文必須一字不差、答案裡的數字必須出現在引用段落；
           不合格就把錯誤告訴模型重試一次
  5. 拒答：資料裡沒有的問題，必須回答「資料中找不到」，不可以用常識補
"""
import argparse, glob, json, math, os, re, sys, time, urllib.error, urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "knowledge_base")
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-5-5"
TOP_K = 4
NOT_FOUND = "資料中找不到這個問題的答案，建議請顧問協助確認。"


# ---------- 1. 切段 ----------
def load_chunks(kb_dir=KB_DIR):
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        text = open(path, encoding="utf-8").read()
        title = re.search(r"^# (.+)$", text, flags=re.M)
        doc_title = re.sub(r"（.*?）", "", title.group(1)).strip() if title else stem
        for i, sec in enumerate(re.split(r"^## ", text, flags=re.M)[1:], 1):
            heading, _, body = sec.partition("\n")
            chunks.append({"id": f"{stem}#{i}", "doc": doc_title, "heading": heading.strip(),
                           "text": f"{doc_title}－{heading.strip()}\n{body.strip()}"})
    return chunks


# ---------- 2. 檢索（BM25） ----------
def tokenize(text):
    """英文、數字取整個詞；中文連續字串切成兩字一組（bigram），單一個中文字就保留單字。"""
    text = text.lower().replace(",", "")
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    for run in re.findall(r"[一-鿿]+", text):
        tokens += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


# 同義詞擴充：學員的說法 → 文件裡的用詞。關鍵字檢索認不出同義詞，所以在搜尋前補上文件用語
SYNONYMS = {
    "錯過": "缺課", "沒去上": "缺課", "請假": "缺課", "補上": "補課",
    "暫停": "保留學籍", "停課": "保留學籍", "先停": "保留學籍", "休學": "保留學籍",
    "打折": "優惠 折", "便宜": "優惠", "多少錢": "價 費用", "學費": "價",
    "幾點": "時間", "開門": "營業時間",
}


def expand(question):
    extra = [v for k, v in SYNONYMS.items() if k in question]
    return question + (" " + " ".join(extra) if extra else "")


def retrieve(question, chunks, index, k=TOP_K):
    ranked = sorted(zip(index.scores(expand(question)), chunks), key=lambda x: -x[0])
    return [dict(c, score=round(s, 2)) for s, c in ranked[:k] if s > 0]


# ---------- 3. 生成 ----------
SYSTEM = """你是補習班的課程客服，只能根據使用者提供的「參考資料」回答，用繁體中文。
規則：
- 參考資料沒有寫的內容，不可以用常識或推測補充。
- 參考資料不足以回答時，answerable 填 false，answer 填「資料中找不到這個問題的答案，建議請顧問協助確認。」，citations 留空。
- 可以回答時，每個重點都要引用：chunk_id 填段落編號，quote 從該段落原文直接複製一小段，一字不改。
- 答案裡的數字（金額、比例、天數）必須和參考資料完全相同。
一律呼叫 answer 工具回傳。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean", "description": "參考資料是否足以回答"},
        "answer": {"type": "string", "description": "給學員的回答，2–4 句"},
        "citations": {"type": "array", "items": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string", "description": "段落原文，一字不改"}},
            "required": ["chunk_id", "quote"]}},
    },
    "required": ["answerable", "answer", "citations"],
}


def build_context(retrieved):
    return "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in retrieved)


def call_claude(question, retrieved, model, api_key, feedback=None):
    content = f"參考資料：\n{build_context(retrieved)}\n\n問題：{question}"
    if feedback:
        content += f"\n\n上一次的回答有問題，請修正：\n{feedback}"
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["citations"]["items"]["properties"]["chunk_id"]["enum"] = [c["id"] for c in retrieved]
    body = {"model": model, "max_tokens": 1024, "system": SYSTEM,
            "tools": [{"name": "answer", "description": "回傳答案與引用", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "answer"},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"API 錯誤 {e.code}：{e.read().decode('utf-8', 'ignore')[:300]}")
    tool = next(b for b in resp["content"] if b["type"] == "tool_use")
    return tool["input"], resp.get("usage", {})


# ---------- 4. 驗證 ----------
def norm(s):
    return re.sub(r"\s+", "", s).replace(",", "").replace("，", "")


def numbers(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s.replace(",", "")))


def validate(resp, retrieved):
    errs = []
    by_id = {c["id"]: c for c in retrieved}
    cites = resp.get("citations", [])
    if not resp.get("answerable"):
        if cites:
            errs.append("無法回答時不應附引用")
        return errs
    if not cites:
        errs.append("可以回答時至少要有一個引用")
    cited_text = ""
    for c in cites:
        chunk = by_id.get(c.get("chunk_id"))
        if not chunk:
            errs.append(f"引用的段落 {c.get('chunk_id')} 不在檢索結果中")
            continue
        if norm(c.get("quote", "")) not in norm(chunk["text"]):
            errs.append(f"{c['chunk_id']} 的引用不是原文，請直接複製段落文字")
        cited_text += chunk["text"]
    missing = numbers(resp.get("answer", "")) - numbers(cited_text)
    if missing and cited_text:
        errs.append(f"答案中的數字 {sorted(missing)} 沒有出現在引用段落中")
    return errs


def answer(question, chunks, index, model=None, api_key=None, mock=None):
    """回傳 (結果, 檢索段落, 驗證問題, token 用量, 是否重試)。"""
    retrieved = retrieve(question, chunks, index)
    usage = {"input_tokens": 0, "output_tokens": 0}
    if not retrieved:
        return {"answerable": False, "answer": NOT_FOUND, "citations": []}, [], [], usage, False
    if mock is not None:
        tries = mock if isinstance(mock, list) else [mock]
        resp = tries[0]
        errs = validate(resp, retrieved)
        retried = bool(errs) and len(tries) > 1
        if retried:
            resp = tries[1]
            errs = validate(resp, retrieved)
        return resp, retrieved, errs, usage, retried
    resp, u = call_claude(question, retrieved, model, api_key)
    errs, retried = validate(resp, retrieved), False
    if errs:
        retried = True
        resp, u2 = call_claude(question, retrieved, model, api_key, feedback="\n".join(errs))
        errs = validate(resp, retrieved)
        for k in usage:
            usage[k] += u2.get(k, 0)
    for k in usage:
        usage[k] += u.get(k, 0)
    return resp, retrieved, errs, usage, retried


# ---------- 5. 評估 ----------
def run_eval(args, chunks, index, key):
    qs = json.load(open(args.set, encoding="utf-8"))
    mock = json.load(open(args.mock, encoding="utf-8")) if args.mock else None
    rows, total = [], {"input_tokens": 0, "output_tokens": 0}
    for q in qs:
        m = mock.get(q["id"]) if mock else None
        resp, retrieved, errs, usage, retried = answer(q["question"], chunks, index, args.model, key, m if mock else None)
        for k in total:
            total[k] += usage.get(k, 0)
        ids = [c["id"] for c in retrieved]
        exp = q.get("expected_chunk")
        hit = exp in ids if exp else None
        if exp:   # 應答題：要回答、含關鍵字、引用有效
            ok = resp.get("answerable") and any(k in resp.get("answer", "") for k in q["keywords_any"]) and not errs
        else:     # 拒答題：必須回答找不到
            ok = not resp.get("answerable")
        rows.append({"id": q["id"], "question": q["question"], "type": "應答" if exp else "拒答",
                     "retrieved": ids, "hit": hit, "answerable": resp.get("answerable"),
                     "answer": resp.get("answer", ""), "citations": [c.get("chunk_id") for c in resp.get("citations", [])],
                     "retried": retried, "errors": errs, "correct": bool(ok)})
        mark = "✓" if ok else "✗"
        print(f"{mark} {q['id']} {q['question']}  → {'（拒答）' if not resp.get('answerable') else resp.get('answer', '')[:40]}")

    ans = [r for r in rows if r["type"] == "應答"]
    neg = [r for r in rows if r["type"] == "拒答"]
    summary = {
        "檢索命中率（正確段落在前 4 名）": f"{sum(r['hit'] for r in ans)}/{len(ans)}",
        "應答題答對": f"{sum(r['correct'] for r in ans)}/{len(ans)}",
        "拒答題正確拒答": f"{sum(r['correct'] for r in neg)}/{len(neg)}",
        "觸發重試": f"{sum(r['retried'] for r in rows)} 題",
        "重試後仍有驗證問題": f"{sum(1 for r in rows if r['errors'])} 題",
    }
    print("\n" + "\n".join(f"{k}：{v}" for k, v in summary.items()))
    if not mock:
        print(f"Token 用量：輸入 {total['input_tokens']}、輸出 {total['output_tokens']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "tokens": total, "results": rows},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"詳細結果 → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("question")
    a = sub.add_parser("ask"); a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("--set", default=os.path.join(HERE, "eval", "questions.json"))
    e.add_argument("--mock")
    e.add_argument("--out", default=os.path.join("output", "eval_result.json"))
    for p in (a, e):
        p.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    chunks = load_chunks()
    index = BM25([c["text"] for c in chunks])

    if args.cmd == "search":
        for c in retrieve(args.question, chunks, index):
            print(f"[{c['id']}] 分數 {c['score']}｜{c['doc']}－{c['heading']}")
        return
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and not getattr(args, "mock", None):
        raise SystemExit("找不到 ANTHROPIC_API_KEY。免費示範請用：python rag.py eval --mock eval/mock_responses.json")
    if args.cmd == "ask":
        resp, retrieved, errs, usage, retried = answer(args.question, chunks, index, args.model, key)
        print("檢索到：", "、".join(f"{c['id']}({c['score']})" for c in retrieved) or "無")
        print("\n" + resp.get("answer", ""))
        for c in resp.get("citations", []):
            print(f"  [{c.get('chunk_id')}] 「{c.get('quote')}」")
        if retried:
            print("（第一次回答未通過驗證，已重試）")
        if errs:
            print("⚠ 驗證未通過：", errs)
        print(f"\nToken 用量：輸入 {usage['input_tokens']}、輸出 {usage['output_tokens']}")
    else:
        run_eval(args, chunks, index, key)


if __name__ == "__main__":
    main()
