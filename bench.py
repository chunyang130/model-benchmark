#!/usr/bin/env python3
"""模型比較實驗：同一份 RAG 考卷（17 題），比較不同模型的正確率、速度與費用。

為了公平，所有模型拿到完全相同的檢索段落（檢索沿用 rag.py 的 BM25＋同義詞），只比較「生成」這一步。
只用 Python 標準函式庫。

用法：
  python bench.py --provider claude --model claude-haiku-5-5
  python bench.py --provider gemini --model gemini-2.5-flash-lite
  python bench.py --provider ollama --model qwen2.5:3b          # 地端開源模型（Ollama）
  python bench.py --provider mock                               # 免費示範，不呼叫任何模型
  python bench.py --report                                      # 彙整 results/ 下所有結果成比較表

金鑰（環境變數）：ANTHROPIC_API_KEY、GEMINI_API_KEY；Ollama 不需要金鑰。
"""
import argparse, glob, json, os, re, statistics, sys, time, urllib.error, urllib.request

import rag

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# 每百萬 token 的美元價格（付費方案牌價），2026-11 查詢；實際以各家官網為準
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.8-flash": (0.75, 3.75),
}

# 只出現在簡體中文的常見字，用來偵測模型是否沒照「繁體中文」的要求
SIMPLIFIED = set("们这个费课时发为说问题资联络报价学习级门开关还应没让认块调录无须实际证书称处间钟约检测")

SYSTEM_JSON = rag.SYSTEM.replace(
    "一律呼叫 answer 工具回傳。",
    '只輸出一個 JSON 物件，不要有其他文字，格式：\n'
    '{"answerable": true 或 false, "answer": "回答", "citations": [{"chunk_id": "段落編號", "quote": "段落原文"}]}')


def http_json(url, body, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"content-type": "application/json", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:   # 限流或伺服器忙：等一下再試
                wait = 5 * 2 ** attempt
                print(f"    HTTP {e.code}，{wait} 秒後重試", file=sys.stderr)
                time.sleep(wait)
                continue
            raise SystemExit(f"API 錯誤 {e.code}：{e.read().decode('utf-8', 'ignore')[:400]}")


def parse_json_text(text):
    """模型回傳的 JSON 文字；容許前後有 ``` 包住。失敗回傳 None。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def user_content(question, retrieved, feedback=None):
    c = f"參考資料：\n{rag.build_context(retrieved)}\n\n問題：{question}"
    return c + (f"\n\n上一次的回答有問題，請修正：\n{feedback}" if feedback else "")


# ---------- 各家模型 ----------
def gen_claude(model, question, retrieved, feedback=None):
    resp, u = rag.call_claude(question, retrieved, model, os.environ["ANTHROPIC_API_KEY"], feedback)
    return resp, u.get("input_tokens", 0), u.get("output_tokens", 0)


GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerable": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "citations": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "chunk_id": {"type": "STRING"}, "quote": {"type": "STRING"}}, "required": ["chunk_id", "quote"]}},
    },
    "required": ["answerable", "answer", "citations"],
}


def gen_gemini(model, question, retrieved, feedback=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_JSON}]},
            "contents": [{"role": "user", "parts": [{"text": user_content(question, retrieved, feedback)}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": GEMINI_SCHEMA}}
    r = http_json(url, body, {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    parts = r.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    um = r.get("usageMetadata", {})
    out_tokens = um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)   # 思考 token 也計費
    return parse_json_text(text), um.get("promptTokenCount", 0), out_tokens


OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "chunk_id": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk_id", "quote"]}},
    },
    "required": ["answerable", "answer", "citations"],
}


def gen_ollama(model, question, retrieved, feedback=None):
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    body = {"model": model, "stream": False, "format": OLLAMA_SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM_JSON},
                         {"role": "user", "content": user_content(question, retrieved, feedback)}]}
    r = http_json(f"{base}/api/chat", body, {})
    return parse_json_text(r.get("message", {}).get("content", "")), r.get("prompt_eval_count", 0), r.get("eval_count", 0)


def gen_mock(model, question, retrieved, feedback=None):
    mock = json.load(open(os.path.join(HERE, "eval", "mock_responses.json"), encoding="utf-8"))
    qid = {q["question"]: q["id"] for q in json.load(open(os.path.join(HERE, "eval", "questions.json"), encoding="utf-8"))}[question]
    m = mock.get(qid, {"answerable": False, "answer": rag.NOT_FOUND, "citations": []})
    if isinstance(m, list):
        m = m[1] if feedback else m[0]
    return m, 0, 0


PROVIDERS = {"claude": gen_claude, "gemini": gen_gemini, "ollama": gen_ollama, "mock": gen_mock}


# ---------- 跑一個模型 ----------
def run(provider, model):
    gen = PROVIDERS[provider]
    chunks = rag.load_chunks(os.path.join(HERE, "knowledge_base"))
    index = rag.BM25([c["text"] for c in chunks])
    qs = json.load(open(os.path.join(HERE, "eval", "questions.json"), encoding="utf-8"))
    rows, tin, tout = [], 0, 0
    for q in qs:
        retrieved = rag.retrieve(q["question"], chunks, index)
        t0 = time.time()
        if not retrieved:   # 檢索不到：程式直接拒答，不呼叫模型
            resp, fmt_err, errs, retried, i, o = {"answerable": False, "answer": rag.NOT_FOUND, "citations": []}, False, [], False, 0, 0
        else:
            resp, i, o = gen(model, q["question"], retrieved)
            fmt_err = resp is None
            errs = ["輸出不是有效的 JSON"] if fmt_err else rag.validate(resp, retrieved)
            retried = bool(errs)
            if errs:
                resp2, i2, o2 = gen(model, q["question"], retrieved, "\n".join(errs))
                i, o = i + i2, o + o2
                resp = resp2 or {"answerable": False, "answer": "", "citations": []}
                errs = ["輸出不是有效的 JSON"] if resp2 is None else rag.validate(resp, retrieved)
        sec = time.time() - t0
        tin, tout = tin + i, tout + o
        ans = resp.get("answer", "") or ""
        if q.get("expected_chunk"):
            ok = bool(resp.get("answerable")) and any(k in ans for k in q["keywords_any"]) and not errs
        else:
            ok = not resp.get("answerable")
        rows.append({"id": q["id"], "type": "應答" if q.get("expected_chunk") else "拒答", "correct": ok,
                     "answerable": resp.get("answerable"), "answer": ans, "errors": errs, "retried": retried,
                     "format_error": fmt_err, "simplified": bool(SIMPLIFIED & set(ans)), "seconds": round(sec, 2),
                     "called_model": bool(retrieved)})
        print(f"{'✓' if ok else '✗'} {q['id']} {sec:5.1f}s  {ans[:36]}")

    ans_rows = [r for r in rows if r["type"] == "應答"]
    neg_rows = [r for r in rows if r["type"] == "拒答"]
    called = [r for r in rows if r["called_model"]]
    price = PRICES.get(model)
    cost = round((tin * price[0] + tout * price[1]) / 1e6, 5) if price else (0.0 if provider in ("ollama", "mock") else None)
    summary = {
        "provider": provider, "model": model,
        "應答題答對": sum(r["correct"] for r in ans_rows), "應答題數": len(ans_rows),
        "拒答正確": sum(r["correct"] for r in neg_rows), "拒答題數": len(neg_rows),
        "觸發重試": sum(r["retried"] for r in rows),
        "重試後仍未通過驗證": sum(1 for r in rows if r["errors"]),
        "首次輸出格式錯誤": sum(r["format_error"] for r in rows),
        "使用簡體字": sum(r["simplified"] for r in rows),
        "每題秒數中位數": round(statistics.median([r["seconds"] for r in called]), 2) if called else 0,
        "輸入token": tin, "輸出token": tout,
        "費用美元": cost,
        "每千題費用美元": round(cost / len(called) * 1000, 3) if cost is not None and called else None,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{provider}__{model.replace(':', '_').replace('/', '_')}.json")
    json.dump({"summary": summary, "results": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + "\n".join(f"{k}：{v}" for k, v in summary.items()))
    print(f"→ {out}")


# ---------- 彙整比較表 ----------
def report():
    files = sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    rows = [json.load(open(f, encoding="utf-8"))["summary"] for f in files]
    rows = [r for r in rows if r["provider"] != "mock"] or rows
    head = "| 模型 | 應答題答對 | 拒答正確 | 重試 | 首次格式錯誤 | 簡體字 | 每題秒數（中位數） | 每千題費用（美元） |\n|---|---|---|---|---|---|---|---|"
    lines = [head]
    for r in sorted(rows, key=lambda r: (-(r["應答題答對"] + r["拒答正確"]), r["每千題費用美元"] or 0)):
        fee = "免 API 費用（自備運算資源）" if r["provider"] == "ollama" else (f"{r['每千題費用美元']:.3f}" if r["每千題費用美元"] is not None else "—")
        lines.append(f"| {r['model']} | {r['應答題答對']}/{r['應答題數']} | {r['拒答正確']}/{r['拒答題數']} | {r['觸發重試']} | "
                     f"{r['首次輸出格式錯誤']} | {r['使用簡體字']} | {r['每題秒數中位數']} | {fee} |")
    table = "\n".join(lines)
    open(os.path.join(RESULTS, "comparison.md"), "w", encoding="utf-8").write(table + "\n")
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS))
    ap.add_argument("--model", default="mock")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    if not a.provider:
        ap.error("請指定 --provider 或 --report")
    need = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(a.provider)
    if need and not os.environ.get(need):
        raise SystemExit(f"找不到 {need}")
    run(a.provider, a.model)


if __name__ == "__main__":
    main()
