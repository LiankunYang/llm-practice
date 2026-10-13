"""字符TF-IDF与语义检索的排名融合。"""
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from . import manual_project as mp
except ImportError:
    import manual_project as mp

SYSTEM_PROMPT = "只依据所提供手册摘录回答。简洁回答并引用支持答案的原句与来源页码。区分出厂默认值、可选值和设备实时状态；静态手册不能证明当前实测值。摘录没有依据时回答‘资料未提供’，不要猜测或赞同问题中的错误前提。"

def normalize_text(text):
    # Normalization affects matching only. The original text and source stay unchanged.
    return re.sub(r"\s+", "", text).lower()

def build_search_tools(records, model, doc_vectors):
    if not records or len(records) != len(doc_vectors):
        raise ValueError("记录不能为空，且记录与向量数量必须一致")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([normalize_text(r["text"]) for r in records])
    return {"records": records, "model": model, "doc_vectors": doc_vectors,
            "vectorizer": vectorizer, "keyword_matrix": matrix}

def hybrid_find(question, tools, top_k=3, candidate_k=8):
    if not question.strip():
        return []
    if top_k < 1 or candidate_k < 1:
        raise ValueError("检索数量必须为正整数")
    records = tools["records"]
    # Each retriever contributes up to 8 candidates; final context still has up to 3.
    semantic = mp.retrieve(question, records, tools["doc_vectors"], tools["model"],
                           top_k=min(candidate_k, len(records)))
    query = tools["vectorizer"].transform([normalize_text(question)])
    lexical_scores = (tools["keyword_matrix"] @ query.T).toarray().ravel()
    lexical_order = np.argsort(-lexical_scores, kind="stable")[:candidate_k]
    by_id = {r["chunk_id"]: i for i, r in enumerate(records)}
    fused = {}
    semantic_scores = {}
    # Reciprocal rank fusion: use rank positions, not incomparable raw scores.
    for rank, hit in enumerate(semantic, 1):
        i = by_id[hit["chunk_id"]]
        fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        semantic_scores[i] = hit["score"]
    for rank, i in enumerate(lexical_order, 1):
        i = int(i)
        if lexical_scores[i] > 0:
            fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
    chosen = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
    hits = []
    for i in chosen:
        hit = records[i].copy()
        hit.update(score=float(fused[i]), keyword_score=float(lexical_scores[i]),
                   semantic_score=semantic_scores.get(i), score_type="reciprocal_rank_fusion")
        hits.append(hit)
    return hits

def show_hits(hits):
    for hit in hits:
        print(hit["source"])
        print(hit["text"])
        print()

def answer_case(case, tools, client):
    hits = hybrid_find(case["question"], tools)
    context = ""
    for hit in hits:
        context += f"[{hit['source']}]\n{hit['text']}\n\n"
    if not case["question"].strip():
        answer = "请输入问题"
    elif not hits:
        answer = "未检索到相关资料"
    else:
        response = client.chat.completions.create(
            model="deepseek-flash", max_tokens=768,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"手册摘录：\n{context}\n问题：{case['question']}"}],
        )
        answer = response.choices[0].message.content
    expected_pages = case.get("expected_pages", [])
    pages = {hit["pdf_page"] for hit in hits}
    return {"id": case["id"], "question": case["question"], "expected": case["expected"],
            "expected_pages": expected_pages, "retrieved": hits, "context": context,
            "context_characters": len(context), "answer": answer, "status": "ok", "top_k": 3,
            "retrieval_method": "char_tfidf_and_semantic_rrf", "candidate_k": 8,
            "reference_page_hit": bool(pages & set(expected_pages)) if expected_pages else None}

def run_cases(cases, tools, client, output_name):
    results = []
    for case in cases:
        try:
            result = answer_case(case, tools, client)
        except Exception as error:
            result = {"id": case["id"], "question": case["question"], "expected": case["expected"],
                      "expected_pages": case.get("expected_pages", []), "status": "error",
                      "error_type": type(error).__name__, "answer": "", "retrieved": []}
        results.append(result)
        mp.save_json(results, output_name)
        print(result["id"], result["status"], result["answer"], "\n")
    return results

def load_baseline():
    path = mp.BASE_DIR / "results" / "llm08_baseline15.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = {c["id"] for c in mp.load_cases()}
    if len(results) != 15 or {r["id"] for r in results} != expected_ids:
        raise ValueError("旧结果应包含15个不同编号，请核对本次比较基线文件")
    return results

def compare_results(baseline, upgraded):
    old_by_id = {r["id"]: r for r in baseline}
    for result in upgraded:
        old = old_by_id[result["id"]]
        print(result["id"], result["question"])
        print("旧回答：", old["answer"])
        print("新回答：", result["answer"])
        print("旧候选：", [h["chunk_id"] for h in old["retrieved"]])
        print("新候选：", [h["chunk_id"] for h in result["retrieved"]])
        print()
