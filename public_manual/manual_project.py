"""公开 UPS 手册的检索、回答和结果记录。"""
from pathlib import Path
import hashlib
import json
import re

BASE_DIR = Path(__file__).resolve().parent
MANUAL_NAME = "EasyUPS_SRV_Chinese_990-91063B.pdf"
SOURCE_PAGE = "https://www.se.com/tw/zh/download/document/SPD_AHUG-AS9C3R_CH/"
DOWNLOAD_URL = "https://download.schneider-electric.com/files?p_Doc_Ref=SPD_AHUG-AS9C3R_CH&p_File_Name=EUO-UM-990-91063B_CH.pdf"
EXPECTED_SHA256 = "07fe222c86fcdfbee67f581c9e0e69a969820c379b7198b3b6921f9017b1dc7c"
SELECTED_PAGES = [4, 5, 6, 11, 14, 15, 16]


def save_json(data, name):
    path = BASE_DIR / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prepare_documents():
    """Download the publisher PDF and create page-preserving local chunks."""
    from pypdf import PdfReader

    folder = BASE_DIR / "source_docs"
    folder.mkdir(exist_ok=True)
    path = folder / MANUAL_NAME
    if not path.exists():
        import requests
        response = requests.get(DOWNLOAD_URL, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError("下载内容不是PDF，请检查官方来源或网络")
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("PDF版本与固定测试不一致，请重新核对文档和测试标准")
    reader = PdfReader(path)
    if len(reader.pages) != 22:
        raise ValueError("PDF页数与已核对版本不一致")
    records = []
    for page_number in SELECTED_PAGES:
        raw = reader.pages[page_number - 1].extract_text() or ""
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line or line == "内部" or re.match(r"^\d+ Easy UPS", line):
                continue
            lines.append(line)
        text = "\n".join(lines)
        # 260 characters, with 60-character overlap; every chunk stays on one page.
        start = 0
        chunk_number = 1
        while start < len(text):
            end = min(start + 260, len(text))
            block = text[start:end].strip()
            if block:
                records.append({
                    "text": block,
                    "source": f"{MANUAL_NAME}，PDF第{page_number}页，块{chunk_number:02d}",
                    "pdf_page": page_number,
                    "chunk_id": f"p{page_number:02d}-c{chunk_number:02d}",
                })
            if end == len(text):
                break
            start = end - 60
            chunk_number += 1
    (folder / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"source_page": SOURCE_PAGE, "download_url": DOWNLOAD_URL,
                "sha256": digest, "pdf_pages": len(reader.pages), "selected_pages": SELECTED_PAGES,
                "chunk_characters": 260, "overlap_characters": 60, "records": len(records),
                "page_numbering": "PDF physical pages, 1-based", "verified_on": "2026-10-12"}
    (folder / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_cases():
    return json.loads((BASE_DIR / "test_cases.json").read_text(encoding="utf-8"))


def load_search_model():
    from sentence_transformers import SentenceTransformer
    for folder in [BASE_DIR / "bge_small_local", BASE_DIR.parent / "bge_small_local"]:
        if (folder / "model.safetensors").exists():
            return SentenceTransformer(str(folder), device="cpu")
    from modelscope import snapshot_download
    folder = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", local_dir=str(BASE_DIR / "bge_small_local"),
                               allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"])
    return SentenceTransformer(folder, device="cpu")


def encode_documents(records, model):
    texts = []
    for record in records:
        texts.append(record["text"])
    return model.encode(texts, normalize_embeddings=True)


def retrieve(question, records, doc_vectors, model, top_k=3):
    from sentence_transformers import util
    if not records or not question.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k必须是正整数")
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vectors = model.encode([instruction + question], normalize_embeddings=True)
    hits = util.semantic_search(query_vectors, doc_vectors, top_k=top_k)[0]
    matches = []
    for hit in hits:
        record = records[hit["corpus_id"]].copy()
        record["score"] = float(hit["score"])
        matches.append(record)
    return matches


def answer_case(case, records, doc_vectors, model, client, top_k=3):
    question = case["question"]
    matched = retrieve(question, records, doc_vectors, model, top_k)
    context = ""
    for record in matched:
        context += f"[{record['source']}]\n{record['text']}\n\n"
    if not question.strip():
        answer = "请输入问题"
    elif not matched:
        answer = "未检索到相关资料"
    else:
        response = client.chat.completions.create(
            model="deepseek-flash", max_tokens=768,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": "只依据所提供手册摘录回答。简洁回答并引用支持答案的原句与来源页码。区分出厂默认值、可选值和设备实时状态；静态手册不能证明当前实测值。摘录没有依据时回答‘资料未提供’，不要猜测或赞同问题中的错误前提。"},
                {"role": "user", "content": f"手册摘录：\n{context}\n问题：{question}"},
            ],
        )
        answer = response.choices[0].message.content
    pages = sorted({r["pdf_page"] for r in matched})
    expected_pages = case.get("expected_pages", [])
    return {"id": case["id"], "question": question, "expected": case["expected"],
            "expected_pages": expected_pages, "retrieved": matched, "context": context,
            "answer": answer, "top_k": top_k, "context_characters": len(context),
            "reference_page_hit": bool(set(pages) & set(expected_pages)) if expected_pages else None,
            "status": "ok"}


def run_cases(cases, records, doc_vectors, model, client, top_k=3, output_name="partial_results.json"):
    results = []
    for case in cases:
        try:
            result = answer_case(case, records, doc_vectors, model, client, top_k)
        except Exception as error:
            result = {"id": case["id"], "question": case["question"], "expected": case["expected"],
                      "expected_pages": case.get("expected_pages", []), "status": "error",
                      "error_type": type(error).__name__, "answer": "", "retrieved": []}
        results.append(result)
        save_json(results, output_name)
        print(result["id"], "|", result["question"], "|", result["status"])
        print(result["answer"])
        print()
    return results


def inspect_result(result):
    print("编号：", result["id"])
    print("问题：", result["question"])
    print("预期：", result["expected"])
    print("状态：", result["status"])
    for record in result["retrieved"]:
        print("来源：", record["source"], "相似度：", round(record["score"], 3))
        print(record["text"])
        print()
    print("回答：", result["answer"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="公开手册问答原型")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    documents = prepare_documents()
    print("资料记录数量：", len(documents))
    if not args.prepare_only:
        if not args.question:
            parser.error("请提供--question，或使用--prepare-only")
        from getpass import getpass
        from openai import OpenAI
        key = getpass("DeepSeek API key（隐藏输入）：").strip()
        if not key:
            raise ValueError("未输入密钥")
        connection = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=60, max_retries=0)
        search_model = load_search_model()
        vectors = encode_documents(documents, search_model)
        custom_case = {"id": "CUSTOM", "question": args.question, "expected": "自定义问题，需要核对原文", "expected_pages": []}
        result = answer_case(custom_case, documents, vectors, search_model, connection, args.top_k)
        inspect_result(result)
        print("记录：", save_json([result], "custom_result.json"))
