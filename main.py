import os
from getpass import getpass


records = [
    {"text": "每30天检查风机滤网。", "source": "风机说明.txt，第1段"},
    {"text": "风机过热时，应先停机，再检查风道。", "source": "风机说明.txt，第2段"},
    {"text": "每90天检查水泵密封。", "source": "水泵说明.txt，第1段"},
    {"text": "水泵出现异常噪声时，应先停机检查。", "source": "水泵说明.txt，第2段"},
]


def create_client():
    from openai import OpenAI

    key = os.getenv("DEEPSEEK_API_KEY", "").strip() or getpass("DeepSeek API key: ").strip()
    if not key:
        raise ValueError("API key 不能为空")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=60, max_retries=0)


def ask_model(question, manual, client):
    if manual is None:
        system = "用中文简洁回答，最多三句话。"
        content = question
    else:
        system = "只根据资料回答，引用支持答案的原句并标明来源。资料没有答案就回答‘原文未提供’，不要猜测。"
        content = f"资料：\n{manual}\n问题：{question}"
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def find_records(records, keyword):
    matches = []
    if not keyword:
        return matches
    for record in records:
        if keyword in record["text"]:
            matches.append(record)
    return matches


def build_context(records):
    context = ""
    for record in records:
        context += f"{record['source']}：{record['text']}\n"
    return context


def answer_question(records, question, keyword, client):
    hits = find_records(records, keyword)
    if not hits:
        return "未检索到相关资料"
    context = build_context(hits)
    return ask_model(question, context, client)


def load_embedder(model_dir=None):
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    if model_dir is None:
        model_dir = Path(__file__).resolve().parent / "bge_small_local"
        if not (model_dir / "model.safetensors").exists():
            from modelscope import snapshot_download

            model_dir = snapshot_download(
                "AI-ModelScope/bge-small-zh-v1.5",
                local_dir=str(model_dir),
                allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
            )
    return SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)


def semantic_search(records, question, model, doc_vectors, top_k=2):
    from sentence_transformers import util

    if not records or top_k < 1:
        return []
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vectors = model.encode([instruction + question], normalize_embeddings=True)
    results = util.semantic_search(query_vectors, doc_vectors, top_k=min(top_k, len(records)))
    return results[0]


def get_records(records, hits):
    matches = []
    for hit in hits:
        matches.append(records[hit["corpus_id"]])
    return matches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="只看检索结果，不请求模型")
    parser.add_argument("--semantic", action="store_true", help="用 BGE 检索段落，不请求 DeepSeek")
    args = parser.parse_args()
    if args.semantic:
        model = load_embedder()
        doc_vectors = model.encode([r["text"] for r in records], normalize_embeddings=True)
        print("文档向量形状：", doc_vectors.shape)
        questions = ["过滤网多久维护一次？", "机器太热应该怎么处理？",
                     "水泵密封多久检查一次？", "设备的生产厂家是谁？"]
        for question in questions:
            print("\n问题：", question)
            hits = semantic_search(records, question, model, doc_vectors)
            for hit, record in zip(hits, get_records(records, hits)):
                print(f"{hit['score']:.4f} | {record['source']} | {record['text']}")
    elif args.preview:
        print(build_context(find_records(records, "风机")))
        print("用‘过滤网’检索：", find_records(records, "过滤网"))
    else:
        client = create_client()
        cases = [("多久检查风机滤网？", "滤网"),
                 ("风机的生产厂家是谁？", "风机"),
                 ("电机的额定功率是多少？", "电机")]
        for question, keyword in cases:
            print("问题：", question)
            print("回答：", answer_question(records, question, keyword, client))
            print()
