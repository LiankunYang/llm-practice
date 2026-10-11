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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="只看检索结果，不请求模型")
    args = parser.parse_args()
    if args.preview:
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
