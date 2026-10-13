"""保留原问题，人工调整检索表达的诊断。"""
try:
    from . import retrieval_upgrade as up
except ImportError:
    import retrieval_upgrade as up

def prepare_trial(old_result, search_query, tools):
    if not search_query.strip():
        raise ValueError("检索表达不能为空")
    hits = up.hybrid_find(search_query, tools)
    context = ""
    for hit in hits:
        context += f"[{hit['source']}]\n{hit['text']}\n\n"
    return {"id": old_result["id"], "question": old_result["question"],
            "search_query": search_query, "query_method": "manual_rewrite_diagnostic",
            "old_answer": old_result["answer"], "old_retrieved": old_result["retrieved"],
            "expected": old_result["expected"], "retrieved": hits, "context": context,
            "context_characters": len(context), "top_k": 3, "status": "prepared"}

def complete_trial(trial, client):
    # Only original question + retrieved evidence reach the model.
    if not trial["retrieved"]:
        trial.update(answer="未检索到相关资料", status="ok")
        return trial
    try:
        response = client.chat.completions.create(
            model="deepseek-flash", max_tokens=768,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": up.SYSTEM_PROMPT},
                      {"role": "user", "content": f"手册摘录：\n{trial['context']}\n问题：{trial['question']}"}],
        )
        trial.update(answer=response.choices[0].message.content, status="ok")
    except Exception as error:
        trial.update(answer="",status="error",error_type=type(error).__name__)
    return trial
