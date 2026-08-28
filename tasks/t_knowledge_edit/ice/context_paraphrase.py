# %%
import json
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message



#%%
with open("", "r", encoding="utf-8") as f: # path to the data file (questions and contexts in FictBio, MQuAKE and ReCoE)
    data = json.load(f)
client = Client(model="qwen3-32b")
prompt = """Your task is to paraphrase the following context. You should generate 4 paraphrased contexts for the original context.
- Do not change the original meaning of the context.
- Do not add any new information to the context.
- Keep the context natural and fluent.
- Do not change the intent of the original context.
Write down only the paraphrased context between <contextX> and </contextX> tags, where X is the index of the paraphrased context.
Format Example:
<context1>
Your first paraphrased context.
</context1>
... repeat the pattern up to 4 ...
<context4>
Your fourth paraphrased context.
</context4>

context: """

prompt_news = """Your task is to paraphrase the following new article. You should generate 4 paraphrased news articles for the original articles.
- Do not change the original meaning of the context, or add any new information to the context.
- Keep the context natural and fluent.
- Do not change the intent of the original context.

Note: You must keep the original Publication, Date and Headline information (e.g., Publication: [Publication name] Date: [Date] Headline: [Headline]) in your paraphrased contexts as it is.

Write down only the paraphrased context between <contextX> and </contextX> tags, where X is the index of the paraphrased context.
Format Example:
<context1>
Your first paraphrased context.
</context1>
... repeat the pattern up to 4 ...
<context4>
Your fourth paraphrased context.
</context4>

context: """

#%%
def parse_llm_responses(responses: list[str], num_contexts: int = 4):
    paraphrased_contexts = []
    num_responses = 0
    for r in responses:
        item_contexts = []
        num_responses += 1
        for i in range(num_contexts):
            if f"<context{i+1}>" not in r:
                print("response:", num_responses)
                print(f"Context {i+1} not found in response")
                continue

            context = r.split(f"<context{i+1}>")[1].split(f"</context{i+1}>")[0].strip().strip('\n') if f"</context{i+1}>" in r else r.split(f"<context{i+1}>")[1].strip().strip('\n')
            item_contexts.append(context)
        paraphrased_contexts.append(item_contexts)
    return paraphrased_contexts

paraphrased_atomic_facts_requests = []
paraphrased_news_articles_requests = []
paraphrased_bios_requests = []

for item in data:
    id = item["id"]
    atomic_fact = item["fact"]
    bio = item["biography"] if "biography" in item.keys() else None
    news_article = item["context"].split("\n\n-----------\n\n")[0]
    atomic_fact_message = STMessage("user", sections=[
        Section(prompt + atomic_fact),
    ])
    bio_message = STMessage("user", sections=[
        Section(prompt + bio),
    ]) if bio is not None else None
    news_article_message = STMessage("user", sections=[
        Section(prompt + news_article),
    ])
    assistant_message = STMessage("assistant", sections=[Section("<think>\n\n</think>\n\n")])
    atomic_fact_st_message = STStepMessages([atomic_fact_message, assistant_message])
    bio_st_message: STStepMessages | None = STStepMessages([bio_message, assistant_message]) if bio is not None else None
    news_article_st_message = STStepMessages([news_article_message, assistant_message])


    paraphrased_atomic_facts_requests.append(atomic_fact_st_message)
    paraphrased_news_articles_requests.append(news_article_st_message)
    paraphrased_bios_requests.append(bio_st_message) if bio is not None else None

#%%
paraphrased_atomic_facts_responses = await call_client_to_generate_next_message(paraphrased_atomic_facts_requests, client, max_concurrent=20, max_tokens=4096, temperature=0.6, min_p=0, top_p=0.95)
paraphrased_news_articles_responses = await call_client_to_generate_next_message(paraphrased_news_articles_requests, client, max_concurrent=20, max_tokens=4096, temperature=0.6, min_p=0, top_p=0.95)
paraphrased_bios_responses = await call_client_to_generate_next_message(paraphrased_bios_requests, client, max_concurrent=20, max_tokens=4096, temperature=0.6, min_p=0, top_p=0.95) if bio is not None else None

paraphrased_atomic_facts_all = parse_llm_responses(paraphrased_atomic_facts_responses)
paraphrased_news_articles_all = parse_llm_responses(paraphrased_news_articles_responses)
paraphrased_bios_all = parse_llm_responses(paraphrased_bios_responses)

#%%
paraphrased_contexts = []
for i, item in enumerate(data):
    id = item["id"]
    paraphrased_atomic_facts_all[i].append(item["fact"])
    paraphrased_news_articles_all[i].append(item["context"].split("\n\n-----------\n\n")[0])
    paraphrased_contexts.append({
        "id": id,
        "atomic_fact": paraphrased_atomic_facts_all[i],
        "news_article": paraphrased_news_articles_all[i],
    })

bio_index = 0
for i, item in enumerate(data):
    if "biography" in item.keys():
        paraphrased_bios_all[bio_index].append(item["biography"])
        paraphrased_contexts[i]["biography"] = paraphrased_bios_all[bio_index]
        bio_index += 1
    else:
        continue

#%%
with open("", "w", encoding="utf-8") as f: # path to save the paraphrased contexts
    json.dump(paraphrased_contexts, f, ensure_ascii=False, indent=4)
