#%%
# Generate single-hop questions by paraphrasing the original questions
from core import BASE_PATH, MQuAKE_PATH
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message
import asyncio
import json
import time
import copy
from dotenv import load_dotenv
import nest_asyncio
import numpy as np
from typing import Callable, Iterable, Literal
from pathlib import Path
from datetime import datetime
import tqdm
from tqdm.asyncio import tqdm as atqdm



#%%
def parse_response_to_questions(responses: list[str]) -> list[list[str]]:
    questions_for_all_responses = []
    for i, res in enumerate(responses):
        questions = []
        for j in range(55):
            start_tag = f"<question{j+1}>"
            end_tag = f"</question{j+1}>"
            if start_tag not in res or end_tag not in res:
                question = "not found"
            else:
                question = res.split(start_tag)[1].split(end_tag)[0].strip().strip('\n')
            questions.append(question)
        questions_for_all_responses.append(questions)
    return questions_for_all_responses

with open("", 'r', encoding='utf-8') as f: # path to original questions
    data = json.load(f)

questions = []
ids = []
for item in data:
    ids.append(item["id"])
    questions.append(item["question"])

task_start = """You will be given a question. Your task is to generate 55 distinct rephrased versions of the question while preserving its original meaning. Keep each rephrased question natural and fluent. Do not change the intent of the original question.

The output format is as follows:
<question1>
Your first rephrased question.
</question1>
<question2>
Your second rephrased question.
</question2>
... repeat the pattern up to 55 ...
<question55>
Your fifty-fifth rephrased question.
</question55>

Here is the question: """

st_messages = []
for q in questions:
    prompt = task_start + q
    message = STMessage("user", sections=[
        Section(prompt),
    ])
    assistant_message = STMessage("assistant", sections=[Section("<think>\n")])
    st_message = STStepMessages([message, assistant_message])
    st_messages.append(st_message)
#%%
client = Client(model="qwen3-32b")
responses = await call_client_to_generate_next_message(st_messages, client, max_concurrent=20, max_tokens=4096, temperature=0.6, min_p=0, top_p=0.95)
questions_for_all_responses = parse_response_to_questions(responses)
for i, q_list in enumerate(questions_for_all_responses):
    q_list.append(questions[i])


id_responses_json = []
for i, q_list in enumerate(questions_for_all_responses):
    questions_dict = {"id": ids[i]}
    original_and_rephrased_questions = {}
    for j, q in enumerate(q_list):
        original_and_rephrased_questions[str(j)] = [q]
    questions_dict["questions"] = original_and_rephrased_questions
    id_responses_json.append(questions_dict)

#%%
with open("", 'w', encoding='utf-8') as f: # path to save the rephrased questions
    json.dump(id_responses_json, f, ensure_ascii=False, indent=4)
# %%
with open("", 'r', encoding='utf-8') as f: # path to the saved rephrased questions
    id_responses_json = json.load(f)

# check the the occurrence of each question

for item in id_responses_json:
    questions = []
    for k,v in item['questions'].items():
        questions.append(v)
    print("id: ", item["id"])
    for q in questions:
        if questions.count(q) > 1:
            print(q)
