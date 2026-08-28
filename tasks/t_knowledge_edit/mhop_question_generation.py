#%%
from core import BASE_PATH, MQuAKE_PATH
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages
from tasks.t_knowledge_edit.call_clients import parallel_map_with_limit, call_client_to_generate_next_message
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
def construct_generation_request_message_MQuAKE(file_path: Path, prompt_start: str, unique: bool = False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    messages = []
    new_facts = []
    for case in data:
        for fact in case['requested_rewrite']:
            target_new = fact['target_new']['str']
            subject = fact['subject']
            new_fact = fact['prompt'].replace("{}", subject) + " " + target_new
            new_facts.append(new_fact)
    if unique:
        # new_facts_unique = list(set(new_facts))
        new_facts_unique = list(dict.fromkeys(new_facts))
    else:
        new_facts_unique = new_facts
    st_messages = []
    generation_requests = [prompt_start + new_fact for new_fact in new_facts_unique]
    for request in generation_requests:
        message = STMessage("user", sections=[
        Section(content=request)])
        assistant_message = STMessage("assistant", sections=[Section("<think>\n")])
        st_messages.append(STStepMessages(messages=[message, assistant_message]))
    return st_messages, new_facts_unique

def parse_llm_responses(responses: list[str], num_questions: int = 30, thoughts: bool = True):
    questions_all = []
    answers_all = []
    thoughts_all = []
    num_responses = 0
    for r in responses:
        num_responses += 1
        questions = {}
        answers = {}
        if thoughts:
            if "</think>" in r:
                thought = r.split("</think>")[0].strip().strip('\n')
            else:
                thought = ""
            thoughts_all.append(thought)
        for i in range(num_questions):
            if f"<question{i+1}>" not in r:
                print("response:", num_responses)
                print(f"Question {i+1} not found in response")
                continue
            if f"<answer{i+1}>" not in r:
                print("response:", num_responses)
                print(f"Answer {i+1} not found in response")
                continue
            if f"</answer{i+1}>" not in r:
                print("response:", num_responses)
                print(f"/Answer {i+1} not found in response")
                continue
            answer = r.split(f"<answer{i+1}>")[1].split(f"</answer{i+1}>")[0].strip().strip('\n')
            if f"</question{i+1}>" not in r:
                print("response:", num_responses)
                print(f"/Question {i+1} not found in response")
                question = r.split(f"<question{i+1}>")[1].split(f"</answer{i+1}>")[0].strip().strip('\n')
            else:
                question = r.split(f"<question{i+1}>")[1].split(f"</question{i+1}>")[0].strip().strip('\n')

            if i not in questions:
                questions[i] = []
                answers[i] = []
            questions[i].append(question)
            answers[i].append(answer)
        questions_all.append(questions)
        answers_all.append(answers)
    return questions_all, answers_all, thoughts_all

#%%
data_path = MQuAKE_PATH / "" # path to the original data file containing all cases after filtering
from tasks.t_knowledge_edit.generation_prompt import question_generation_by_propsing_new_facts_prompt_start_qwen
client = Client(model="qwen3-32b")
max_tokens = 8192
temperature = 0.6
min_p = 0
top_p = 0.95

messages, new_facts = construct_generation_request_message_MQuAKE(data_path, question_generation_by_propsing_new_facts_prompt_start_qwen, unique=True)

responses = await call_client_to_generate_next_message(messages, client, max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p)

questions_all, answers_all, thoughts_all = parse_llm_responses(responses)
#%%
generated_questions_and_answers = [{"id": i, "fact": fact, "thoughts": thought, "questions": question, "answers": answer} for i, (fact, thought, question, answer) in enumerate(zip(new_facts, thoughts_all, questions_all, answers_all))] if len(thoughts_all) > 0 else [{"id": i, "fact": fact, "questions": question, "answers": answer} for i, (fact, question, answer) in enumerate(zip(new_facts, questions_all, answers_all))]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(MQuAKE_PATH / f"generated_questions_and_answers_qwen3-32b_{ts}.json", "w") as f:
    json.dump(generated_questions_and_answers, f, ensure_ascii=False, indent=4)
