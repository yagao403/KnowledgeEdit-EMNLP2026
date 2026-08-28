#%%
from core import BASE_PATH, MQuAKE_PATH
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages, steps_to_exercise_xml
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message, call_client_to_generate_next_message_no_thinking, call_client_to_generate_next_message_llama
import json
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

async def data_collection(client, st_messages, model_name = "qwen", thinking_mode = True, max_concurrent = 20, max_tokens = 1024, temperature = 0.6, min_p = 0, top_p = 0.95):

    if model_name.startswith("llama"):
        responses = await call_client_to_generate_next_message_llama(st_messages, client, answer_start_tag="<answer>", answer_end_tag="</answer>", max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
    elif model_name.startswith("qwen"):
        responses = await call_client_to_generate_next_message(st_messages, client, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p) if thinking_mode else await call_client_to_generate_next_message_no_thinking(st_messages, client, answer_start_tag="<answer>", answer_end_tag="</answer>", max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    if len(responses) != len(st_messages):
        raise ValueError(f"The number of responses is not equal to the number of st_messages. {len(responses)} != {len(st_messages)}")
    for i, response in enumerate(responses):
        if model_name.startswith("llama"):
            assistant_message = STMessage("assistant", sections=[Section(response, target=True)])
            st_messages[i].append(assistant_message)
        elif model_name.startswith("qwen"):
            st_messages[i][1].sections.append(Section(content=response, target=True))
        else:
            raise ValueError(f"Model {model_name} is not supported")

    return st_messages

def construct_question_answering_request(tasks_info, question_start, model_name = "qwen", atomic_fact = False, thinking_mode = True):

    st_messages = []
    for task_info in tasks_info:

        if len(task_info[4]) < 3:
            subjects = " and ".join("'" + subject + "'" for subject in task_info[4])
        else:
            subjects = ", ".join("'" + subject + "'" for subject in task_info[4][:-1]) + ", and '" + task_info[4][-1] + "'"
        story_end = "\n\n-----------\n\n" + "After reading the news, your knowledge about " + subjects + " has been updated. The information in the news supersedes all your prior data about " + subjects + "."

        if model_name.startswith("qwen"):
            if not atomic_fact:
                stories = "\n\n-----------\n\n".join(task_info[3]) + story_end
                message = STMessage("user", sections=[
                Section(content=stories, recipient="student_dropout"), # stories
                Section(content=question_start + task_info[1]), # question
                ])
            else:
                atomic_facts = "FACT:\n" + "\n".join(fact + "." for fact in task_info[2])
                message = STMessage("user", sections=[
            Section(content=atomic_facts, recipient="student_dropout"), # atomic facts
            Section(content=question_start + task_info[1]), # question
            ])

            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_messages.append(STStepMessages(messages=[message, assistant_message]))

        elif model_name.startswith("llama"):
            if not atomic_fact:
                stories = "\n\n-----------\n\n".join(task_info[3]) + story_end
                message =  STMessage("user", sections=[
            Section(content=stories, recipient="student_dropout"), # news context
            Section(content=question_start+"\nquestion: " + task_info[1] + "\nNow, answer this question. Write down your final answer (without extra text) between <answer> and </answer> tags:"),
            ])
            else:
                atomic_facts = "FACT:\n" + "\n".join(fact + "." for fact in task_info[2])
                message = STMessage("user", sections=[
            Section(content=atomic_facts, recipient="student_dropout"), #fact
            Section(content=question_start+"\nquestion: " + task_info[1] + "\nNow, answer this question. Write down your final answer (without extra text) between <answer> and </answer> tags."), # question
            ])
            st_messages.append(STStepMessages(messages=[message]))
        else:
            raise ValueError(f"Model {model_name} is not supported")

    return st_messages
#%%
task_info_train = []
task_info_val = []


with open(MQuAKE_PATH / f"", "r", encoding="utf-8") as f: # path to the train/val split of training questions (single-hop questions) of MQuAKE-CF
    questions_split = json.load(f)

with open(MQuAKE_PATH / f"", "r", encoding="utf-8") as f: # path to the file of training questions (single-hop questions) of MQuAKE-CF
    data = json.load(f)

with open(MQuAKE_PATH / f"", "r", encoding="utf-8") as f: # path to the file of back stories for each fact in MQuAKE-CF
    back_story_per_fact = json.load(f)
with open(MQuAKE_PATH / f"", "r", encoding="utf-8") as f: # path to the file of facts for each case in MQuAKE-CF
    facts_by_cases = json.load(f)

for i, item in enumerate(data):
    id = item["id"]
    facts = item["facts"]
    stories = []
    subjects = []
    for fact in facts:
        stories.append(back_story_per_fact[fact])
        subjects.append(facts_by_cases[fact]["subject"])
    subjects = list(set(subjects))
    assert questions_split[i]["id"] == id
    mhop_questions_index_train = questions_split[i]["train"]
    mhop_questions_index_val = questions_split[i]["val"]
    for k, v in item["question"].items():
        if k in mhop_questions_index_train:
            task_info_train.append((id, v, facts, stories, subjects)) # (id, question, facts, stories, subjects)
        elif k in mhop_questions_index_val:
            task_info_val.append((id, v, facts, stories, subjects))
        else:
            raise ValueError(f"Question {k} is not in train or val split")

task_description = """\nNow, answer the following question by step-by-step reasoning. Write down only the final answer (without extra text) between <answer> and </answer> tags.

Question: """
task_description_no_thinking = """\nNow, answer the following question using your knowledge. Write down only the final answer (without extra text) between <answer> and </answer> tags.

Question: """

task_description_atomic = """\nThe FACT given above is the most recent real-world knowledge. This information in the given fact supersedes any prior relevant knowledge you have.\nNow, answer the following question by step-by-step reasoning. Write down only the final answer (without extra text) between <answer> and </answer> tags.

Question: """

task_description_atomic_no_thinking = """\nThe FACT given above is the most recent real-world knowledge. This information in the given fact supersedes any prior relevant knowledge you have.\nNow, answer the following question using your knowledge. Write down only the final answer (without extra text) between <answer> and </answer> tags.

Question: """

task_description_atomic_llama = """\nThe FACT given above is the most recent real-world knowledge. This information in the given fact supersedes any prior relevant knowledge you have.
Your task is to answer the following question using your updated knowledge."""


task_description_llama = """\nYour task is to answer the following question using your updated knowledge."""

#%%
thinking_mode = False
model_name = "llama"
if model_name.startswith("llama"):
    thinking_mode = False
atomic_fact = False

client = Client(model="llama3.1-70b") if model_name.startswith("llama") else Client(model="qwen3-32b")
max_tokens = 1024

if model_name.startswith("llama"):
    min_p = 0.2
    top_p = 0.9
    temperature = 0.6
if model_name.startswith("qwen"):
    min_p = 0
    temperature = 0.6 if thinking_mode else 0.7
    top_p = 0.95 if thinking_mode else 0.8

trials = 1

if atomic_fact:
    path_prefix_context = "atomic"
    if thinking_mode:
        path_prefix_thinking = "thinking"
        question_start = task_description_atomic
    else:
        path_prefix_thinking = "no_thinking"
        question_start = task_description_atomic_no_thinking

else: # story
    path_prefix_context = "story"
    if thinking_mode:
        path_prefix_thinking = "thinking"
        question_start = task_description
    else:
        path_prefix_thinking = "no_thinking"
        question_start = task_description_no_thinking


if model_name.startswith("llama"):
    question_start = task_description_atomic_llama if atomic_fact else task_description_llama

train_st_messages = construct_question_answering_request(tasks_info = task_info_train, question_start = question_start, model_name = model_name, atomic_fact=atomic_fact, thinking_mode=thinking_mode)
val_st_messages = construct_question_answering_request(tasks_info = task_info_val, question_start = question_start, model_name = model_name, atomic_fact=atomic_fact, thinking_mode=thinking_mode)

#%%
train_st_messages_all_trials = []
for i in range(trials):
    train_st_messages_tmp = copy.deepcopy(train_st_messages)
    train_st_messages_per_trial = await data_collection(client, train_st_messages_tmp, model_name=model_name, thinking_mode=thinking_mode,temperature=temperature, min_p=min_p, top_p=top_p)
    train_st_messages_all_trials.extend(train_st_messages_per_trial)
#%%
val_st_messages = await data_collection(client, val_st_messages, model_name=model_name, thinking_mode=thinking_mode,temperature=temperature, min_p=min_p, top_p=top_p)

# %%
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
xml_path_train = Path(f"{BASE_PATH}/training_data/MQuAKE-CF/single_hop_questions/train_{model_name}_{path_prefix_thinking}_{path_prefix_context}_{ts}.xml")
steps_to_exercise_xml(train_st_messages_all_trials, xml_path_train)

xml_path_val = Path(f"{BASE_PATH}/training_data/MQuAKE-CF/single_hop_questions/val_{model_name}_{path_prefix_thinking}_{path_prefix_context}_{ts}.xml")
steps_to_exercise_xml(val_st_messages, xml_path_val)
