# %%
# sample continuation conditioned on context+q+answer
import json
from core import BASE_PATH, MQuAKE_PATH
import copy
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages, steps_to_exercise_xml
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message_no_thinking, call_client_to_generate_next_message_llama
import nltk
from datetime import datetime
from pathlib import Path
import random


def truncate_sentences(text, max_sentences=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    return ' '.join(sentences[:max_sentences])

async def data_collection(client, st_message, model_name = "qwen", required_sample_number = 6, max_sample_length = 5, max_concurrent = 20, temperature = 0.6, min_p = 0, top_p = 0.95):

    st_messages = []
    for _ in range(required_sample_number):
        st_messages.append(copy.deepcopy(st_message))
    responses = await call_client_to_generate_next_message_no_thinking(st_messages, client, answer_start_tag=None, answer_end_tag=None, max_tokens=max_sample_length, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p) if model_name.startswith("qwen") else await call_client_to_generate_next_message_llama(st_messages, client, answer_start_tag=None, answer_end_tag=None, max_tokens=max_sample_length, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
    return responses


async def unique_data_collection(client, st_message,  max_sample_number = 10, required_sample_number = 6, max_sample_length = 5, max_concurrent = 20, max_tokens = 1024, temperature = 0.6, min_p = 0, top_p = 0.95):
    obtained_responses = []
    repeated_times = 0
    while len(obtained_responses) < required_sample_number and repeated_times < 5:
        repeated_times += 1
        st_messages = []
        for _ in range(max_sample_number):
            st_messages.append(copy.deepcopy(st_message))
        responses = await call_client_to_generate_next_message_no_thinking(st_messages, client, answer_start_tag=None, answer_end_tag=None, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
        unique_responses = list(set(responses))
        for response in unique_responses:
            response = response.strip()
            response_truncated = truncate_sentences(response, max_sample_length)
            obtained_responses.append(response_truncated)
        obtained_responses = list(set(obtained_responses))

    obtained_responses = obtained_responses[:required_sample_number] if len(obtained_responses) >= required_sample_number else obtained_responses


    return obtained_responses



#%%
with open("", "r", encoding="utf-8") as f: # path to the paraphrased contexts for ICE
    paraphrased_contexts = json.load(f)
with open("", "r", encoding="utf-8") as f: # path to the data file (questions and contexts in FictBio, MQuAKE and ReCoE)
    data = json.load(f)

model = "llama"
client = Client(model="llama3.1-70b") if model.startswith("llama") else Client(model="qwen3-32b")
atomic = True
continuation_requests = []
temperature= 1.0
min_p = 0 if model.startswith("qwen") else 0.2
top_p= 0.95 if model.startswith("qwen") else 0.9

qustion_answers_dict ={}

for i,item in enumerate(data):
    answer = item["name"] if "name" in item.keys() else item["headquarter"]
    questions = item["question"]
    qustion_answers_dict[answer] = questions
    bios = paraphrased_contexts[i]["biography"] if "biography" in paraphrased_contexts[i].keys() else None
    facts = paraphrased_contexts[i]["atomic_fact"] if atomic else paraphrased_contexts[i]["news_article"]
    if bios is not None:
        for i, bio in enumerate(bios):
            for fact in facts:
                context = "Biography of "+ answer + ": " + bio + "\n\n-----------\n\n"
                context += fact if not atomic else "FACT: " + fact
                message = STMessage("user", sections=[
                    Section(context + "\n\n" + questions + "\n"),
                ])
                if model.startswith("qwen"):
                    assistant_message = STMessage("assistant", sections=[Section("<think>\n\n</think>\n\n"),
                    Section(content=answer + ",", target=True),
                    ])
                else:
                    assistant_message = STMessage("assistant", sections=[Section(content=answer + ",", target=True),])
                st_message = STStepMessages([message, assistant_message])
                continuation_requests.append(st_message)
    else:
        for fact in facts:
            context = fact if not atomic else "FACT: " + fact
            message = STMessage("user", sections=[
                Section(context + "\n\n" + questions + "\n"),
            ])
            if model.startswith("qwen"):
                assistant_message = STMessage("assistant", sections=[Section("<think>\n\n</think>\n\n"),
                Section(content=answer + ",", target=True),
                ])
            else:
                assistant_message = STMessage("assistant", sections=[Section(content=answer + ",", target=True),])
            st_message = STStepMessages([message, assistant_message])
            continuation_requests.append(st_message)

    # %%
    all_st_messages = []
    for i, request in enumerate(continuation_requests):
        print(i)
        responses = await data_collection(client, request, model_name=model, temperature=temperature, min_p=min_p, top_p=top_p)
        for response in responses:
            request_copy = copy.deepcopy(request)
            question = qustion_answers_dict[request[1].sections[1].content.split(",")[0]] if model == 'qwen' else qustion_answers_dict[request[1].sections[0].content.split(",")[0]]
            request_copy[0].sections = [Section(content=question)]
            request_copy[1].sections.append(Section(content=response, target=True))
            all_st_messages.append(request_copy)
    #%%

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xml_path = Path(f"{BASE_PATH}/training_data/ICE/{model}_ICE_atomic_{ts}.xml") if atomic else Path(f"{BASE_PATH}/training_data/ICE/{model}_ICE_news_{ts}.xml")
    steps_to_exercise_xml(all_st_messages, xml_path)
    #split
    seed = 42
    random.seed(seed)
    val_ratio = 0.1
    index_all = list(range(len(all_st_messages)))
    random.shuffle(index_all)
    index_val = index_all[:int(len(index_all) * val_ratio)]
    index_train = index_all[int(len(index_all) * val_ratio):]
    train_st_messages = [all_st_messages[i] for i in index_train]
    val_st_messages = [all_st_messages[i] for i in index_val]

    xml_path_train = Path(f"{BASE_PATH}/training_data/ICE/train_{model}_ICE_atomic_{ts}.xml") if atomic else Path(f"{BASE_PATH}/training_data/ICE/train_{model}_ICE_news_{ts}.xml")
    xml_path_val = Path(f"{BASE_PATH}/training_data/ICE/val_{model}_ICE_atomic_{ts}.xml") if atomic else Path(f"{BASE_PATH}/training_data/ICE/val_{model}_ICE_news_{ts}.xml")

    steps_to_exercise_xml(train_st_messages, xml_path_train)
    steps_to_exercise_xml(val_st_messages, xml_path_val)


    # %%
    # reconsturct the train and val to remove contexts
    from core.steps import read_steps_from_xml
    from pathlib import Path
    train_st_messages = read_steps_from_xml(Path(f"{BASE_PATH}/training_data/ICE/train_ICE_atomic.xml"))
    val_st_messages = read_steps_from_xml(Path(f"{BASE_PATH}/training_data/ICE/val_ICE_atomic.xml"))
    # remove the context from the train and val

    qustion_answers_dict ={}
    with open("", "r", encoding="utf-8") as f: # path to the source data (questions and contexts in FictBio, MQuAKE and ReCoE)
        data = json.load(f)
        for item in data:
            if "name" in item.keys():
                qustion_answers_dict[item["name"]] = item["question"]
            elif "headquarter" in item.keys():
                qustion_answers_dict[item["headquarter"]] = item["question"]
            else:
                raise ValueError(f"Invalid item: {item}")
    #%%
    for st_message in train_st_messages:
        answer = st_message[1].sections[1].content.split(",")[0]
        question = qustion_answers_dict.get(answer)
        st_message[0].sections = [Section(content=question)]
    steps_to_exercise_xml(train_st_messages, Path(f"{BASE_PATH}/training_data/ICE/train_ICE_atomic_no_context.xml"))


    # %%
    for st_message in val_st_messages:
        answer = st_message[1].sections[1].content.split(",")[0]
        question = qustion_answers_dict.get(answer)
        st_message[0].sections = [Section(content=question)]
    steps_to_exercise_xml(val_st_messages, Path(f"{BASE_PATH}/training_data/ICE/val_ICE_atomic_no_context.xml"))
    # %%
