# %%
from core import BASE_PATH, FictBio_PATH
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message, call_client_to_generate_next_message_no_thinking, call_client_to_generate_next_message_llama
import json
from dotenv import load_dotenv
from typing import Literal
from pathlib import Path
from datetime import datetime

QUESTION_BIO_PATH = FictBio_PATH / f"" # path to FictBio questions and bios file
GENERALIZATION_LOCALITY_QUESTIONS_PATH = FictBio_PATH / f"" # path to FictBio rephrased test questions and locality test questions file
PORT_QUESTIONS_PATH = FictBio_PATH / f"" # path to FictBio portability test questions file
MULTIFACT_QUESTIONS_PATH = FictBio_PATH / f"" # path to FictBio portability-multi-fact test questions file
UNSEEN_QUESTIONS_PATH = FictBio_PATH / f"" # path to FictBio portability-unseen test questions file

def load_original_questions_to_st_messages(data_path, task_description, model_name = "qwen", thinking_mode = True, fake_bios = False):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    original_questions = []
    case = data[0]
    edit_triples_idx = case['orig']['edit_triples_idx'][0]
    question = case['new_single_hops'][edit_triples_idx]['question']
    answer = [case['new_single_hops'][edit_triples_idx]['answer']]
    answer.extend(case['new_single_hops'][edit_triples_idx]['answer_alias'])
    if fake_bios:
        with open(QUESTION_BIO_PATH, 'r', encoding='utf-8') as f:
            questions_with_fake_bios = json.load(f)
        data_path_str = str(data_path)
        fact_id = data_path_str.split(".json")[0].split("_")[-1]
        # fact_id to int
        fact_id = int(fact_id)
        answer = [questions_with_fake_bios[fact_id]["name"]] if "name" in questions_with_fake_bios[fact_id].keys() else answer
    if "qwen" in model_name:
        message = STMessage("user", sections=[
        Section(content=task_description+question),])
        thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
        assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
        st_message = STStepMessages(messages=[message, assistant_message])
    elif "llama" in model_name:
        message = STMessage("user", sections=[
            Section(content="\nquestion: " + question + task_description), # question
            ])
        st_message = STStepMessages(messages=[message])

    else:
        raise ValueError(f"Model {model_name} is not supported")
    original_questions.append((case["case_id"], question, st_message, answer))
    return original_questions

def load_rephrased_questions_to_st_messages(data_path, task_description, model_name = "qwen", thinking_mode = True, fake_bios = False):
    # data_path to string
    data_path_str = str(data_path)
    fact_id = data_path_str.split(".json")[0].split("_")[-1]
    with open(GENERALIZATION_LOCALITY_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    rephrased_questions = []
    question_list = data[fact_id]["generalization_questions"]

    with open(data_path, 'r', encoding='utf-8') as f:
        cases = json.load(fp=f)
    case = cases[0]
    edit_triples_idx = case['orig']['edit_triples_idx'][0]
    answer = [case['new_single_hops'][edit_triples_idx]['answer']]
    answer.extend(case['new_single_hops'][edit_triples_idx]['answer_alias'])
    if fake_bios:
        with open(QUESTION_BIO_PATH, 'r', encoding='utf-8') as f:
            questions_with_fake_bios = json.load(f)
        fact_id = int(fact_id)
        answer = [questions_with_fake_bios[fact_id]["name"]] if "name" in questions_with_fake_bios[fact_id].keys() else answer
    for question in question_list:
        if "qwen" in model_name:
            message = STMessage("user", sections=[
            Section(content=task_description+question),])
            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_message = STStepMessages(messages=[message, assistant_message])
        elif "llama" in model_name:
            message = STMessage("user", sections=[
            Section(content="\nquestion: " + question + task_description), # question
            ])
            st_message = STStepMessages(messages=[message])
        else:
            raise ValueError(f"Model {model_name} is not supported")
        rephrased_questions.append((case["case_id"], question, st_message, answer))
    return rephrased_questions

def load_neighbor_questions_to_st_messages(data_path, task_description, model_name = "qwen", thinking_mode = True):
    # data_path to string
    data_path_str = str(data_path)
    fact_id = data_path_str.split(".json")[0].split("_")[-1]
    with open(GENERALIZATION_LOCALITY_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    locality_questions = []
    question_list = data[fact_id]["locality_questions"]
    for question_answer in question_list:
        if "qwen" in model_name:
            message = STMessage("user", sections=[
            Section(content=task_description+question_answer['question']),])
            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_message = STStepMessages(messages=[message, assistant_message])
        elif "llama" in model_name:
            message = STMessage("user", sections=[
            Section(content="\nquestion: " + question_answer['question'] + task_description), # question
            ])
            st_message = STStepMessages(messages=[message])
        else:
            raise ValueError(f"Model {model_name} is not supported")
        locality_questions.append((0, question_answer['question'], st_message, question_answer['answer'].split("/")))
    return locality_questions


def load_mhop_questions_to_st_messages(data_path, case_id, task_description, model_name = "qwen", thinking_mode = True, fake_bios = False):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    mhop_questions = []

    if fake_bios:
        with open(QUESTION_BIO_PATH, 'r', encoding='utf-8') as f:
            questions_with_fake_bios = json.load(f)

    for item in data:
        if case_id is None or item["case_id"] in case_id :
            reasoning_answers = [item["new_answer"]]
            reasoning_answers.extend(item["new_answer_alias"])
            if fake_bios:
                data_path_str = str(data_path)
                fact_id = data_path_str.split(".json")[0].split("_")[-1]
                fact_id = int(fact_id)
                reasoning_answers = [questions_with_fake_bios[fact_id]["name"]] if "name" in questions_with_fake_bios[fact_id].keys() else reasoning_answers
            for q in item["questions"]:
                if "qwen" in model_name:
                    message = STMessage("user", sections=[
                    Section(content=task_description+q),])
                    thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
                    assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
                    st_message = STStepMessages(messages=[message, assistant_message])
                elif "llama" in model_name:
                    message = STMessage("user", sections=[
                    Section(content="\nquestion: " + q + task_description), # question
                    ])
                    st_message = STStepMessages(messages=[message])
                else:
                    raise ValueError(f"Model {model_name} is not supported")
                mhop_questions.append((item["case_id"], q, st_message, reasoning_answers))
    return mhop_questions

def load_transformed_mhop_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True,):
    with open(UNSEEN_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    with open(PORT_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data_original = json.load(fp=f)
    mhop_questions = []
    for i,item in enumerate(data):
        case_id = item['case_id']
        question = item['question']
        answer = item['answer']
        answer.extend(data_original[i]['answer'])
        if "qwen" in model_name:
            message = STMessage("user", sections=[
            Section(content=task_description+question),])
            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_message = STStepMessages(messages=[message, assistant_message])
        elif "llama" in model_name:
            message = STMessage("user", sections=[
            Section(content="\nquestion: " + question + task_description), # question
            ])
            st_message = STStepMessages(messages=[message])
        else:
            raise ValueError(f"Model {model_name} is not supported")
        mhop_questions.append((case_id, question, st_message, answer))
    return mhop_questions

def load_unedited_questions_to_st_messages(data_path, task_description, model_name = "qwen", thinking_mode = True,):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    unedited_questions = []
    for item in data:
        edit_triples_idx = item['orig']['edit_triples_idx'][0]
        questions = []
        answers = []
        for i, single_hop in enumerate(item['new_single_hops']):
            if i != edit_triples_idx:
                questions.append(single_hop['question'])
                single_hop_answer = [single_hop['answer']]
                single_hop_answer.extend(single_hop['answer_alias'])
                answers.append(single_hop_answer)
        for question, answer in zip(questions, answers):
            if "qwen" in model_name:
                message = STMessage("user", sections=[
                Section(content=task_description+question),])
                thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
                assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
                st_message = STStepMessages(messages=[message, assistant_message])
            elif "llama" in model_name:
                message = STMessage("user", sections=[
                    Section(content="\nquestion: " + question + task_description), # question
                    ])
                st_message = STStepMessages(messages=[message])
            else:
                raise ValueError(f"Model {model_name} is not supported")
            unedited_questions.append((item["case_id"], question, st_message, answer))
    return unedited_questions

def load_challenging_mhop_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True):

    with open(MULTIFACT_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(fp=f)
    mhop_questions = []
    for item in data:
        if "qwen" in model_name:
            message = STMessage("user", sections=[
            Section(content=task_description+item['question']),])
            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_message = STStepMessages(messages=[message, assistant_message])
        elif "llama" in model_name:
            message = STMessage("user", sections=[
            Section(content="\nquestion: " + item['question'] + task_description), # question
            ])
            st_message = STStepMessages(messages=[message])
        else:
            raise ValueError(f"Model {model_name} is not supported")
        mhop_questions.append((item['id'], item['question'], st_message, [item['answer']]))
    return mhop_questions

def evaluate_answer(response, correct_answer, condition = Literal["partial", "exact"], thinking_mode = True, model_type = None, model_name = "qwen"):

    report = ""

    if "qwen" in model_name:
        model_answer = response.split("</think>")[1] if thinking_mode else response
    elif "llama" in model_name:
        if thinking_mode:
            if "</think>" in response:
                model_answer = response.split("</think>")[1]
            else:
                model_answer = response
                report += "No thinking process found."
        else:
            model_answer = response

    if "<answer>" in model_answer and "</answer>" not in model_answer:
        model_answer = model_answer.split("<answer>")[1]
    elif  "<answer>" not in model_answer and "</answer>" not in model_answer:
        if model_type != "sft":
            report += "No answer found in the response."
            return "no answer", 0, report
    elif "<answer>" not in model_answer and "</answer>" in model_answer:
        model_answer = model_answer.split("</answer>")[0]
    else:
        model_answer = model_answer.split("<answer>")[1].split("</answer>")[0]


    for a in correct_answer:
        a = a.lower()
        model_answer = model_answer.lower()
        if condition == "partial":
            if model_answer in a or a in model_answer:
                report += f"The model's answer is correct. {model_answer} is in {a} or {a} is in {model_answer}"
                return model_answer, 1, report
        elif condition == "exact" and model_answer == a:
            report += f"The model's answer is correct. {model_answer} equals to {a}"
            return model_answer, 1, report
    report += f"The model's answer is incorrect. {model_answer} is not in {correct_answer}"
    return model_answer, 0, report

def evaluate_answer_challenging_mhop(response: str, correct_answer: list[list[str]], condition = Literal["partial", "exact"], thinking_mode = True, model_type = None, model_name = "qwen"):

    report = ""

    if "qwen" in model_name:
        model_answer = response.split("</think>")[1] if thinking_mode else response
    elif "llama" in model_name:
        if thinking_mode:
            if "</think>" in response:
                model_answer = response.split("</think>")[1]
            else:
                model_answer = response
                report += "No thinking process found."
        else:
            model_answer = response


    if "<answer>" in model_answer and "</answer>" not in model_answer:
        model_answer = model_answer.split("<answer>")[1]
    elif  "<answer>" not in model_answer and "</answer>" not in model_answer:
        if model_type != "sft":
            report += "No answer found in the response."
            return "no answer",0, report
    elif "<answer>" not in model_answer and "</answer>" in model_answer:
        model_answer = model_answer.split("</answer>")[0]
    else:
        model_answer = model_answer.split("<answer>")[1].split("</answer>")[0]
    for a in correct_answer:
        for item in a:
            item = item.lower()
            model_answer = model_answer.lower()
            if condition == "partial":
                if model_answer not in item and item not in model_answer:
                    report += f"The model's answer is incorrect. {model_answer} is not in {item}"
                    return model_answer, 0, report
            elif condition == "exact":
                if model_answer != item:
                    report += f"The model's answer is incorrect. {model_answer} is not equal to {item}"
                    return model_answer, 0, report
    report += f"The model's answer is correct. {model_answer} is in {correct_answer}"
    return model_answer, 1, report

async def evaluate(client = None, task_description = None, data_path = None, case_id = None, fact_id = None, time_stamp = None, question_type = Literal["original", "rephrased", "neighbor", "port", "port-unseen", "unedited"], evaluate_condition = Literal["partial", "exact"], skip_generation = False, thinking_mode = True, fake_bios = False, model_name = "qwen", model_type = None,response_path = None, max_concurrent = 20, max_tokens = 1024, temperature = 0.6, min_p = 0, top_p = 0.95):
    if time_stamp is None:
        time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # match the question type to the function
    match question_type:
        case "port":
            questions = load_mhop_questions_to_st_messages(data_path, case_id, task_description, model_name, thinking_mode, fake_bios)
        case "port-unseen":
            questions = load_transformed_mhop_questions_to_st_messages(task_description, model_name, thinking_mode)
        case "original":
            questions = load_original_questions_to_st_messages(data_path, task_description, model_name, thinking_mode, fake_bios)
        case "rephrased":
            questions = load_rephrased_questions_to_st_messages(data_path, task_description, model_name, thinking_mode, fake_bios)
        case "neighbor":
            questions = load_neighbor_questions_to_st_messages(data_path, task_description, model_name, thinking_mode)
        case "unedited":
            questions = load_unedited_questions_to_st_messages(data_path, task_description, model_name, thinking_mode)
        case "port-multi-fact":
            questions = load_challenging_mhop_questions_to_st_messages(task_description, model_name, thinking_mode)
        case _:
            raise ValueError(f"Invalid question type: {question_type}")
    st_messages = [questions[i][2] for i in range(len(questions))]
    thinking_mode_str = "thinking" if thinking_mode else "nothinking"
    if not skip_generation:
        responses_dict = []
        if "qwen" in model_name:
            responses = await call_client_to_generate_next_message(st_messages, client, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p) if thinking_mode else await call_client_to_generate_next_message_no_thinking(st_messages, client, answer_start_tag="<answer>", answer_end_tag="</answer>", max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
        elif "llama" in model_name:
            responses = await call_client_to_generate_next_message_llama(st_messages, client, answer_start_tag='<answer>', answer_end_tag='</answer>', max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
        else:
            raise ValueError(f"Model {model_name} is not supported")
        if len(responses) != len(questions):
            raise ValueError(f"The number of responses is not equal to the number of mhop questions. {len(responses)}, {len(questions)}")
        for i in range(len(questions)):
            responses_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'response': responses[i]})
        # Create directory if it doesn't exist

        output_dir = Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/fact_{fact_id}_{thinking_mode_str}") if fact_id is not None else Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/{thinking_mode_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "responses.json", 'w', encoding='utf-8') as f:
            json.dump(responses_dict, f, ensure_ascii=False, indent=4)
    else:
        if response_path is None:
            raise ValueError("response_path is required when skip_generation is True")
        with open(response_path, 'r', encoding='utf-8') as f:
            report_json = json.load(fp=f)
        responses = [report_json[i]['response'] for i in range(len(report_json))]

    true_answers = [questions[i][3] for i in range(len(questions))]
    if len(responses) != len(true_answers):
        raise ValueError(f"The number of responses is not equal to the number of true answers. {len(responses)},{len(true_answers)}")
    total_score = 0
    reports = []
    results_dict = []
    for i in range(len(responses)):
        model_answer, score, report = evaluate_answer(responses[i], true_answers[i], evaluate_condition, thinking_mode, model_type, model_name) if question_type != "port-multi-fact" else evaluate_answer_challenging_mhop(responses[i], true_answers[i], evaluate_condition, thinking_mode, model_type, model_name)
        reports.append(report)
        total_score += score
        results_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'model_answer': model_answer, 'true_answer': '/'.join(true_answers[i]), 'score': score, 'report': report}) if question_type != "port-multi-fact" else results_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'model_answer': model_answer, 'true_answer': true_answers[i], 'score': score, 'report': report})
    results_dict.append({'total_score': total_score / len(responses)})
    # Create directory if it doesn't exist
    output_dir = Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/fact_{fact_id}_{thinking_mode_str}") if fact_id is not None else Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/{thinking_mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"results_{evaluate_condition}.json", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    return total_score / len(responses), len(responses), reports

# %%
#################################################
# task_description = """\nNow, answer the following question by step-by-step reasoning using your knowledge. Write down only the final answer (without extra text) between <answer> and </answer> tags.

# Question: """ # thinking mode + story

# task_description = """\nNow, answer the following question by step-by-step reasoning. Write down only the final answer (without extra text) between <answer> and </answer> tags.

# Question: """ # thinking mode + atomic

# task_description = """\nNow, answer the following question using your knowledge. Your final answer should be wrapped with <answer> and </answer> tags.

# Question: """ # non-thinking + story

# task_description = """Now, answer the following question using your updated knowledge. Between <answer> and </answer> tags, write down only the final answer.""" # non-thinking + atomic

#################################################


task_description = """\nNow, answer this question. Write down your final answer (without extra text) between <answer> and </answer> tags:""" # llama non-thinking; story

# task_description = """\nNow, answer this question by step-by-step reasoning. First, write down your thinking process between <think> and </think> tags. Then, write down your final answer (without extra text) between <answer> and </answer> tags:""" # llama thinking; story

# task_description = """\nNow, answer this question. Write down your final answer (without extra text) between <answer> and </answer> tags.""" # llama non-thinking; atomic

# task_description = """\nNow, answer this question by step-by-step reasoning. First, write down your thinking process between <think> and </think> tags. Then, write down your final answer (without extra text) between <answer> and </answer> tags.""" # llama thinking; atomic



from tasks.t_knowledge_edit.model_name import MODEL_NAME

DATASET_NAME = "FictBio"
client = Client(model=f"{MODEL_NAME}")


# %%
# for the cases are stored in the separate files
time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')

thinking_mode = True
fake_bios = True



if "llama" in MODEL_NAME:
    min_p = 0.2
    top_p = 0.9
    temperature = 0.6
if "qwen" in MODEL_NAME:
    min_p = 0
    temperature = 0.6 if thinking_mode else 0.7
    top_p = 0.95 if thinking_mode else 0.8

# locality questions consist of `neighbor` and `unedited` questions.
# original: edit success - original
# rephrased: edit success - rephrased
for question_type in ["original", "rephrased", "neighbor", "port", "port-unseen", "unedited"]:

    total_score = 0
    total_len_responses = 0
    score_per_fact = {}
    fact_id =  [i for i in range(38)] # batch editing; use set-specific-id for sequential evaluation
    # fact_id = [1]
    for i in fact_id:
        data_path = FictBio_PATH / f"FictBio_cases_per_fact/case_with_most_common_fact_{i}.json"
        response_path = None
        score, len_responses, reports = await evaluate(client, task_description, data_path, case_id = None, fact_id = i, time_stamp = time_stamp, question_type = question_type, evaluate_condition = "partial", skip_generation = False, thinking_mode = thinking_mode, fake_bios = fake_bios, model_name = MODEL_NAME, model_type = None, response_path = response_path, max_concurrent = 20, max_tokens = 1024, temperature = temperature, min_p = min_p, top_p = top_p)
        score_per_fact[i] = score
        total_score += (score * len_responses)
        total_len_responses += len_responses
    print(f"Total score: {total_score / total_len_responses}")
    score_per_fact["total_score"] = total_score / total_len_responses
    with open(Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/score_per_fact.json"), 'w', encoding='utf-8') as f:
        json.dump(score_per_fact, f, ensure_ascii=False, indent=4)

# %%
# test portabilit-multi-fact

time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
thinking_mode = True
fake_bios = True

if "llama" in MODEL_NAME:
    min_p = 0.2
    top_p = 0.9
    temperature = 0.6
if "qwen" in MODEL_NAME:
    min_p = 0
    temperature = 0.6 if thinking_mode else 0.7
    top_p = 0.95 if thinking_mode else 0.8

question_type = "port-multi-fact"

score, len_responses, reports = await evaluate(client, task_description, data_path=None, case_id = None, fact_id = None, time_stamp = time_stamp, question_type = question_type, evaluate_condition = "partial", skip_generation = False, thinking_mode = thinking_mode, fake_bios = fake_bios, model_name = MODEL_NAME, model_type = None, response_path = None, max_concurrent = 20, max_tokens = 1024, temperature = temperature, min_p = min_p, top_p = top_p)

print(f"Total score: {score}")
