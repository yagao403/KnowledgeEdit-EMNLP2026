# %%
from core import BASE_PATH, MQuAKE_PATH
from core.message import STMessage, Section
from core.client import Client
from core.steps import STStepMessages
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message, call_client_to_generate_next_message_no_thinking, call_client_to_generate_next_message_llama
import json
from typing import Literal
from pathlib import Path
from datetime import datetime


with open(Path(MQuAKE_PATH / f""), 'r', encoding='utf-8') as f: # path to original data file containing all cases after filtering
        DATA = json.load(fp=f)
with open(Path(MQuAKE_PATH / f""), 'r', encoding='utf-8') as f: # path to the locality questions
        DATA_LOC_S = json.load(fp=f)
with open(Path(MQuAKE_PATH / f""), 'r', encoding='utf-8') as f: # path to rephrased questions file
        REPHRASED_QUESTIONS = json.load(fp=f)
with open(Path(MQuAKE_PATH / f""), 'r', encoding='utf-8') as f: # path to MQuAKE portability test questions file
        ALL_PORT_QUESTIONS = json.load(fp=f)
with open(Path(MQuAKE_PATH / f""), 'r', encoding='utf-8') as f: # path to MQuAKE portability-unseen test questions file
        ALL_PORT_QUESTIONS_TRANSFORMED = json.load(fp=f)

def load_eff_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True,):
    eff_questions = []
    for case in DATA:
        edit_triples_idx = case['orig']['edit_triples_idx']
        for i, new_shingle_hop in enumerate(case['new_single_hops']):
            if i in edit_triples_idx:
                question = new_shingle_hop['question']
                answer = [new_shingle_hop['answer']]
                answer.extend(new_shingle_hop['answer_alias'])
                if "qwen" in model_name:
                    message = STMessage("user", sections=[
                    Section(content=task_description+question),])
                    thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
                    assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
                    st_message = STStepMessages(messages=[message, assistant_message])
                elif "llama" in model_name:
                    message = STMessage("user", sections=[
                        Section(content="Your task is to answer the following question using your updated knowledge.\nquestion: " + question + task_description), # question
                        ])
                    st_message = STStepMessages(messages=[message])
                else:
                    raise ValueError(f"Model {model_name} is not supported")
                eff_questions.append((case["case_id"], question, st_message, answer))
    return eff_questions

def load_gen_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True):
    gen_questions = []
    for case in DATA:
        edit_triples_idx = case['orig']['edit_triples_idx']
        for i, new_shingle_hop in enumerate(case['new_single_hops']):
            if i in edit_triples_idx:
                fact = new_shingle_hop['cloze'] + " " + new_shingle_hop['answer']
                question = new_shingle_hop['question']
                answer = [new_shingle_hop['answer']]
                answer.extend(new_shingle_hop['answer_alias'])
                rephrased_questions = REPHRASED_QUESTIONS[question]
                for rephrased_question in rephrased_questions:
                    if "qwen" in model_name:
                        message = STMessage("user", sections=[
                        Section(content=task_description+rephrased_question),])
                        thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
                        assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
                        st_message = STStepMessages(messages=[message, assistant_message])
                    elif "llama" in model_name:
                        message = STMessage("user", sections=[
                            Section(content="Your task is to answer the following question using your updated knowledge.\nquestion: " + question + task_description), # question
                            ])
                        st_message = STStepMessages(messages=[message])
                    else:
                        raise ValueError(f"Model {model_name} is not supported")
                    gen_questions.append((case["case_id"], rephrased_question, st_message, answer))
    return gen_questions

def load_loc_s_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True,):
    # data_path to string
    loc_s_questions = []
    for item in DATA_LOC_S:
        case_id = item['id']
        question = item['question']
        answer = item['answer']
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
        loc_s_questions.append((case_id, question, st_message, answer))
    return loc_s_questions

def load_port_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True):
    port_questions = []
    for case in DATA:
        answer = [case['new_answer']]
        answer.extend(case['new_answer_alias'])
        for question in case['questions']:
            if "qwen" in model_name:
                message = STMessage("user", sections=[Section(content=task_description+question),])
                thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
                assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
                st_message = STStepMessages(messages=[message, assistant_message])
            elif "llama" in model_name:
                message = STMessage("user", sections=[
                    Section(content="Your task is to answer the following question using your updated knowledge.\nquestion: " + question + task_description), # question
                    ])
                st_message = STStepMessages(messages=[message])
            port_questions.append((case["case_id"], question, st_message, answer))
    return port_questions

def load_transformed_port_questions_to_st_messages(task_description, model_name = "qwen", thinking_mode = True):
    port_questions = []
    for i, item in enumerate(ALL_PORT_QUESTIONS_TRANSFORMED):
        case_id = item['case_id']
        question = item['question']
        answer = item['answer']
        answer.extend(ALL_PORT_QUESTIONS[i]['answer'])
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
        port_questions.append((case_id, question, st_message, answer))
    return port_questions


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
                report += f"The model's answer is correct"
                return model_answer, 1, report
        elif condition == "exact" and model_answer == a:
            report += f"The model's answer is correct."
            return model_answer, 1, report
    report += f"The model's answer is incorrect. {model_answer} is not in {correct_answer}"
    return model_answer, 0, report

async def evaluate(client = None, task_description = None, time_stamp = None, question_type = Literal["eff", "gen", "port"], evaluate_condition = Literal["partial", "exact"], skip_generation = False, thinking_mode = True, model_name = "qwen", model_type = None, response_path = None, max_concurrent = 20, max_tokens = 1024, temperature = 0.6, min_p = 0, top_p = 0.95):
    if time_stamp is None:
        time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # match the question type to the function
    match question_type:
        case "port":
            questions = load_port_questions_to_st_messages(task_description, model_name, thinking_mode)
        case "port-unseen":
            questions = load_transformed_port_questions_to_st_messages(task_description, model_name, thinking_mode)
        case "eff":
            questions = load_eff_questions_to_st_messages(task_description, model_name, thinking_mode)
        case "gen":
            questions = load_gen_questions_to_st_messages(task_description, model_name, thinking_mode)
        case "loc_s":
            questions = load_loc_s_questions_to_st_messages(task_description, model_name, thinking_mode)
        case _:
            raise ValueError(f"Invalid question type: {question_type}")
    st_messages = [questions[i][2] for i in range(len(questions))]
    thinking_mode_str = "thinking" if thinking_mode else "nothinking"
    if not skip_generation:
        responses_dict = []
        if "qwen" in model_name:
            answer_start_tag = "<answer>" if model_type != "sft" else None
            answer_end_tag = "</answer>" if model_type != "sft" else None
            responses = await call_client_to_generate_next_message(st_messages, client, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p) if thinking_mode else await call_client_to_generate_next_message_no_thinking(st_messages, client, answer_start_tag=answer_start_tag, answer_end_tag=answer_end_tag, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
        elif "llama" in model_name:
            answer_start_tag = "<answer>"
            answer_end_tag = "</answer>"
            responses = await call_client_to_generate_next_message_llama(st_messages, client, answer_start_tag=answer_start_tag, answer_end_tag=answer_end_tag, max_tokens=max_tokens, max_concurrent=max_concurrent, temperature=temperature, min_p=min_p, top_p=top_p)
        else:
            raise ValueError(f"Model {model_name} is not supported")
        if len(responses) != len(questions):
            raise ValueError(f"The number of responses is not equal to the number of mhop questions. {len(responses)}, {len(questions)}")
        for i in range(len(questions)):
            responses_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'response': responses[i]})
        # Create directory if it doesn't exist

        output_dir = Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/{thinking_mode_str}")
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
        model_answer, score, report = evaluate_answer(responses[i], true_answers[i], evaluate_condition, thinking_mode, model_type, model_name)
        reports.append(report)
        total_score += score
        results_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'model_answer': model_answer, 'true_answer': '/'.join(true_answers[i]), 'score': score, 'report': report})
    results_dict.append({'total_score': total_score / len(responses)})
    # Create directory if it doesn't exist
    output_dir = Path(f"{BASE_PATH}/evaluation_records/{DATASET_NAME}/{MODEL_NAME}/{question_type}/{time_stamp}/{thinking_mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"results_{evaluate_condition}.json", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    with open(output_dir / f"task_description.json", 'w', encoding='utf-8') as f:
        json.dump([task_description], f, ensure_ascii=False, indent=4)


    return total_score / len(responses), len(responses), reports

# %%
from tasks.t_knowledge_edit.model_name import MODEL_NAME

thinking_mode = True

if 'sft' in MODEL_NAME:
    model_type = 'sft'
else:
    model_type = None
#################################################
if 'qwen' in MODEL_NAME:
    if thinking_mode:
        task_description = """\nNow, answer the following question by step-by-step reasoning using your knowledge. Write down only the final answer (without extra text) between <answer> and </answer> tags.\n\nQuestion: """
        if 'ICE' in MODEL_NAME:
            task_description = """""" # ICE thinking mode
    else:
        task_description = """\nNow, answer the following question using your knowledge. Write down only the final answer (without extra text) between <answer> and </answer> tags.\n\nQuestion: """
        if model_type == "sft":
            task_description = """""" # ICE non-thinking mode
if 'llama' in MODEL_NAME:
    if thinking_mode:
            task_description = """\nNow, answer this question by step-by-step reasoning. First, write down your thinking process between <think> and </think> tags. Then, write down your final answer (without extra text) between <answer> and </answer> tags."""

    else:
        task_description = """\nNow, answer this question. Write down your final answer (without extra text) between <answer> and </answer> tags."""

print(task_description)


DATASET_NAME = "MQuAKE-CF"

client = Client(model=f"{MODEL_NAME}")


# %%
time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
if "llama" in MODEL_NAME:
    min_p = 0.2
    top_p = 0.9
    temperature = 0.6
if "qwen" in MODEL_NAME:
    min_p = 0
    temperature = 0.6 if thinking_mode else 0.7
    top_p = 0.95 if thinking_mode else 0.8
# eff: edit success - original
# gen: edit success - rephrased
for question_type in ["eff", "gen", "loc_s", "port", "port-unseen"]:
    score, len_responses, reports = await evaluate(client, task_description, time_stamp = time_stamp, question_type = question_type, evaluate_condition = "partial", skip_generation = False, thinking_mode = thinking_mode, model_name = MODEL_NAME, model_type = model_type, response_path = None, max_concurrent = 20, max_tokens = 1024, temperature = temperature, min_p = min_p, top_p = top_p)
    print("score: ", score)
