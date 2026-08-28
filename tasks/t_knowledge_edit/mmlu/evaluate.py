# %%
from core import BASE_PATH, MMLU_PATH
from core.message import STMessage, Section
from core.steps import STStepMessages
from pathlib import Path
import json
from datetime import datetime
from core.client import Client
from tasks.t_knowledge_edit.call_clients import call_client_to_generate_next_message, call_client_to_generate_next_message_no_thinking, call_client_to_generate_next_message_llama



#14042
Batch_Index = 3 # 0 - 4

START_INDEX = Batch_Index * 3000
END_INDEX = (Batch_Index + 1) * 3000

def load_mmlu_data():
    """
    Load MMLU test data from JSON file.
    Expected format: List of dictionaries with keys:
    - 'question': str
    - 'choices': list[str] (typically 4 options: A, B, C, D)
    - 'answer': str or int (correct answer)
    - 'subject': str (optional, subject category)
    """
    with open(MMLU_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[START_INDEX:END_INDEX]

def format_mmlu_question(question: str, choices: list[str]) -> str:
    """
    Format MMLU question with multiple choice options.
    """
    formatted = question + "\n\n"
    labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices[:len(labels)]):
        formatted += f"{labels[i]}. {choice}\n"
    return formatted

def load_mmlu_to_st_messages(
    task_description: str,
    model_name: str = "qwen",
    thinking_mode: bool = True,
    subject: str | None = None
):
    """
    Load MMLU test questions and convert to STMessage format.

    Args:
        task_description: Task instruction prompt
        model_name: Model type ("qwen" or "llama")
        thinking_mode: Whether to use thinking mode
        subject: Optional filter by subject

    Returns:
        List of tuples: (id, question, st_message, answer)
    """
    data = load_mmlu_data()
    st_messages = []

    for idx, item in enumerate(data):
        # Filter by subject if specified
        if subject and item.get('subject') != subject:
            continue

        question_text = item['question']
        choices = item['choices']
        answer = item['answer']
        item_id = item['id']

        # Format the question with choices
        formatted_question = format_mmlu_question(question_text, choices)

        # Prepare answer - ensure it's in list format
        if isinstance(answer, int):
            # Convert index to letter
            answer_list = [['A', 'B', 'C', 'D'][answer]]
        elif isinstance(answer, str):
            answer_list = [answer]
        else:
            answer_list = answer if isinstance(answer, list) else [str(answer)]

        # Create message based on model type
        if "qwen" in model_name:
            message = STMessage("user", sections=[
                Section(content=task_description + formatted_question)
            ])
            thinking_start = "<think>\n" if thinking_mode else "<think>\n\n</think>\n\n"
            assistant_message = STMessage("assistant", sections=[Section(thinking_start)])
            st_message = STStepMessages(messages=[message, assistant_message])

        elif "llama" in model_name:
            message = STMessage("user", sections=[
                Section(content="Question: " +
                        formatted_question + "\n" + task_description)
            ])
            st_message = STStepMessages(messages=[message])
        else:
            raise ValueError(f"Model {model_name} is not supported")

        # Store as (id, question, st_message, answer)
        st_messages.append((item_id, formatted_question, st_message, answer_list))

    return st_messages

def evaluate_answer(response, correct_answer, thinking_mode = True, model_type = None, model_name = "qwen"):

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

    if model_answer == "":
        report += "No answer found in the response."
        return "no answer", 0, report

    for a in correct_answer:
        a = a.lower()
        model_answer = model_answer.lower().strip()
        # remove the punctuation
        # a = a.translate(str.maketrans('', '', string.punctuation))
        # model_answer = model_answer.translate(str.maketrans('', '', string.punctuation))
        if model_answer == a or (len(model_answer) > 0 and len(a) > 0 and model_answer[0] == a[0]):
            report += f"The model's answer is correct."
            return model_answer, 1, report
    report += f"The model's answer is incorrect. {model_answer} is not in {correct_answer}"
    return model_answer, 0, report

async def evaluate(client = None, task_description = None, time_stamp = None, st_messages = None, skip_generation = False, thinking_mode = True, model_name = "qwen", model_type = None, response_path = None, max_concurrent = 20, max_tokens = 1024, temperature = 0.6, min_p = 0, top_p = 0.95):
    if time_stamp is None:
        time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    questions = load_mmlu_to_st_messages(task_description, model_name, thinking_mode)
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

        output_dir = Path(f"{BASE_PATH}/evaluation_records/MMLU/{MODEL_NAME}/{Batch_Index}/{time_stamp}/{thinking_mode_str}")
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
        model_answer, score, report = evaluate_answer(responses[i], true_answers[i], thinking_mode, model_type, model_name)
        reports.append(report)
        total_score += score
        results_dict.append({'case_id': questions[i][0], 'question': questions[i][1], 'model_answer': model_answer, 'true_answer': '/'.join(true_answers[i]), 'score': score, 'report': report})
    results_dict.append({'total_score': total_score / len(responses)})
    # Create directory if it doesn't exist
    output_dir = Path(f"{BASE_PATH}/evaluation_records/MMLU/{MODEL_NAME}/{Batch_Index}/{time_stamp}/{thinking_mode_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"results.json", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    with open(output_dir / f"task_description.json", 'w', encoding='utf-8') as f:
        json.dump([task_description], f, ensure_ascii=False, indent=4)


    return total_score / len(responses), len(responses), reports



#%%
# Example usage

task_description_thinking = """
Answer the following choice question by step-by-step reasoning. There is only one correct choice.
Write down only the final choice (A, B, C, or D) between <answer> and </answer> tags.

Question: """

task_description_thinking_llama = """
Answer the above choice question by step-by-step reasoning. There is only one correct choice. Write down your reasoning process between <think> and </think> tags. Then, write down only the final choice (A, B, C, or D) between <answer> and </answer> tags."""


thinking_mode = True



MODEL_NAME = "llama3.1-70b"
# MODEL_NAME = "qwen3-32b"
model_type = None
client = Client(model=f"{MODEL_NAME}")


# %%
time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')


if "llama" in MODEL_NAME:
    min_p = 0.2
    top_p = 0.9
    temperature = 0.6
    task_description = task_description_thinking_llama
if "qwen" in MODEL_NAME:
    min_p = 0
    temperature = 0.6 if thinking_mode else 0.7
    top_p = 0.95 if thinking_mode else 0.8
    task_description = task_description_thinking


score, len_responses, reports = await evaluate(client, task_description, time_stamp = time_stamp, skip_generation = False, thinking_mode = thinking_mode, model_name = MODEL_NAME, model_type = model_type, response_path = None, max_concurrent = 20, max_tokens = 4096, temperature = temperature, min_p = min_p, top_p = top_p)
print("score: ", score)


# %%
time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
response_path = Path("")  # path to a previously generated responses.json file
score, len_responses, reports = await evaluate(client, task_description, time_stamp = time_stamp, skip_generation = True, thinking_mode = thinking_mode, model_name = MODEL_NAME, model_type = model_type, response_path = response_path, max_concurrent = 20, max_tokens = 4096, temperature = temperature, min_p = min_p, top_p = top_p)
print("score: ", score)
