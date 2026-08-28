#%% import json
import json
from pathlib import Path
from core import BASE_PATH


#%%
MODEL_NAME = "qwen3-32b"
for MODEL_NAME in ["qwen3-32b", "llama3.1-70b", "",]: # add model names of the trained models you want to evaluate

    scores = []
    for Batch_Index in range(5):
        for results_file in Path(f"{BASE_PATH}/evaluation_records/MMLU/{MODEL_NAME}/{Batch_Index}").glob("*/thinking/results.json"):
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        for result in results[:-1]:
            scores.append(result['score'])
    average_score = sum(scores) / len(scores)
    print(f"{MODEL_NAME} Average score: {average_score}")
