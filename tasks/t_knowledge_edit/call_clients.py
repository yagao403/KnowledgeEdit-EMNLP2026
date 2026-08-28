#%%
from core.message import Section, STMessage
import asyncio
import copy
from typing import Callable, Iterable
from tqdm.asyncio import tqdm as atqdm

async def parallel_map_with_limit(func: Callable, iterable: Iterable, max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_task(item, index):
        async with semaphore:
            result = await func(item)
            return index, result

    items = list(iterable)
    tasks = [asyncio.create_task(sem_task(item, i)) for i, item in enumerate(items)]

    # Track progress while maintaining order
    results = [None] * len(items)
    with atqdm(total=len(tasks), desc="Processing requests") as pbar:
        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            results[index] = result
            pbar.update(1)

    return results

async def call_client_to_generate_next_message(
    messages,
    client,
    max_concurrent: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    min_p: float = 0,
    top_p: float = 0.95,
):
    async def call(st_message):
        response = await client.call(st_message.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True, )
        if "</think>" not in response:
            response += "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"
            st_message_tmp = copy.deepcopy(st_message)
            st_message_tmp[1].sections.append(Section(content=response, target=True))
            new_response = await client.call(st_message_tmp.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True)
            response += new_response
        return response

    # if max_concurrent and max_concurrent > 1:
    responses = await parallel_map_with_limit(call, messages, max_concurrent)
    # else:
    #     responses = [await call(step_messages) for step_messages in tqdm(message_lists)]

    return responses

async def call_client_to_generate_next_message_no_thinking(
    messages,
    client,
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
    max_concurrent: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    min_p: float = 0,
    top_p: float = 0.95,
):
    async def call(st_message):
        response = await client.call(st_message.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True, )

        validate_answer_start_tag = False
        validate_answer_end_tag = False
        continue_generation = False

        if answer_start_tag is not None:
            validate_answer_start_tag = True
        if answer_end_tag is not None:
            validate_answer_end_tag = True
        if validate_answer_start_tag:
            if answer_start_tag not in response:
                continue_generation = True
        if validate_answer_end_tag:
            if answer_end_tag not in response:
                continue_generation = True
        if continue_generation:
        # if answer_start_tag not in response or answer_end_tag not in response:
            # if answer_end_tag not in response:
            response += f"\nConsidering the limited time by the user, I have to give the solution directly now.\nI will wrap my final answer with correct answer tags.\n"
            st_message_tmp = copy.deepcopy(st_message)
            st_message_tmp[1].sections.append(Section(content=response, target=True))
            new_response = await client.call(st_message_tmp.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True)
            response += new_response
        return response

    responses = await parallel_map_with_limit(call, messages, max_concurrent)

    return responses

async def call_client_to_generate_next_message_llama(
    messages,
    client,
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
    max_concurrent: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    min_p: float = 0,
    top_p: float = 0.95,
):
    async def call(st_message):
        response = await client.call(st_message.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True, )

        validate_answer_start_tag = False
        validate_answer_end_tag = False
        continue_generation = False

        if answer_start_tag is not None:
            validate_answer_start_tag = True
        if answer_end_tag is not None:
            validate_answer_end_tag = True
        if validate_answer_start_tag:
            if answer_start_tag not in response:
                continue_generation = True
        if validate_answer_end_tag:
            if answer_end_tag not in response:
                continue_generation = True
        if continue_generation:
        # if answer_start_tag not in response or answer_end_tag not in response:
            response += f"\nI will answer this question and wrap my final answer with correct answer tags.\n"
            st_message_tmp = copy.deepcopy(st_message)
            assistant_message = STMessage("assistant", sections=[Section(response, target=True)])
            st_message_tmp.append(assistant_message)
            new_response = await client.call(st_message_tmp.to_messages(teacher=True), max_tokens=max_tokens, temperature=temperature, min_p=min_p, top_p=top_p, continue_last_message=True)
            response += new_response
        return response
    responses = await parallel_map_with_limit(call, messages, max_concurrent)

    return responses
