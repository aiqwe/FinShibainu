""" 데이터를 자동으로 생성하는 Graph Pipeline"""

from typing import List, Dict
from functools import partial
import random
import os
from datetime import datetime

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from loguru import logger
import pandas as pd
from .config import PathConfig
from .utils import concurrent_execute
from .api import completion
from .io import write_jsonl, write_json
from .processing import (
    split_chunk_with_index,
    convert_list_to_dict,
    convert_dict_to_list,
    find_answer_errors,
    find_option_errors,
    find_reasoning_errors,
    find_classification_errors,
    find_hallucination_errors,
    find_hallucination_filters,
    find_preference_errors,
    shuffle_answer,
    find_deficient_errors,
    find_num_matching_errors,
    add_conclusion
)

# ==================== Langraph 사용법 ====================
# 1. TypedDict로 state의 타입을 정의함
# 2. Node 정의
# 3. edge 정의
# 4. Node와 Edge를 이용해서 Graph를 생성
# =======================================================

class MCQAGraphState(TypedDict):
    save_dir: str
    save_file_name: str
    task_type: str
    eval_type: str
    domain_type: str
    n_datasets: int
    n_workers: int
    step: int
    if_fewshot: bool
    max_step: int
    oai_model: str
    error_tolerance_ratio: float
    hallucination_tolerance_ratio: float
    show_error_log: bool
    data: List[Dict]
    error: List[Dict]
    result: List[List]
    hallucination: bool

class QAGraphState(TypedDict):
    save_dir: str
    save_file_name: str
    task_type: str
    eval_type: str
    domain_type: str
    n_datasets: int
    n_workers: int
    step: int  # start = 1
    max_step: int
    oai_model: str
    error_tolerance_ratio: float
    show_error_log: bool
    data: List[Dict]  # title, contents, init: data
    error: List[Dict]  # title, contents, init: None
    result: List[List]  # title, contents, question, options, reasoning_process, answers, init: None
    hallucination: bool

class ClassificationGraphState(TypedDict):
    save_dir: str
    save_file_name: str
    n_workers: int
    oai_model: str
    error_tolerance_ratio: float
    show_error_log: bool
    data: List[Dict]  # title, contents, init: data
    error: List[Dict]  # title, contents, init: None
    result: List[List]  # title, contents, question, options, reasoning_process, answers, init: None

# MCQA와 QA는 일부 Node와 Edge를 공유한다
# =====================================================
# =======================  Node  ======================
# =====================================================
def node_generate_mcqa_step_1(state):
    """ MCQA에서 첫번째 데이터 생성 노드 """
    step = 1
    data = state['data']
    error = state.get("error", None)
    show_error_log = state.get("show_error_log", True)
    if_fewshot = state.get("if_fewshot", True)

    state_dict = {k: v for k, v in state.items() if k != 'data'}
    param = []
    for k, v in state_dict.items():
        params = f"{k}: {v}"
        param.append(params)

    param_msg = '\n'.join(param)
    logger.info("**State Parameters**" + "\n" + param_msg)

    logger.info(f"node_generate_mcqa_step_1")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['step_1']

    chunks = split_chunk_with_index(data, n=4, by="num", embedding_index=True)
    random.shuffle(chunks)
    result = concurrent_execute(
        partial(
            completion,
            eval_type='mcqa',
            task_type=state['task_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            if_fewshot=if_fewshot,
            oai_model=oai_model
        ), iterables=chunks, n_workers=state['n_workers'], safe_wait=False)

    result = result[0]
    state.update({"result": result, "step": step, "error": error, "show_error_log": show_error_log, "if_fewshot": if_fewshot})
    return state

def node_regenerate_mcqa(state):
    """ MCQA에서 에러 index들을 재생성 하는 노드 """
    data = state['error']
    step = state['step']
    if_fewshot = state['if_fewshot']
    logger.info(f"node_regenerate_mcqa: step {step}")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['regenerate']
    chunks = split_chunk_with_index(data, n=4, by="num", embedding_index=True)
    random.shuffle(chunks)
    result = concurrent_execute(
        partial(
            completion,
            task_type=state['task_type'],
            eval_type=state['eval_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            if_fewshot=if_fewshot,
            oai_model=oai_model
        ), iterables=chunks, n_workers=state['n_workers'], safe_wait=False)

    processed = state['result'] + result[0]
    state.update({"result": processed, "step": step})

    return state

def node_generate_mcqa_step_n(state):
    """ MCQA에서 Step2 이상에서 데이터를 생성 하는 노드 """
    step = state['step']
    data = state['data']
    if_fewshot = state['if_fewshot']

    logger.info(f"node_generate_mcqa_step_n: step {step}")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['step_n']

    # prev examples + reference 합치기
    result = state['result']  # Step1의 결과
    df = pd.DataFrame(data)  # Reference
    if "prev_example" in df.columns:
        df = df.drop(columns="prev_example", axis=1)
    converted = convert_list_to_dict(result, keys=result[0].keys())
    converted_pdf = pd.DataFrame.from_dict(converted)
    converted_pdf['prev_example'] = \
        "### 질문:\n" + converted_pdf['question'] + \
        "### 선택지:\n" + converted_pdf['options'].apply(lambda x: "\n".join(x)) + \
        "### 풀이:\n" + converted_pdf['reasoning_process']
    converted_pdf = converted_pdf.groupby("index")['prev_example'].agg(lambda x: "\n".join(x)).reset_index()
    data = df.merge(converted_pdf, how="left", on="index")
    data = data.to_dict("records")

    chunks = split_chunk_with_index(data, n=4, by="num", embedding_index=True)
    result = concurrent_execute(
        partial(
            completion,
            task_type=state['task_type'],
            eval_type=state['eval_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            if_fewshot=if_fewshot,
            oai_model=oai_model
        ), iterables=chunks, n_workers=state['n_workers'], safe_wait=False)

    result = result[0]
    state.update({"result": result, "data": data, "step": step})
    return state

def node_classify_mcqa_hallucination(state):
    """ MCQA에서 Hallucination을 판단 하는 노드 """
    step = state['step']
    logger.info(f"node_classify_mcqa_hallucination: step {step}")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['hallucination']
    result = state['result']
    hallucination = True
    converted = convert_list_to_dict(result, keys=result[0].keys())
    converted_pdf = pd.DataFrame.from_dict(converted)
    data = converted_pdf.to_dict("records")

    result = concurrent_execute(
        partial(
            completion,
            eval_type="hallucination",
            oai_model=oai_model
        ), iterables=data, n_workers=state['n_workers'], safe_wait=False)

    result =  result[0] # flatten 되어있음
    state.update({"result": result, "hallucination": hallucination})
    return state

def node_reclassify_hallucination(state):
    """ Hallucination에러시 다시 데이터를 생성하는 노드 """
    error = state['error']
    step = state['step']
    logger.info(f"node_reclassify_hallucination: step {step}")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['hallucination']

    result = concurrent_execute(
        partial(
            completion,
            eval_type="hallucination",
            oai_model=oai_model
        ), iterables=error, n_workers=state['n_workers'], safe_wait=False)

    processed = state['result'] + result[0]
    state.update({"result": processed, "step": step})
    return state

def node_process_hallucination_errors(state):
    """ Hallucination의 에러를 체크하는 노드 """
    logger.info("node_process_hallucination_errors")
    result = state['result']

    # 에러 수집
    all_error = []
    _, hallucination_error, processed = find_hallucination_errors(response=result)
    if state['show_error_log']:
        logger.info(f"Hallucination Errors:{hallucination_error}({len(hallucination_error)})")
    all_error.extend(hallucination_error)

    # 중복 제거
    all_error = sorted(list(set(all_error)))
    if state['show_error_log']:
        logger.info(f"All Errors:{all_error}({len(all_error)})")
    if len(all_error) > 0:
        error = [item for item in result if item['index'] in all_error]

    else:
        error = []
    state.update({"result": processed, "error": error})

    return state

def node_process_mcqa_errors(state):
    """ MCQA에서 에러를 체크하는 노드 """
    logger.info("node_process_mcqa_errors")
    result = state['result']
    data = state['data']

    # 에러 수집
    all_error = []
    # Option -> Answer -> 그외로 진행해야함

    # Option이 정상이 아닌경우
    _, option_error, processed = find_option_errors(result)
    if state['show_error_log']:
        logger.info(f"Option Errors:{option_error}({len(option_error)})")
    all_error.extend(option_error)

    # Answer가 정상이 아닌경우
    origin, answer_error, processed = find_answer_errors(processed)
    if state['show_error_log']:
        logger.info(f"Answer Errors:{answer_error}({len(answer_error)})")
    all_error.extend(answer_error)

    # answer, question, options, reasoning_process의 갯수가 다른경우
    _, num_matching_error, processed = find_num_matching_errors(eval_type=state['eval_type'], response=processed)
    if state['show_error_log']:
        logger.info(f"Num of Options Not Matching Errors:{num_matching_error}({len(num_matching_error)})")
    all_error.extend(num_matching_error)

    # 생성하라고 지시한 n_datasets와 다른 경우
    _, deficient_error, processed = find_deficient_errors(eval_type=state['eval_type'], n_datasets=state['n_datasets'], response=processed)
    if state['show_error_log']:
        logger.info(f"Deficient for Options Errors:{deficient_error}({len(deficient_error)})")
    all_error.extend(deficient_error)

    # reasoning_process에서 선택지 항목을 언급한 경우
    _, reasoning_error, processed = find_reasoning_errors(response=processed)
    if state['show_error_log']:
        logger.info(f"Reasoning Utterance Errors:{reasoning_error}({len(reasoning_error)})")
    all_error.extend(reasoning_error)

    # 할루시네이션 처리중이면 에러 추가
    if state.get('hallucination', None):
        _, hallucination_error, processed = find_hallucination_errors(response=processed)
        if state['show_error_log']:
            logger.info(f"Hallucination Errors:{hallucination_error}({len(hallucination_error)})")
        all_error.extend(hallucination_error)

    # 중복 제거
    all_error = sorted(list(set(all_error)))
    if state['show_error_log']:
        logger.info(f"All Errors:{all_error}({len(all_error)})")
    if len(all_error) > 0:
        error = [item for item in data if item['index'] in all_error]

    else:
        error = []
    state.update({"result": processed, "error": error})

    return state

def node_value_grade_and_save_data(state):
    """ MCQA / QA에서 교육적 가치를 측정하고 저장하는 노드 """
    step = state['step']
    result = state['result']
    logger.info("node_value_grade_and_save_data")
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['value']

    try:
        # MCQA는 Step마다 Value를 grading하고 저장한다
        if state['eval_type'] == 'mcqa': # mcqa
            logger.info("Convert Batch to Element: List[Dict[List]] -> List[Dict]")
            result = convert_dict_to_list(result)
            result = convert_list_to_dict(result, keys=list(result[0].keys()))
            result = shuffle_answer(result)
            result = pd.DataFrame.from_dict(result)
            result = add_conclusion(result)
            result = result.to_dict("records")

            # Value grade
            logger.info("Grade value MCQA")

            result = concurrent_execute(
                partial(
                    completion,
                    eval_type='value',
                    oai_model=oai_model
                ), iterables=result, n_workers=state['n_workers'], safe_wait=False)
            result = result[0]
            eval_type = "mcqa"

        if state['eval_type'] == 'preference':
            # 에러체크때문에 Flatten 다시해줘야함
            logger.info(f"Convert Data type for Grade Value")
            result = convert_list_to_dict(result, result[0].keys())
            result = pd.DataFrame.from_dict(result)
            result = result.to_dict("records")

            logger.info("Grade value QA")
            result = concurrent_execute(
                partial(
                    completion,
                    eval_type='value',
                    oai_model=state['oai_model']
                ), iterables=result, n_workers=state['n_workers'], safe_wait=False)
            result = result[0]
            eval_type = "qa"

        # value 에러 처리
        result = [item for item in result if item['value'] in (0, 1, 2, 3, 4, 5)]
        # 저장
        logger.info("Save Data")
        os.makedirs(state['save_dir'], exist_ok=True)
        write_jsonl(result, f"{state['save_dir']}/{state['save_file_name']}_{state['task_type']}_{eval_type}_{state['domain_type']}_step_{step}.jsonl", ensure_ascii=False, overwrite=True)
    except Exception as e:
        eval_type = state['eval_type']
        path = PathConfig()
        logger.error(f"{repr(e)}")
        log_dir = os.path.join(path.ROOT, "logs")
        timesuffix = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(log_dir, exist_ok=True)

        if state['eval_type'] == 'mcqa': # mcqa
            logger.info("Convert Batch to Element: List[Dict[List]] -> List[Dict]")
            result = convert_dict_to_list(result)
            result = convert_list_to_dict(result, keys=list(result[0].keys()))
            result = pd.DataFrame.from_dict(result)
            result = result.to_dict("records")

        if isinstance(result, dict):
            write_json(result,
                       f"{log_dir}/error_{state['save_file_name']}_{state['task_type']}_{eval_type}_{state['domain_type']}_step_{step}_{timesuffix}.json",
                        ensure_ascii=False, overwrite=True)
        if isinstance(result, list):
            write_jsonl(result,f"{log_dir}/error_{state['save_file_name']}_{state['task_type']}_{eval_type}_{state['domain_type']}_step_{step}_{timesuffix}.jsonl",
                    ensure_ascii=False, overwrite=True)

def node_add_step(state):
    """ MCQA에서 Step을 증가시키는 노드 """
    ex = state['step']
    to = state['step'] + 1
    if state.get("hallucination", None):
        hallucination = False
    if state['eval_type'] == 'mcqa':
        result = convert_dict_to_list(state['result'])
        state.update({"result": result, "step": to, "hallucination": hallucination})
    else:
        state.update({"step": to})
    logger.info(f"node_add_step: {ex} -> {to}")
    return state

# =====================================================
# =======================  Edge  ======================
# =====================================================

def edge_check_error(state):
    """ 에러를 체크하고 에러율 결과별로 노드로 이어지는 엣지
      - 에러율 이하일 시: MCQA는 Hallucination으로, QA는 다음 Step으로(2번째 Answer 생성, Preference 생성 등)
      - 에러율 이상일 시: 저장된 에러 Index를 재생성
    """
    logger.info(f"edge_check_error: step {state['step']}")
    logger.info(f"num of errors: {len(state['error'])}")
    error_tolerance_size = int(state['error_tolerance_ratio'] * len(state['data']))
    error_tolerance_size = 1 if error_tolerance_size == 0 else error_tolerance_size
    error_size = len(state['error'])
    logger.info(f"Error tolerance: {error_size}/{error_tolerance_size}({error_size/error_tolerance_size:.2%})")
    if error_size > error_tolerance_size:
        if state['eval_type'] == 'mcqa':
            return "node_regenerate_mcqa"
        if state['eval_type'] == 'qa':
            return "node_regenerate_qa"
    else:
        if state['eval_type'] == 'mcqa':
            return "node_classify_mcqa_hallucination"
        else:
            return "node_add_step" # QA는 바로 add_step

def edge_check_hallucination(state):
    """ Hallucination의 에러 결과를 체크하고 재생성 또는 저장으로 이어지는 엣지 """
    logger.info(f"edge_check_hallucination: step {state['step']}")
    logger.info(f"num of errors: {len(state['error'])}")
    error_tolerance_size = int((state['error_tolerance_ratio'] / 2) * len(state['result']))
    error_tolerance_size = 1 if error_tolerance_size == 0 else error_tolerance_size
    error_size = len(state['error'])
    logger.info(f"Error tolerance: {error_size}/{error_tolerance_size}({error_size/error_tolerance_size:.2%})")
    if error_size > error_tolerance_size:
        return "node_reclassify_hallucination"

    else:
        return "node_value_grade_and_save_data"

def edge_check_next_step(state):
    """ Maxstep에 도달하였는지 체크하는 엣지 """
    logger.info(f"edge_check_next_step [max step: {state['max_step']} vs current step: {state['step']}]")

    if state['max_step'] >= state['step']:
        if state['eval_type'] == 'mcqa':
            return "node_generate_mcqa_step_n"
        else:
            return f"node_generate_qa_step_{state['step']}"
    else:
        return "end"

# =====================================================
# =====================  QA Node  =====================
# =====================================================
def node_generate_qa_step_1(state):
    """ QA에서 Reference를 기반으로 첫번째 QA를 생성하는 노드 """
    step = 1
    logger.info(f"node_generate_qa_step_1")
    data = state['data']
    error = state.get("error", None)
    show_error_log = state.get("show_error_log", True)

    state_dict = {k: v for k, v in state.items() if k != 'data'}
    param = []
    for k, v in state_dict.items():
        params = f"{k}: {v}"
        param.append(params)

    param_msg = '\n'.join(param)
    logger.info("**State Parameters**" + "\n" + param_msg)

    # qa는 항상 qa -> a -> hallucination -> preference 4개의 스텝 고정임
    # hallucination 제거 (2024. 11. 24)
    max_step = 3
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['step_1']

    result = concurrent_execute(
        partial(
            completion,
            task_type=state['task_type'],
            eval_type=state['eval_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            oai_model=oai_model
        ), iterables=data, n_workers=state['n_workers'], safe_wait=False)

    result = result[0]
    state.update({"result": result, "step": step, "max_step": max_step, "error": error, "show_error_log": show_error_log})
    return state

def node_regenerate_qa(state):
    """ QA에서 에러 Index를 재생성하는 노드 """
    step = state['step']
    logger.info(f"node_regenerate_qa: step {step}")
    data = state['error']
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['regenerate']
    result = concurrent_execute(
        partial(
            completion,
            task_type=state['task_type'],
            eval_type=state['eval_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            oai_model=oai_model
        ), iterables=data, n_workers=state['n_workers'], safe_wait=False)
    processed = state['result'] + result[0]
    state.update({"result": processed, "step": step})

    return state

# step2: Generate Only Answer with providing only question
def node_generate_qa_step_2(state):
    """ QA에서 Reference와 답변 없이 질문만으로 2번째 답변을 생성하는 노드 """
    logger.info(f"node_generate_qa_step_2 : Answer without restriction")
    step = state['step']
    data = state['data']
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['step_2']
    result = state['result']
    # 에러체크때문에 다시 Flatten해줘야함
    logger.info(f"Convert Data type for step {step}")
    converted = convert_list_to_dict(result, result[0].keys())
    converted = pd.DataFrame.from_dict(converted)
    converted = converted.to_dict("records")

    result = concurrent_execute(
        partial(
            completion,
            task_type=state['task_type'],
            eval_type=state['eval_type'],
            domain_type=state['domain_type'],
            n_datasets=state['n_datasets'],
            step=step,
            oai_model=oai_model
        ), iterables=converted, n_workers=state['n_workers'], safe_wait=False)
    result = result[0]
    state.update({"result": result, "data": data, "step": step})
    return state

# step4: Preference
def node_generate_qa_step_3(state):
    """ QA에서 Preference를 생성하는 노드 """
    logger.info(f"node_generate_qa_step_3 : Preference")
    step = state['step']
    data = state['data']
    eval_type = "preference"
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['preference']
    result = state['result']
    # 에러체크때문에 다시 Flatten해줘야함
    logger.info(f"Convert Data type for step {step}")
    result = convert_list_to_dict(result, result[0].keys())
    result = pd.DataFrame.from_dict(result)
    result = result.to_dict("records")

    result = concurrent_execute(
        partial(
            completion,
            eval_type=eval_type,
            oai_model=oai_model
        ), iterables=result, n_workers=state['n_workers'], safe_wait=False)
    result = result[0]
    state.update({"result": result, "data": data, "step": step, "eval_type": eval_type})
    return state


def node_process_qa_errors(state):
    """ QA에서 에러를 체크하는 노드 """
    logger.info(f"node_process_qa_errors: step {state['step']}")
    result = state['result']
    data = state['data']

    if state['step'] >= 2:
        result = convert_dict_to_list(result)

    # 에러 수집
    all_error = []

    # answer, question의 갯수가 다른경우
    _, num_matching_error, processed = find_num_matching_errors(eval_type=state['eval_type'], response=result)
    if state['show_error_log']:
        logger.info(f"Not nums of matching Errors:{num_matching_error}({len(num_matching_error)})")
    all_error.extend(num_matching_error)

    # 생성하라고 지시한 n_datasets와 다른 경우
    _, deficient_error, processed = find_deficient_errors(eval_type=state['eval_type'], n_datasets=state['n_datasets'], response=processed)
    if state['show_error_log']:
        logger.info(f"Deficient Errors:{deficient_error}({len(deficient_error)})")
    all_error.extend(deficient_error)

    if state['eval_type'] == 'preference':
        _, preference_error, processed = find_preference_errors(response=processed)
        if state['show_error_log']:
            logger.info(f"Preference choice Errors:{preference_error}({len(preference_error)})")
        all_error.extend(preference_error)

    # 중복 제거
    all_error = sorted(list(set(all_error)))
    if state['show_error_log']:
        logger.info(f"All Errors:{all_error}({len(all_error)})")
    if len(all_error) > 0:
        error = [item for item in data if item['index'] in all_error]

    else:
        error = []

    state.update({"result": processed, "error": error})

    return state

def mcqa_graph():
    """ MCQA Graph 생성.
    - Input할 Reference 데이터는 반드시 jsonline 포맷이어야 함
    - jsonline의 각 element는 "title"과 "contents" 값을 가져야함
    - Input할 Reference 데이터는 반드시 이전에 utils.processing.make_index 함수로 index를 생성해야함

    Examples:
        ```python
        data = pd.DataFrame(read_jsonl("sisa_encyc.jsonl"))
        data = make_index(data.to_dict("records"), prefix="시사경제용어사전")
        inputs = {
            "save_dir": ".", # 저장할 디렉토리명
            "save_file_name": "sisa_encyc", # 저장할 파일명. 파일명에 task_type과 eval_type, domain_type이 추가됨. 확장자는 자동으로 jsonl로 생성됨.
            "task_type": "common", # common, knowledge, math
            "eval_type": "mcqa", # mcqa, qa
            "domain_type": "common", # common, market, accounting, law, quant
            "n_datasets": 5, # Step별로 Refence당 생성할 MCQA 데이터셋 수
            "max_step": 2, # 몇 Step까지 실행할 것인지
            "n_workers": 300, # GPT로 생성시 Concurrent의 Worker 수
            "oai_model": "gpt-4o-mini", # gpt 모델명. Dictionary 타입으로 각 completion마다 정의 가능
            "error_tolerance_ratio": 0.03, # 허용할 에러율. 해당 에러율 이하로 에러 발생시 다음 단계로 진행
            "show_log_error": False, # 에러별로 발생한 Index를 로그로 표시
            "data": data # Reference 데이터
        }
        graph = mcqa_graph()
        result = graph.invoke(inputs=inputs, config={"recursion_limit": 30}) # Graph에서 재귀적으로 실행되는 허용치
        ```
    """
    # define graph
    workflow = StateGraph(MCQAGraphState)
    # node
    workflow.add_node("node_generate_mcqa_step_1", node_generate_mcqa_step_1)
    workflow.add_node("node_generate_mcqa_step_n", node_generate_mcqa_step_n)
    workflow.add_node("node_regenerate_mcqa", node_regenerate_mcqa)
    workflow.add_node("node_process_mcqa_errors", node_process_mcqa_errors)
    workflow.add_node("node_classify_mcqa_hallucination", node_classify_mcqa_hallucination)
    workflow.add_node("node_process_hallucination_errors", node_process_hallucination_errors)
    workflow.add_node("node_reclassify_hallucination", node_reclassify_hallucination)

    workflow.add_node("node_add_step", node_add_step)
    workflow.add_node("node_value_grade_and_save_data", node_value_grade_and_save_data)
    # edge
    workflow.set_entry_point("node_generate_mcqa_step_1")
    workflow.add_edge("node_generate_mcqa_step_1", "node_process_mcqa_errors")
    workflow.add_conditional_edges(
        "node_process_mcqa_errors",
        edge_check_error,
        {
            "node_regenerate_mcqa": "node_regenerate_mcqa",
            "node_classify_mcqa_hallucination": "node_classify_mcqa_hallucination"
        }
    )
    workflow.add_edge("node_regenerate_mcqa", "node_process_mcqa_errors")
    # Hallucination Part
    workflow.add_edge("node_classify_mcqa_hallucination", "node_process_hallucination_errors")
    workflow.add_conditional_edges(
        "node_process_hallucination_errors",
        edge_check_hallucination,
        {
            "node_reclassify_hallucination": "node_reclassify_hallucination",
            "node_value_grade_and_save_data": "node_value_grade_and_save_data"
        }
    )
    workflow.add_edge("node_reclassify_hallucination", "node_process_hallucination_errors")
    workflow.add_edge("node_value_grade_and_save_data", "node_add_step")
    workflow.add_conditional_edges(
        "node_add_step",
        edge_check_next_step,
        {
            "node_generate_mcqa_step_n": "node_generate_mcqa_step_n",
            "end": END
        }
    )
    workflow.add_edge("node_generate_mcqa_step_n", "node_process_mcqa_errors")
    app = workflow.compile()
    return app

def qa_graph():
    """ QA Graph 생성.
    - Input할 Reference 데이터는 반드시 jsonline 포맷이어야 함
    - jsonline의 각 element는 "title"과 "contents" 값을 가져야함
    - Input할 Reference 데이터는 반드시 이전에 utils.processing.make_index 함수로 index를 생성해야함

    Examples:
        ```python
        data = pd.DataFrame(read_jsonl("sisa_encyc.jsonl"))
        data = make_index(data.to_dict("records"), prefix="시사경제용어사전")
        inputs = {
            "save_dir": ".", # 저장할 디렉토리명
            "save_file_name": "sisa_encyc", # 저장할 파일명. 파일명에 task_type과 eval_type, domain_type이 추가됨. 확장자는 자동으로 jsonl로 생성됨.
            "task_type": "common", # common, knowledge, math
            "eval_type": "qa", # mcqa, qa
            "domain_type": "common", # common, market, accounting, law, quant
            "n_datasets": 5, # Refence당 생성할 QA 데이터셋 수
            "n_workers": 300, # GPT로 생성시 Concurrent의 Worker 수
            "oai_model": "gpt-4o-mini", # gpt 모델명. Dictionary 타입으로 각 completion마다 정의 가능
            "error_tolerance_ratio": 0.03, # 허용할 에러율. 해당 에러율 이하로 에러 발생시 다음 단계로 진행
            "show_log_error": False, # 에러별로 발생한 Index를 로그로 표시
            "data": data # Reference 데이터
        }
        graph = qa_graph()
        result = graph.invoke(inputs=inputs, config={"recursion_limit": 30}) # Graph에서 재귀적으로 실행되는 허용치
        ```
    """
    # define graph
    workflow = StateGraph(QAGraphState)
    # node
    workflow.add_node("node_generate_qa_step_1", node_generate_qa_step_1)
    workflow.add_node("node_generate_qa_step_2", node_generate_qa_step_2)
    workflow.add_node("node_generate_qa_step_3", node_generate_qa_step_3)
    workflow.add_node("node_regenerate_qa", node_regenerate_qa)
    workflow.add_node("node_process_qa_errors", node_process_qa_errors)
    workflow.add_node("node_add_step", node_add_step)
    workflow.add_node("node_value_grade_and_save_data", node_value_grade_and_save_data)
    # edge
    workflow.set_entry_point("node_generate_qa_step_1")
    workflow.add_edge("node_generate_qa_step_1", "node_process_qa_errors")
    workflow.add_conditional_edges(
        "node_process_qa_errors",
        edge_check_error,
        {
            "node_regenerate_qa": "node_regenerate_qa",
            "node_add_step": "node_add_step" # QA는 Step마다 저장하지 않고 바로 add_step
        }
    )
    workflow.add_edge("node_regenerate_qa", "node_process_qa_errors")
    workflow.add_conditional_edges(
        "node_add_step",
        edge_check_next_step,
        {
            "node_generate_qa_step_2": "node_generate_qa_step_2",
            "node_generate_qa_step_3": "node_generate_qa_step_3",
            "end": "node_value_grade_and_save_data"
        }
    )
    workflow.add_edge("node_generate_qa_step_2", "node_process_qa_errors")
    workflow.add_edge("node_generate_qa_step_3", "node_process_qa_errors")
    workflow.add_edge("node_value_grade_and_save_data", END)
    app = workflow.compile()
    return app

# =======================================
# ======    Classification    ===========
# =======================================
# Classification은 별개의 Node와 Edge로 운영한다

def node_generate_classification_step_1(state):
    """ Classification 생성 """
    logger.info(f"node_generate_classification_step_1")
    data = state['data']
    error = state.get("error", None)

    state_dict = {k: v for k, v in state.items() if k != 'data'}
    param = []
    for k, v in state_dict.items():
        params = f"{k}: {v}"
        param.append(params)

    param_msg = '\n'.join(param)
    logger.info("**State Parameters**" + "\n" + param_msg)

    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['step_1']

    result = concurrent_execute(
        partial(
            completion,
            eval_type='classification',
            oai_model=oai_model
        ), iterables=data, n_workers=state['n_workers'], safe_wait=False)

    result = result[0]
    state.update({"result": result, "error": error})
    return state


def node_process_classification_errors(state):
    """ Classification 에러를 체크 """
    logger.info("node_process_classification_errors")
    result = state['result']
    data = state['data']

    # 에러 수집
    all_error = []

    # answer, question의 갯수가 다른경우
    _, classification_error, processed = find_classification_errors(response=result)
    if state['show_error_log']:
        logger.info(f"Not nums of matching Errors:{classification_error}({len(classification_error)})")
    all_error.extend(classification_error)

    # 중복 제거
    all_error = sorted(list(set(all_error)))
    if state['show_error_log']:
        logger.info(f"All Errors:{all_error}({len(all_error)})")
    if len(all_error) > 0:
        error = [item for item in data if item['index'] in all_error]

    else:
        error = []

    state.update({"result": processed, "error": error})

    return state

def node_regenerate_classification(state):
    """ Classification 에러를 재생성 """
    logger.info(f"node_regenerate_classification")
    data = state['error']
    oai_model = state['oai_model']
    if isinstance(state['oai_model'], dict):
        oai_model = state['oai_model']['regenerate']
    result = concurrent_execute(
        partial(
            completion,
            eval_type='classification',
            oai_model=oai_model
        ), iterables=data, n_workers=state['n_workers'], safe_wait=False)
    processed = state['result'] + result[0]
    state.update({"result": processed})

    return state

def node_save_data_classification(state):
    """ Classification이 추가된 데이터를 저장 """
    logger.info("node_save_data_classification")
    result = state['result']
    save_path = state.get('save_file_path', None)
    if save_path is None:
        fname = state['save_file_name'].split(".")[-1]
        if fname != 'jsonl':
            fname = f"{state['save_file_name']}.jsonl"
        save_path = os.path.join(state['save_dir'], fname)
    try:
        write_jsonl(result, save_path, ensure_ascii=False, overwrite=True)
    except Exception as e:
        path = PathConfig()
        logger.error(f"{repr(e)}")
        log_dir = os.path.join(path.ROOT, "logs")
        timesuffix = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        fname = os.path.basename(save_path).split(".")[0]
        write_jsonl(result, f"{log_dir}/error_{fname}_{timesuffix}.jsonl", ensure_ascii=False, overwrite=True)

def edge_check_classification_error(state):
    logger.info(f"edge_check_classification_error")
    logger.info(f"num of errors: {len(state['error'])}")
    error_tolerance_size = int(state['error_tolerance_ratio'] * len(state['data']))
    error_tolerance_size = 1 if error_tolerance_size == 0 else error_tolerance_size
    error_size = len(state['error'])
    logger.info(f"Error tolerance: {error_size}/{error_tolerance_size}({error_size / error_tolerance_size:.2%})")
    if error_size > error_tolerance_size:
        return "node_regenerate_classification"
    else:
        return "node_save_data_classification"


def classification_graph():
    """ classfication graph 생성. classification은 Reference가 회계, 금융, 수학문제, 일반 지식 등으로 분류해준다.
    - Input할 Reference 데이터는 반드시 jsonline 포맷이어야 함
    - jsonline의 각 element는 "title"과 "contents" 값을 가져야함
    - Input할 Reference 데이터는 반드시 이전에 utils.processing.make_index 함수로 index를 생성해야함

    Examples:
        ```python
        data = pd.DataFrame(read_jsonl("hf_flare_finqa.jsonl"))
        data = make_index(data.to_dict("records"), prefix="flare-finqa")
        inputs = {
            "save_dir": ".", # 저장할 디렉토리명
            "save_file_name": "sisa_encyc", # 저장할 파일명. 파일명에 task_type과 eval_type, domain_type이 추가됨. 확장자는 자동으로 jsonl로 생성됨.
            "eval_type": "classfication", # mcqa, qa
            "oai_model": "gpt-4o-mini", # gpt 모델명. Dictionary 타입으로 각 completion마다 정의 가능
            "error_tolerance_ratio": 0.03, # 허용할 에러율. 해당 에러율 이하로 에러 발생시 다음 단계로 진행
            "show_log_error": False, # 에러별로 발생한 Index를 로그로 표시
            "data": data # Reference 데이터
        }
        graph = classification_graph()
        result = graph.invoke(inputs=inputs, config={"recursion_limit": 30}) # Graph에서 재귀적으로 실행되는 허용치
        ```
    """

    # define graph
    workflow = StateGraph(ClassificationGraphState)
    # node
    workflow.add_node("node_generate_classification_step_1", node_generate_classification_step_1)
    workflow.add_node("node_process_classification_errors", node_process_classification_errors)
    workflow.add_node("node_regenerate_classification", node_regenerate_classification)
    workflow.add_node("node_save_data_classification", node_save_data_classification)
    # edge
    workflow.set_entry_point("node_generate_classification_step_1")
    workflow.add_edge("node_generate_classification_step_1", "node_process_classification_errors")
    workflow.add_conditional_edges(
        "node_process_classification_errors",
        edge_check_classification_error,
        {
            "node_regenerate_classification": "node_regenerate_classification",
            "node_save_data_classification": "node_save_data_classification"
        }
    )
    workflow.add_edge("node_regenerate_classification", "node_process_classification_errors")
    workflow.add_edge("node_save_data_classification", END)
    app = workflow.compile()
    return app

# 기타 함수들
def save_graph(app, save_path: str):
    """ Graph를 시각화하고 png로 저장해주는 함수

    Args:
        app: graph application
        save_path: 저장위치

    Returns: None

    """
    logger.info(f"Save Graph on {save_path}")
    app.get_graph().draw_mermaid_png(output_file_path=save_path, draw_method=MermaidDrawMethod.API)

def show_graph(app):
    """ Graph를 IPython에서 시각화하여 출력해주는 함수

    Args:
        app: graph application

    Returns: None

    """
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))