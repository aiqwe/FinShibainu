""" Graph Pipeline에서 데이터 오류 체크 및 데이터 전처리, 데이터 후처리 등 관련 모듈"""

import re
import os
import random
import string
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Literal
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger
import pymupdf
from .api import calculate_tokens
from .io import read_jsonl, write_jsonl

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
REVERSE_ANSWER_MAP = {v: k for k, v in ANSWER_MAP.items()} # {"A": 0, "B": 1 ...}
WHITESPACE_PATTERN = re.compile(r"\s+")

def make_index(data: List[Dict], prefix: str):
    """ Prefix와 Incremental int 값을 기반으로 Index를 생성한다. 입력값은 jsonline 형태를 따라야한다

    Args:
        data: index를 생성하려는 데이터
        prefix: index에 추가될 접두사. 예) prefix="sisa" -> sisa_0, sisa_1 ...

    Returns: index가 추가된 jsonline

    """
    df = pd.DataFrame(data)
    df['index'] = [f"{prefix}_{i}" for i in range(len(df))]
    return df.to_dict("records")

def convert_list_to_dict(result: List[Dict], keys: List[str]):
    """ Concurrent로 얻은 GPT의 Completion을 Dataset으로 저장하기 위해 dict형태로 변환.
    concurrent execute 함수는 첫번째 요소가 결과값, 두번째 요소가 에러가 발생한 요소이므로,
    result[0]을 파라미터로 받아야함.
    예시)
    result[0] = [
        {"question": ['충당부채를 계산할때 ...', '전환사채를 평가할때 ...', '신주인수권부사채를 매도할때 ...']},
        {"question": ['리스회계의 적용 ...', '사용권자산상각 ...', '리스부채를 인식 ...']}
    ]의 형태를 입력으로 받으며
    {"question": [
        충당부채를 계산할때 ...', '전환사채를 평가할때 ...', '신주인수권부사채를 매도할때 ...',
        '리스회계의 적용 ...', '사용권자산상각 ...', '리스부채를 인식 ...'
    ]}의 형태로 변환함

    Args:
        result: Concurrent 실행 결과값
        keys: dict로 변환될 key값

    Returns: Dictionary 형태

    """
    converted = defaultdict(list)
    for key in keys:
        converted[key] = [item2 for item1 in result for item2 in item1[key]]
    return converted

def convert_dict_to_list(result: Dict, index='index'):
    """ shuffle answer 등을 사용하려면 Dict로 변환해야하는데, qa step2 등에서 에러 체크할 땐 다시 List[Dict[List]]로 변환해야함
    - Input 값: {"question": [충당부채를 계산할때 ...', '전환사채를 평가할때 ...', '신주인수권부사채를 매도할때 ...',
    '리스회계의 적용 ...', '사용권자산상각 ...', '리스부채를 인식 ...']}형태의 값(convert_list_to_dict의 결과값)
    - Output 값: result = [{"question": ['충당부채를 계산할때 ...', '전환사채를 평가할때 ...', '신주인수권부사채를 매도할때 ...']},
    {"question": ['리스회계의 적용 ...', '사용권자산상각 ...', '리스부채를 인식 ...']}]의 형태로 반환함

    Args:
        result: convert_list_to_dict의 결과겂
        keys: grouping할 index값

    Returns : index별로 그룹핑된 List[Dict]형태의 값

    """
    df = pd.DataFrame.from_dict(result)
    grouped = df.groupby(index).agg(lambda x: list(x)).reset_index().to_dict("records")
    for row in grouped:
        cnt = len(row['question'])
        row['index'] = [row['index']] * cnt
    return grouped

def find_answer_errors(response: List[Dict]):
    """ answer의 요소들 중 ['A', 'B', 'C', 'D', 'E', 'F', 'G']가 아닌 것들을
    1) 찾아내서 출력하고, 2) 전처리하고, 3) 원본 반환함
    반환하는 첫번째요소는 원본, 두번째 요소는 이상데이터, 세번째 요소는 이상데이터가 제거된 데이터
    find_* 함수들은 동일한 리턴값을 갖는다
    """
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []
    for item in target:
        try:
            for answer, option in zip(item['answer'], item['options']):
                if answer not in REVERSE_ANSWER_MAP:
                    delete_idx.extend(item['index'])
                options_first_chracter = [o[0] for o in option]
                if answer not in options_first_chracter: # 선택지 A. B. C. ... 에 없는 답변을 내놓는경우
                    delete_idx.extend(item['index'])
        except:
            delete_idx.extend(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)


def find_option_errors(response: List[Dict]):
    """ options중 "A. blahblah"의 형태가 아닌 경우를 에러롤 리턴"""
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []

    for item in target:
        try:
            for option in item['options']:
                if not (4 <= len(option) <= 7):
                    delete_idx.extend(item['index'])
                for idx, o in enumerate(option):
                    if not re.match(r'^[A-G]\. ', o): # A-G로시작하지 않거나
                        delete_idx.extend(item['index'])
                    if (o is None) or (chr(idx + 65) != o[0]): # 순서에 맞게 A, B, C, ...가 아니거나
                        delete_idx.extend(item['index'])
        except:
            delete_idx.extend(item['index'])


    delete_idx = sorted(list(set(delete_idx)))

    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_num_matching_errors(eval_type: Literal['mcqa', 'qa'], response: List[Dict]):
    """ question, answer, options의 갯수가 하나라도 일치하지 않으면 에러로 반환"""
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []
    for item in target:
        try:
            if eval_type == 'mcqa':
                match = len(item['question']) == len(item['answer']) == len(item['options']) == len(item['reasoning_process'])
            else:
                match = len(item['question']) == len(item['answer'])
            if not match:
                delete_idx.extend(item['index'])
        except:
            delete_idx.extend(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_deficient_errors(eval_type: Literal['mcqa', 'qa'], response: List[Dict], n_datasets: int):
    """ question, answer, options가 n_datasets와 일치하지 않으면 Error로 리턴"""
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []
    for item in target:
        try:
            if eval_type == 'mcqa':
                match = n_datasets == len(item['question']) == len(item['answer']) == len(item['options']) == len(item['reasoning_process'])
            else:
                match = n_datasets == len(item['question']) == len(item['answer'])
            if not match:
                delete_idx.extend(item['index'])
        except:
            delete_idx.extend(item['index'])


    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_reasoning_errors(response: List[Dict]):
    """ reasoning에서 답변을 언급하면 에러로 반환, shuffle_answer의 에러를 막기 위함"""

    def _get_option_list(option_len):
        return [chr(ord('A') + i) for i in range(option_len)]  # ['A', 'B', 'C', 'D']

    def _alpha_or_digit(char):
        english_letters = string.ascii_letters  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = string.digits  # '0123456789'
        return (char in english_letters) or (char in digits)

    def _is_invalid_opt_reasoning(opt: str, reasoning_process: str):
        if opt not in reasoning_process:  # 영문자 'A' char가 한개도 없으면 False
            return False

        for i, c in enumerate(reasoning_process):
            if c == opt:  # 영문자가 있음
                if i == 0:  # 첫글자임
                    if not _alpha_or_digit(reasoning_process[i + 1]):  # 다음 글짜가 한글자 alphanumeric이 아님
                        return True
                elif i == len(reasoning_process) - 1:
                    if not _alpha_or_digit(reasoning_process[i - 1]):  # 이전 글짜가 한글자 alphanumeric이 아님
                        return True
                else:
                    if (not _alpha_or_digit(reasoning_process[i - 1]) and
                            (not _alpha_or_digit(reasoning_process[i + 1]))):  # 다음 앞뒤로 한글자 alphanumeric이 아님
                        return True
        return False

    original = deepcopy(response)
    target = deepcopy(response)
    temp_dict = convert_list_to_dict(response, keys=response[0].keys())
    temp_df = pd.DataFrame.from_dict(temp_dict)

    delete_idx = []
    for i, row in temp_df.iterrows():
        try:
            option_list = _get_option_list(len(row['options']))  # option_list = ['A', 'B', 'C', 'D]

            for o in option_list:
                if _is_invalid_opt_reasoning(o, row['reasoning_process']): # 유효하지 않은 reasoning이 있으면 index를 추가한다
                    delete_idx.append(row['index']) # 다른 error는 ['index_1', 'index_1'] 자체를 추가해서 extend지만 얘는 row의 개별 'index_1'이기 때문에 append
                    break
        except:
            delete_idx.append(row['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target

def find_hallucination_errors(response: List[Dict]):
    """ hallucination의 사전 정의 값으로 판단되는지 분류
    Flatten된 데이터가 in/out 됨
    """

    original = deepcopy(response)
    target = deepcopy(response)

    # 삭제할 인덱스 찾기
    delete_idx = []

    for item in target:
        try:
            if item['hallucination'] not in ('fake_answer', 'fake_references', 'deficient_question', 'deficient_answer', 'usable'):
                delete_idx.append(item['index'])
        except:
            delete_idx.append(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_hallucination_filters(response: List[Dict]):
    """ hallucination의 사전 정의 값으로 판단되는지 분류
    Flatten된 데이터가 in/out 됨
    """

    original = deepcopy(response)
    target = deepcopy(response)

    # 삭제할 인덱스 찾기
    delete_idx = []

    for item in target:
        try:
            if item['hallucination'] not in ('usable'):
                delete_idx.append(item['index'])
        except:
            delete_idx.append(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_preference_errors(response: List[Dict]):
    """ 사전 정의된 preference 분류 (A, B, E)가 일치하지 않으면 에러를 리턴 """
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []
    for item in target:
        try:
            for pre in item['preference']:
                if pre not in ('A', 'B', 'E'):
                    delete_idx.extend(item['index'])
        except:
            delete_idx.extend(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'][0] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)

def find_classification_errors(response):
    """ 사전 task_type에 정의된 타입이 맞는지 체크"""
    original = deepcopy(response)
    target = deepcopy(response)
    # 삭제할 인덱스 찾기
    delete_idx = []
    for item in target:
        try:
            if item['classification'] not in ('knowledge_accounting', 'knowledge_market', 'math_accounting', 'math_market', 'quant', 'law', 'no'):
                delete_idx.append(item['index'])
        except:
            delete_idx.append(item['index'])

    delete_idx = sorted(list(set(delete_idx)))

    # 삭제하기
    target = [item for item in target if item['index'] not in delete_idx]

    return original, delete_idx, target  # (원본, 삭제할 인덱스, 전처리된 데이터)


def add_conclusion(df):
    """ reasoning_process에 Answer의 결론을 마지막에 추가함

    Args:
        df: to_dict("records")로 최종 저장 전 DataFrame

    """

    def _get_conclusion(answer):
        candidates = [f"그래서 답은 {answer} 입니다.",
                      f"그러한 이유로 정답은 {answer} 입니다.",
                      f"따라서 정답은 {answer} 입니다.",
                      f"이런 이유로 {answer} 가 정답입니다.",
                      f"결론적으로 정답은 {answer} 입니다.",
                      f"그러므로 {answer} 를 정답으로 선택해야 합니다.",
                      f"위의 근거로 정답은 {answer} 입니다.",
                      f"그렇기 때문에 {answer} 가 정답입니다.",
                      f"이런 점에서 {answer} 가 맞는 답입니다.",
                      f"논리적으로 보면 답은 {answer} 입니다.",
                      f"분석해 보면 정답은 {answer} 입니다.",
                      f"앞서 언급한 이유로 {answer} 가 정답입니다.",
                      f"모든 것을 고려할 때, 답은 {answer} 입니다.",
                      f"위의 내용을 종합해보면 정답은 {answer} 입니다.",
                      f"결론적으로 판단하면 {answer} 가 답입니다.",
                      f"이유를 종합하면 {answer} 가 올바른 답입니다.",
                      f"요약하자면 정답은 {answer} 입니다.",
                      f"설명을 근거로 보면 {answer} 가 정답입니다.",
                      f"위에서 논의한 바와 같이 답은 {answer} 입니다.",
                      f"그러한 점에서 {answer} 를 답으로 선택해야 합니다."]
        return random.choice(candidates)

    df['reasoning_process'] = df['reasoning_process'] + " " + df['answer'].apply(lambda x: _get_conclusion(x))
    return df


def shuffle_answer(converted: dict):
    """ List -> Dict로 Convert된 상태에서, Answer의 Bias를 없애기 위해 셔플링함
    Args:
        converted: Dict로 Convert 된 결과값
    """
    target = deepcopy(converted)
    for i, (index, option, answer) in enumerate(zip(target['index'], target['options'], target['answer'])):
        try:
            org_index = list(range(len(option)))
            new_index = deepcopy(org_index)
            random.shuffle(new_index)  # Shuffle -> [0, 1, 2, 3] => [3, 0, 2, 1]

            shuffle_mapper = {ANSWER_MAP[org]: ANSWER_MAP[new] for org, new in zip(org_index, new_index)} # {A: C, B: D ...}
            shuffle_mapper = {v: k for k, v in shuffle_mapper.items()} # shuffle_mapper는 A가 0번째에서 어디 위치로 갔는가? 를 나타냄
            target['answer'][i] = shuffle_mapper[answer]  # 변경된 answer를 저장
            target['options'][i] = [f"{chr(65 + j)}{option[n][1:]}" for j, n in enumerate(new_index)]
        except Exception as e:
            logger.error(repr(e) + "\n" + f"index: {index}")
            continue
    return target

def shuffle_jsonl_answer(converted: list, index="index"):
    """ List -> Dict로 Convert된 상태에서, Answer의 Bias를 없애기 위해 셔플링함
    Args:
        converted: Dict로 Convert 된 결과값
    """
    target = deepcopy(converted)
    for item in target:
        try:
            org_index = list(range(len(item['options'])))
            new_index = deepcopy(org_index)
            random.shuffle(new_index)  # Shuffle -> [0, 1, 2, 3] => [3, 0, 2, 1]

            shuffle_mapper = {ANSWER_MAP[org]: ANSWER_MAP[new] for org, new in zip(org_index, new_index)} # {A: C, B: D ...}
            shuffle_mapper = {v: k for k, v in shuffle_mapper.items()} # shuffle_mapper는 A가 0번째에서 어디 위치로 갔는가? 를 나타냄
            item['answer'] = shuffle_mapper[item['answer']]  # 변경된 answer를 저장
            item['options'] = [f"{chr(65 + j)}{item['options'][n][1:]}" for j, n in enumerate(new_index)]
        except Exception as e:
            logger.error(repr(e) + "\n" + f"index: {item[index]}")
            continue
    return target

def normalize_text(text, mode: Literal["white_space"] = None):
    """ Text Normalize 함수

    Args:
        text: Normalize할 함수
        mode: Normalize 방식 선택

    Returns: Normalize된 텍스트

    """
    if not mode:
        raise ValueError(f"{mode=}\ncheck parameters")

    if mode == "white_space":
        return re.sub(WHITESPACE_PATTERN, " ", text)

def extract_report(doc = None, doc_type: Literal['html', 'xml'] = 'html'):
    """ 공시자료에서 사업의 내용만 추출하는 코드

    Args:
        doc: 공시자료 텍스트
        doc_type: 공시자료의 형태
    """
    parser = "lxml" if doc_type == 'xml' else "html.parser"
    soup = BeautifulSoup(doc, parser)
    # business_content_section = soup.find('title', string='II. 사업의 내용').find_parent('section-1')
    if doc_type == 'xml':
        business_section = soup.find('title', string=re.compile('사업의 내용'))
        if business_section:
            # 해당 섹션의 부모 태그인 section-1을 찾음
            section = business_section.find_parent('section-1')
            if section:
                # section-1 태그 내의 모든 <table> 태그를 제거
                for table in section.find_all('table'):
                    table.decompose()
                # section-1 태그 내의 모든 텍스트 추출
                business_content = section.get_text(separator='\n', strip=True)
                text = normalize_text(business_content, mode="white_space")
            return business_content

    if doc_type == 'html':
        business_content_section = soup.find('body')

        # table은 제거
        for table in business_content_section.find_all('table'):
            table.decompose()
        business_content_text = business_content_section.get_text(separator='\n', strip=True)
        text = normalize_text(business_content_text, mode="white_space")
        return text

def show_duplicates(df_or_path, deduplicate=True, subset="title", keep="first"):
    """ 중복값을 제거해주는 함수

    Args:
        df_or_path: Pandas DataFrame 또는 Pandas DataFrame으로 읽을 파일명
        deduplicate: 중복 제거값을 반환할 것인지 여부
        subset: 중복 제거할 subset 컬럼명(pandas의 drop_duplicates의 인수)
        keep: 중복 제거시 유지할값의 선택 방식(pandas의 drop_duplicates의 인수)

    Returns: None

    """
    if isinstance(df_or_path, str):
        df = pd.DataFrame(read_jsonl(df_or_path))
        print(f"df_or_path: \033[38;5;207m{df_or_path}\033[0m")
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    else:
        try:
            df = pd.DataFrame(df_or_path)
        except:
            raise ValueError("cant not infer type of df_or_path")
    print(f"ORIGINAL LENGTH: \033[38;5;118m{len(df)}\033[0m")
    print(f"DEDUPLICATED LENGTH: \033[38;5;63m{len(df.drop_duplicates(subset='title', keep='first'))}\033[0m")
    if deduplicate:
        return df.drop_duplicates(subset=subset, keep=keep)

def split_chunk(obj: List, n: int, by: Literal["size", "num"] = "num"):
    """ 이터러블(보통 List)를 청크로 분할

    Args:
        obj: 분할할 객체
        n: 분할할 청크 수, 또는 1개 청크의 크기
        by: 분할할 방법 정의
          - num: 청크 수에 따라 분할
          - size: 청크당 갯수에 따라 분할

    Returns: Chunk로 분할된 Iterable. 예) [1, 2, 3, 4] -> n=2, by="num"일 시 -> [[1, 2], [3, 4]]
    """
    if by == "num":
        # 균등하게 분할하기 위해 청크 크기와 나머지를 계산
        chunk_size, remainder = divmod(len(obj), n)
        # 첫 remainder개의 청크는 chunk_size + 1로 설정
        return [obj[i * (chunk_size + 1): (i + 1) * (chunk_size + 1)] if i < remainder else
                obj[remainder * (chunk_size + 1) + (i - remainder) * chunk_size: remainder * (chunk_size + 1) + (i - remainder + 1) * chunk_size]
                for i in range(n)]
    elif by == "size":
        return [obj[i:i + n] for i in range(0, len(obj), n)]


def split_chunk_with_index(obj: List, n: int, by: Literal["size", "num"] = "num", embedding_index: bool = False):
    """ 이터러블(보통 List)를 청크로 분할하며, 청크당 index를 부여

    Args:
        obj: 분할할 객체
        n: 분할할 청크 수, 또는 1개 청크의 크기
        by: 분할할 방법 정의
          - num: 청크 수에 따라 분할
          - size: 청크당 갯수에 따라 분할
        embedding_index: index를 iterable 내부에 포함시킬지
          - embedding=False: [(0, iterable1), (1, iterable2) ...]
          - embedding=True: [iterable1, iterable2 ...]로 생성되며 각 iterable1 = {"chunk_index": 0, "question": ...}

    Returns: Chunk로 분할된 Iterable에 index를 추가.
      - by="num"예) [1, 2, 3, 4] -> n=2, by="num"일 시 -> [[1, 2], [3, 4]]
    """
    chunks = split_chunk(obj=obj, n=n, by=by)
    chunks = [(idx, chunk) for idx, chunk in enumerate(chunks)]
    if embedding_index:
        for chunk_index, data in chunks:
            for element in data:
                element['chunk_index'] = chunk_index
        chunks = [chunk[1] for chunk in chunks]
        result = []
        for chunk in chunks:
            result.extend(chunk)
        chunks = result
    return chunks

def value_control(df, mapper: dict, index: str = "reference_index", log=True):
    """ reference_index를 파티션으로 value 필터링 수행
    mapper의 예시는 다음과 같음
    value_controller = {
    'booklist': 3,
    'hf_stock_trading_QA_knowledge_market': 5,
    'krx_stock_beginner': 3,
    ...
}

    Args:
        df: 필터링할 데이터프레임
        mapper: value mapper
        index: 필터링할 인덱스 기준값
        log: 로그 출력 여부

    """
    to_concat = []
    for k in mapper:
        references = df[df['reference_index'] == k]
        alive = references[references['value'] >= mapper[k]]
        to_concat.append(alive)
        if log:
            logger.info(
                f"{k} : value({mapper[k]}) | alive({len(alive)}) | total({len(references)}) | {len(alive) / len(references):.2%}")

    result = pd.concat(to_concat)

    return result

def show_sample(df, n=5, eval_type = None, all_column=False, sample=True):
    """ df의 Sample을 시각화 해주는 함수

    Args:
        df: sampling할 Pandas DataFrame
        n: Sample 수
        eval_type: Sampling할 데이터의 eval_type. mcqa, qa, referenes 등
        all_column: 모든 컬럼을 출력할지 여부
        sample: 랜덤샘플링 여부

    Returns: None

    """
    from colorama import Fore, Style
    if eval_type is None:
        eval_type = df['type'].drop_duplicates().item()

    if eval_type == 'mcqa':
        if sample:
            sample = df.sample(n)
        else:
            sample = df.head(n)
        for idx, row in sample.iterrows():
            if all_column:
                for k, v in row.items():
                    print(Fore.MAGENTA + k + Style.RESET_ALL)
                    print(v)
                print(Fore.GREEN + "=" * 100 + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"{idx}번째 Sample" + Style.RESET_ALL)
                print(Fore.MAGENTA + "question:" + Style.RESET_ALL)
                print(row['question'])
                print(Fore.MAGENTA + "options:" + Style.RESET_ALL)
                print("\n".join(row['options']))
                print(Fore.MAGENTA + "reasoning_process:" + Style.RESET_ALL)
                print(row['reasoning_process'])
                print(Fore.MAGENTA + "answer:" + Style.RESET_ALL)
                print(row['answer'])
                if row.get("hallucination", None):
                    print(Fore.MAGENTA + "hallucination:" + Style.RESET_ALL)
                    print(row['hallucination'])
                    print(row['hallucination_desc'])
                print(Fore.MAGENTA + "value:" + Style.RESET_ALL)
                print(row['value'])
                print(Fore.GREEN + "="*100 + Style.RESET_ALL)

    if eval_type == 'qa':
        sample = df.sample(n)
        for idx, row in sample.iterrows():
            if all_column:
                for k, v in row.items():
                    print(Fore.MAGENTA + k + Style.RESET_ALL)
                    print(v)
                print(Fore.GREEN + "=" * 100 + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"{idx}번째 Sample" + Style.RESET_ALL)
                print(Fore.MAGENTA + "question:" + Style.RESET_ALL)
                print(row['question'])
                print(Fore.MAGENTA + "old_answer:" + Style.RESET_ALL)
                print(row['old_answer'])
                print(Fore.MAGENTA + "answer:" + Style.RESET_ALL)
                print(row['answer'])
                print(Fore.MAGENTA + "preference:" + Style.RESET_ALL)
                print(row['preference'])
                print(row['preference_desc'])
                print(Fore.MAGENTA + "value:" + Style.RESET_ALL)
                print(row['value'])
                print(Fore.GREEN + "="*100 + Style.RESET_ALL)

    if eval_type == 'references':
        sample = df.sample(n)
        for idx, row in sample.iterrows():
            if all_column:
                for k, v in row.items():
                    print(Fore.MAGENTA + k + Style.RESET_ALL)
                    print(v)
                print(Fore.GREEN + "=" * 100 + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"{idx}번째 Sample" + Style.RESET_ALL)
                print(Fore.MAGENTA + "title:" + Style.RESET_ALL)
                print(row['title'])
                print(Fore.MAGENTA + "contents:" + Style.RESET_ALL)
                print(row['contents'])
                print(Fore.GREEN + "="*100 + Style.RESET_ALL)

def show_spec(df, column: Literal["classification", "hallucination", "preference", "value"]):
    """ Validation Set의 특정 컬럼의 집계 결과를 시각화하여 보여주는 함수

    Args:
        df: 시각화하려는 Pandas DataFrame
        column: 집계하려는 컬럼명

    Returns: None

    """
    from colorama import Fore, Style
    length = len(df)
    unique = df[column].drop_duplicates().to_list()
    print(Fore.MAGENTA + f"total cnt: " + Style.RESET_ALL + f"{length}")
    print(Fore.MAGENTA + f"{column}: " + Style.RESET_ALL + f"{unique}")
    print("=" * 100)
    grouped = df.groupby(column)[column].count()
    for k, v in grouped.items():
        print(Fore.CYAN + f"{k}: " + Style.RESET_ALL + str(v))

def remove_unicode(text):
    """
    입력 문자열에서 한글, 숫자, 영어, 공백, 기본 특수문자를 제외한
    모든 유니코드 문자를 제거하는 함수
    """
    # 한글, 숫자, 영어, 공백, 특정 기본 특수문자만 허용
    pattern = re.compile(r"[^\uAC00-\uD7A3\u3131-\u318E\u1100-\u11FFa-zA-Z0-9\s.,!?'\"]+")
    # 위 패턴은 한글 음절(U+AC00-U+D7A3), 호환 자모(U+3131-U+318E), 현대 한글 자모(U+1100-U+11FF)
    # 그리고 공백, 마침표, 쉼표, 느낌표, 물음표, 작은따옴표, 큰따옴표를 허용
    return pattern.sub('', text)

def filter_english(text, ratio=0.5):
    """
    주어진 텍스트에서 영어 비중이 절반 이상인지 확인하는 함수.
    """
    # 알파벳과 숫자 등 영어 문자의 개수를 계산
    english_chars = len(re.findall(r'[A-Za-z]', text))
    # 전체 문자 길이 (공백 제외)
    total_chars = len(re.findall(r'\S', text))

    if total_chars == 0:  # 내용이 없을 경우 처리
        raise ValueError("Total number of string is zero.")

    # 영어 문자 비율 계산
    english_ratio = english_chars / total_chars

    # 영어가 절반 이상인 경우 True 반환
    return english_ratio < ratio

def filter_number(text, ratio=0.5):
    # 숫자 개수와 전체 문자 개수를 계산
    digit_count = sum(char.isdigit() for char in text)
    total_count = len(text)

    # 총 문자가 0인 경우 숫자 비율 계산 방지
    if total_count == 0:
        raise ValueError("Total number of string is zero.")

    # 숫자 비율이 50%를 넘는지 확인
    number_ratio = digit_count / total_count
    return number_ratio < ratio

def filter_punctuation(text, ratio=0.5):
    # 특수 문자 정의
    special_chars = string.punctuation
    # 특수 문자 개수와 전체 문자 개수 계산
    special_char_count = sum(char in special_chars for char in text)
    total_count = len(text)

    # 총 문자가 0인 경우 숫자 비율 계산 방지
    if total_count == 0:
        raise ValueError("Total number of string is zero.")

    # 숫자 비율이 50%를 넘는지 확인
    number_ratio = special_char_count / total_count
    return number_ratio < ratio

def quality_filter(
        data,
        title: str = "",
        if_filter_punctuation: bool = True,
        filter_punctuation_ratio=0.5,
        if_filter_english: bool = True,
        filter_english_ratio=0.5,
        if_filter_number: bool = True,
        filter_number_ratio=0.5,
        if_remove_unicode: bool = True,
        if_normalize: bool = True,
        token_threshold: int = 200,
        return_type: Literal['full', 'split', 'chunk'] = "split",
        chunk_size = 10000

):
    """
    Qualtiy Filtering을 수행하여 의미있는 데이터만 필터링하는 함수
    data Parameter는 Graph Pipeline의 Input 형태와 같이 Json line의 형태로 구성되어야 함:
      - [{"title": title, "contents": contents}, {"title": title, "contents": contents}...]
    다음 필터링을 선택적으로 적용할 수 있음
      - filtering_english: 텍스트 내 영어가 filter_english_ratio 이상이면 제외
      - filtering_number: 텍스트 내 숫자가 filter_number_ratio 이상이면 제외
      - filter_punctuation: 텍스트 내 Punctuation Chracter가 일정 filter_punctuation_ratio 이상이면 제외
        - string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' 기준
      - number filtering

    Args:
        data:
        title:
        if_filter_punctuation: 구둣점 필터링 적용 여부
        filter_punctuation_ratio: 구둣점 필터링을 적용할 텍스트 내에 구둣점 구성비율 기준
        if_filter_english: 영어 필터링 적용 여부
        filter_english_ratio: 영어 필터링을 적용할 텍스트 내에 영어 구성비율 기준
        if_filter_number: 숫자 필터링 적용 여부
        filter_number_ratio: 숫자 필터링을 적용할 텍스트 내에 숫자 구성비율 기준
        if_remove_unicode: 입력 문자열에서 한글, 숫자, 영어, 공백, 기본 특수문자를 제외한 모든 유니코드 문자를 제거할지 여부
        if_normalize: 텍스트의 공백을 제거할 것인지 여부
        token_threshold: data내의 개별 contents의 토큰을 o200k_base로 계산하고 해당 토큰 미만일 경우 해당 contents제외
        return_type: 반환 타입
          - full: data의 filtering 적용후 결과 contents를 모두 이어서 하나의 텍스트로 반환
          - split: data의 filtering 적용후 각 data의 element 형태를 유지한채로 반환
          - chunk: data의 filtering 적용후 하나의 텍스트로 이어붙인다음 chunk_size 크기(문자열 단위)로 분할하여 제공
        chunk_size: return_type = 'chunk'일 시 하나의 chunk의 문자 수

    Returns: filtering이 완료된 data

    """
    # 1. get data
    for idx, item in enumerate(data):
        if 'title' not in item.keys():
            raise ValueError(f"'title' is required.{idx=}th item doesnt have 'title'")
        if 'contents' not in item.keys():
            raise ValueError(f"'contents' is required.{idx=}th item doesnt have 'contents'")
    # 2. Punctuation filter
    if if_filter_punctuation:
        data = [item for item in data if filter_punctuation(item['contents'], ratio=filter_punctuation_ratio)]
    # 3. english filter
    if if_filter_english:
        data = [item for item in data if filter_english(item['contents'], ratio=filter_english_ratio)]
    # 4. number filter
    if if_filter_number:
        data = [item for item in data if filter_number(item['contents'], ratio=filter_number_ratio)]
    # 5. unicode filter
    if if_remove_unicode:
        for item in data:
            item['contents'] = remove_unicode(item['contents'])

    if if_normalize:
        for item in data:
            item['contents'] = normalize_text(item['contents'], mode="white_space")
    for item in data:
        item['tokens'] = calculate_tokens(item['contents'])
    data = [item for item in data if item['tokens'] > token_threshold]

    if return_type == "full":
        full_context = "".join([item['contents'] for item in data])
        return {"title": title, "contents": full_context, "tokens": calculate_tokens(full_context)}

    if return_type == "split":
        result = []
        for seq_index, item in enumerate(data):
            result.append(
                {"seq": seq_index, "title": title, "contents": item['contents'], "tokens": item['tokens']})
        return result

    if return_type == "chunk":
        full_context = "".join([item['contents'] for item in data])
        chunks = split_chunk(full_context, n=chunk_size, by="size")
        result = []
        for seq_index, item in enumerate(chunks):
            result.append({"seq": seq_index, "title": title, "contents": item, "tokens": calculate_tokens(item)})
        return result







