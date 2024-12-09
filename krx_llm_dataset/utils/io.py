""" 파일의 Save, Load를 위한 io 모듈 """

import os
import json
from glob import glob
from typing import Any, List, Literal
from pathlib import Path
import hashlib
from io import StringIO
from loguru import logger
import pandas as pd
import pymupdf

def _split_chunk(obj: List, n: int, by: Literal["size", "num"] = "num"):
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


def _split_chunk_with_index(obj: List, n: int, by: Literal["size", "num"] = "num", embedding_index: bool = False):
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
    chunks = _split_chunk(obj=obj, n=n, by=by)
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



def write_json(obj: Any, path: str, end_newline = False, overwrite = True, ensure_ascii = False, log=True, **kwargs):
    """ 객체를 JSON으로 저장합니다.

    Args:
        obj: 저장할 객체
        path: 저장할 위치
        overwite: 덮어쓰기 여부
        ensure_ascii: ascii로부터 로드됨을 보장하는지 여부, 영어가 아닌 경우 False로 지정해야합니다
        end_newline: 끝에 Newline 추가 여부
        **kwargs: json.dump에 전달될 추가 파라미터

    Returns: None
    """
    mode = "a+"
    if overwrite:
        mode = "w+"
    with open(path, mode) as f:
        json.dump(obj, f, ensure_ascii = ensure_ascii, **kwargs)
        f.write("\n")
    if log:
        logger.info(f"save json file in {path}")

def read_json(path: str, log=True, **kwargs):
    """ JSON 파일을 읽어옵니다.

    Args:
        path: 읽어올 JSON 파일 위치
        **kwargs: json.load에 전달될 추가 파라미터

    Returns: json 객체

    """

    with open(path, "r+") as f:
        result = json.load(f, **kwargs)
    if log:
        logger.info(f"read json file in {path}")
    return result

def write_jsonl(obj: List, path: str, overwrite = True, ensure_ascii = False, log=True, **kwargs):
    """ List 형태의 객체를 JSON LINE 파일로 저장합니다.

    Args:
        obj: 저장할 객체
        path: 저장할 위치
        overwite: 덮어쓰기 여부
        ensure_ascii: ascii로부터 로드됨을 보장하는지 여부, 영어가 아닌 경우 False로 지정해야합니다
        **kwargs: json.dump에 전달될 추가 파라미터

    Returns: None
    """
    mode = "a+"
    if overwrite:
        mode = "w+"
    with open(path, mode) as f:
        for line in obj:
            json_line = json.dumps(line, ensure_ascii = ensure_ascii, **kwargs)
            f.write(json_line + "\n")
    if log:
        logger.info(f"save jsonl file in {path}")

def read_jsonl(path: str, log=True, **kwargs):
    """ JSON LINE 파일을 제너레이터로 읽어옵니다.

    Args:
        path: 읽어올 JSON LINE  파일 위치
        **kwargs: json.load에 전달될 추가 파라미터

    Returns: 읽어온 json line 파일을 Generator로 반환

    """

    with open(path, "r+") as f:
        if log:
            logger.info(f"read jsonl file in {path}")
        for line in f:
            json_obj = json.loads(line.strip(), **kwargs)
            yield json_obj

def read_multi_jsonl(folder: str, return_type: Literal['pandas', 'jsonl'] = "pandas", log=True, **kwargs):
    """ 여러 JSON LINE 파일들을 읽어서 하나의 객체로 리턴합니다

    Args:
        folder: 읽어올 JSON LINE 파일들이 저장된 폴더
        return_type: jsonline 또는 pandas로 읽어올지 선택
        **kwargs: json.load에 전달될 추가 파라미터

    Returns: jsonline 형태 또는 pandas 형태

    """
    files = [os.path.join(dp, f) for dp, _, filenames in os.walk(folder) for f in filenames if f.endswith(".jsonl")]
    to_concat = []
    files = [file for file in files if ".ipynb_checkpoints" not in file]
    for file in files:
        obj = read_jsonl(file, log=log, **kwargs)
        if return_type in ("pandas", "pd"):
            obj = pd.DataFrame(obj)
        to_concat.append(obj)
    return pd.concat(to_concat) if return_type == "pandas" else to_concat

def write_multi_jsonl(
        path_or_data: str,
        directory: str,
        save_file_name: str,
        max_size_mb: int = 10,
        overwrite=True,
        ensure_ascii=False,
        log=True,
        **kwargs
):
    """ 객체를 여러 JSON LINE 파일로 분할하여 저장합니다.

    Args:
        path_or_data: 분할하여 저장할 객체 또는 파일
        directory: 저장할 JSON LINE들의 폴더
        save_file_name: 저장할 JSON LINE의 파일명, {file_name}_{idx}.jsonl로 저장됩니다.
        max_size_mb: 파일당 최대 MB 크기, 실제 파일 크기와 일치하지 않을 수 있습니다.
        **kwargs: json.dump에 전달될 추가 파라미터

    Returns: None

    """

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not isinstance(path_or_data, str):
        temp_file_name = hashlib.sha1(directory.encode()).hexdigest()
        temp_file_name = f"{directory}/{temp_file_name}.jsonl"
        write_jsonl(path_or_data, temp_file_name)
        # 파일 크기 측정
        file_size = os.path.getsize(temp_file_name)
        target = path_or_data
        os.remove(temp_file_name)
    if isinstance(path_or_data, str):
        ext = os.path.splitext(path_or_data)[-1].strip(".")
        if ext == "json":
            target = read_json(path_or_data, log=log)
        if ext == "jsonl":
            target = read_jsonl(path_or_data, log=log)
        if os.path.isdir(path_or_data):
            target = read_multi_jsonl(path_or_data, log=log)
        file_size = os.path.getsize(path_or_data)

    split_size = int(file_size / (max_size_mb * 1024 * 1024)) + 1
    if not isinstance(target, list):
        target = list(target)

    chunks = _split_chunk_with_index(target, n=split_size, by="num")
    for idx, chunk in chunks:
        write_jsonl(
            chunk,
            os.path.join(directory if directory else ".", f"{save_file_name}_{idx}.jsonl"),
            ensure_ascii=ensure_ascii,
            overwrite=overwrite,
            log=log,
            **kwargs
        )

def get_pdf_text(fpath, title=""):
    """ pdf를 각 페이지별로 읽어와서 jsonline 데이터로 변환하는 함수. 각 Item은 Page 단위로 생성된다.
    포맷은 [{"title": title, "contents": contents}, ...] 형태로 생성되며, 빈페이지는 제거된다

    Args:
        fpath: 읽어올 PDF의 파일 Path
        title: 모든 페이지에 공통적으로 포함될 title명 지정
        normalize: contents 텍스트의 whitespace를 normalize할 것인지 여부

    Returns: 페이지 단위로, jsonline 포맷으로 파싱된 pdf 데이터

    """
    doc = pymupdf.open(fpath)
    result = []
    for index, page in enumerate(doc):
        text = page.get_text()
        data = {"page": index, "title": title, "contents": text}
        result.append(data)
    result = [item for item in result if item['contents'] != '']
    return result


def get_pdf_full_text(fpath):
    """ pdf를 각 페이지별로 읽어온 다음 모든 페이지를 합쳐서 하나의 text로 만드는 함수

    Args:
        fpath: 읽어올 PDF 파일의 Path
        normalize: 각 페이지 텍스트의 whitespace를 normalize할 것인지 여부

    Returns: 모든 페이지의 text를 합친 text

    """

    doc = pymupdf.open(fpath)
    result = []
    for index, page in enumerate(doc):
        text = page.get_text()
        result.append(text)
    result = [item for item in result if item != '']

    return "\n".join(result)