""" Dataset들의 메타정보를 관리하도록 보조하는 모듈 """

import pandas as pd
import os
from datetime import datetime
from .io import read_jsonl, write_jsonl
from .processing import show_spec, show_duplicates

class DatasetRegister:
    """ Dataset configuration을 등록하고 관리하기 위한 클래스 """
    def __init__(self, fname: str = "dataset_config.jsonl"):
        """
        Args:
            fname: Dataset Configuration 파일명
        """
        self.fname = fname
        self.ds = list() if not os.path.exists(self.fname) else list(read_jsonl(fname))

    def read_config(self):
        """ Configuration 파일을 읽어오기 """
        return pd.DataFrame(read_jsonl(self.fname))

    def exclude_config(self, en_name: str = None, save=False):
        """ 영문 데이터셋 이름(en_name)의 Configuration을 제외하고 저장

        Args:
            en_name: 영문 데이터셋명
            save: 저장 여부

        Returns: 제외 후 데이터셋을 Pandas DataFrame으로 반환

        """
        df = self.read_config()
        lst = df.to_dict("records")
        if en_name is not None:
            result = [item for item in lst if item['en_name'] != en_name]
        if save:
            fname = self.fname if self.fname else "dataset_config.jsonl"
            write_jsonl(result, fname, ensure_ascii=False)
        return pd.DataFrame(result)

    def add_config(self, ko_name: str, en_name: str, url: str, license: str, save=False):
        """ Datset Configuration을 추가하고 저장

        Args:
            ko_name: 데이터소스 한글명
            en_name: 데이터소스 영문명
            value: Educational value값
            url: 소스 URL
            license: 라이센스 구분
            save: config jsonl 파일에 저장 여부
        """
        additonal_config = {"ko_name": ko_name, "en_name": en_name, "url": url, "license": license}
        ex = self.read_config()
        if len(ex) != 0:
            assert sorted(list(ex.columns)) == sorted(additonal_config), "Not Match Columns"
        result = pd.concat([ex, pd.DataFrame([additonal_config])])
        if save:
            fname = self.fname if self.fname else "dataset_config.jsonl"
            write_jsonl(result.to_dict("records"), fname, ensure_ascii=False)
        return pd.concat([ex, pd.DataFrame([additonal_config])])

class ValidationSetRegister:
    """ Valudation Set을 작성하고 관리하는데 보조하는 도구 클래스 """
    def __init__(self, eval_type: str, fname: str = None):
        """
        Args:
            eval_type: 평가타입. mcqa, qa
            fname: 저장하려는 Validation Set의 파일명
        """
        if fname is None:
            self.fname = f'krx-validation-{eval_type}.jsonl'
        else:
            self.fname = fname
        self.eval_type = eval_type

    def read_validation(self):
        """ Validation Set 파일을 읽어오기 """
        if not os.path.exists(self.fname):
            return pd.DataFrame(columns = ["reference_index", "question", "options", "reasoning_process", "answer"])
        return pd.DataFrame(read_jsonl(self.fname))

    def add_validation(self, valid_type_index: str, question: str, options: list, reasoning_process: str, answer: str, oai_model="o1-preview", save=False):
        """ Validation Set을 추가하고 저장

        Args:
            valid_type_index: 평가 항목.
              + stmt_analysis: 재무제표 분석
              + acc_principle: 회계원칙 적용
              + fin_ratio: 재무비율 계산
              + fin_concept: 금융 개념
              + mkt_analysis: 시장 동향
              + invst_strategy: 투자 전략
            question: Validation set의 질문
            options: Validation set의 선택지
            reasoning_process: Validation set의 문제풀이 과정
            answer: Validation set의 정답
            save: 저장여부
        """
        if valid_type_index not in ("stmt_analysis", "acc_principle", "fin_ratio", "fin_concept", "mkt_analysis", "invst_strategy"):
            raise ValueError(f'valid_type_index should be one of ["stmt_analysis", "acc_principle", "fin_ratio", "fin_concept", "mkt_analysis", "invst_strategy"]')

        additonal_data = {
            "seq": None,
            "valid_type_index": valid_type_index,
            "question": question,
            "options": options,
            "reasoning_process": reasoning_process,
            "answer": answer,
            "n_options": str(int(len(options))),
            "type": self.eval_type,
            "oai_model": oai_model,
            "update_at": datetime.now().strftime("%Y-%m-%d")
        }
        ex = self.read_validation()
        if len(ex) != 0:
            assert sorted(list(ex.columns)) == sorted(additonal_data), "Not Match Columns"
        result = pd.concat([ex, pd.DataFrame([additonal_data])])
        result['seq'] = [i for i in range(len(result))]

        if save:
            write_jsonl(result.to_dict("records"), self.fname)
        return result

    def exclude_config(self, index: str, save=False):
        """ 특정 en_name의 Configuration을 제외하고 저장 """
        df = self.read_validation()
        lst = df.to_dict("records")
        result = [item for item in lst if item['en_name'] != index]
        if save:
            write_jsonl(result, self.fname)
        return pd.DataFrame(result)

    def show_spec(self, column = "reference_index"):
        """ Validation Set의 특정 컬럼의 집계 결과를 시각화하여 보여주는 함수

        Args:
            column: 집계하려는 컬럼명

        Returns: None

        """
        show_spec(self.read_validation(), column)





