""" Path, 추가 Prompt Configuration"""

from pathlib import Path
from textwrap import dedent
from typing import List
from pydantic import BaseModel
import os
from types import SimpleNamespace

AGENT_LIST: str = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
]


class BaseConfigMixin:
    @property
    def get_attrs_list(self):
        return [attrs for attrs in self.__dict__ if not attrs.startswith("_")]
    @property
    def get_attrs_dict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
    def get(self, key):
        return self.get_attrs_dict[key]

class PathConfig(BaseConfigMixin):
    """ 모듈내 각종 Path를 사전 정의하여 손쉽게 불러오게 도와주는 클래스 """
    def __init__(self):
        self.ROOT: str = str(Path(__file__).parent.parent) # krx-llm-competition
        self.CONFIG: str = str(Path(self.ROOT).joinpath("config")) # krx-llm-competition/config
        self.DATA: str = str(Path(self.ROOT).joinpath("data"))  # krx-llm-competition/data
        self.OUTPUTS: str = str(Path(self.DATA).joinpath("outputs"))  # krx-llm-competition/data/outputs
        self.REFERENCES: str = str(Path(self.DATA).joinpath("references"))  # krx-llm-competition/data/outputs
        self.LOG: str = str(Path(self.ROOT).joinpath("logs"))  # krx-llm-competition/data/outputs
        self.MCQA: str = str(Path(self.OUTPUTS).joinpath("mcqa"))  # krx-llm-competition/data/outputs/mcqa
        self.QA: str = str(Path(self.OUTPUTS).joinpath("qa"))  # krx-llm-competition/data/outputs/mcqa
        self.PROMPT_DIR: str = str(Path(self.ROOT).joinpath("utils/prompt_templates"))
        self.FEWSHOT_DIR: str = str(Path(self.ROOT).joinpath("utils/fewshot"))
        super().__init__()

class PromptConfig(BaseConfigMixin):
    """ 각 task_type, domain_type, step 별로 추가 프롬프트를 정의한 Configuration """

    def __init__(self):
        self.ROLE: dict = {
            "accounting": "당신은 전문 회계사 역할을 맡아 재무제표 분석, 회계 원칙 적용, 재무 비율 계산 등 다양한 재무 회계 작업에 도움이되어야합니다.",
            "market": "당신은 금융 및 주식 분석가 역할을 맡아 금융 개념, 시장 동향, 투자 전략, 실제 금융 시나리오를 바탕으로 한 문제 해결에 도움이되어야합니다.",
            "law": "당신은 금융 법률 전문가 역할을 맡아 금융 및 자본시장 법률 문제 해결에 도움이되어야합니다.",
            "quant": "당신은 금융학자 역할을 맡아 금융 통계, 퀀트 모델, 금융학 관련 문제 해결에 도움이되어야합니다.",
            "common": "당신은 유능한 금융 전문가입니다. 필요한 경우 금융 지식 외에도 관련된 경제학적, 법학적, 금융 공학적 또는 실무적인 배경 지식을 통합하여 포괄적인 도움을 제공합니다."
        }

        self.TASK_TYPE: dict = {
            "knowledge": "회계, 금융, 산업 지식, 금융 법률을 이해하고 활용할 수 있도록 이를 활용한 실제 사례를 포함하여, 상세한 답변과 추론 과정을 작성하세요.",
            "math": "현실적인 상황(예: 금융 계산, 투자 분석, 통계적 예측 등)을 기반으로 한 수학적으로 계산하는 문제를 작성하고, 이를 단계별로 해결하세요. latex을 사용하여 작성하세요.",
            "common": dedent("""\
            금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는 다양한 데이터셋을 작성하세요.
                + 회계, 금융, 산업 지식, 금융 법률을 이해하고 활용할 수 있도록 이를 활용한 실제 사례를 포함하여, 상세한 답변과 추론 과정을 작성하세요.
                + 현실적인 상황(예: 금융 계산, 투자 분석, 통계적 예측 등)을 기반으로 한 수학적 문제를 제시하고, 이를 단계별로 해결하세요. latex을 사용하여 작성하세요.""")
        }

        self.LEVEL: list = [
            "CFA, FRM, CPA, LEET와 같은 전문 자격시험의 고난도 문제와 해설을 포함한 데이터셋을 작성하세요.",
            "사용자의 설명을 심화하는 CFA, FRM, CPA, LEET와 같은 전문 자격시험 이상의 최고 난이도 문제와 해설을 포함한 데이터셋을 작성하세요."
        ]
        self.PROMPTS = {
            "fewshot": {
                "mcqa": "fewshot_generator_mcqa.jinja2",
                "qa": "fewshot_generator_qa.jinja2",
            },
            "system_prompt": {
                "mcqa": ["system_mcqa_step1.jinja2", "system_mcqa_step2.jinja2"],
                "qa": ["system_qa.jinja2", "system_qa_answer.jinja2"],
                "hallucination": "system_hallucination.jinja2",
                "preference": "system_preference.jinja2",
                "value": "system_value.jinja2",
                "classification": "system_classification.jinja2",

            },
            # mcqa / qa는 User Prompt 공유함
            "user_prompt": {
                "mcqa": ['user_mcqa_step1.jinja2', 'user_mcqa_step2.jinja2'],
                "qa": ['user_qa.jinja2', 'user_qa_answer.jinja2'],
                "hallucination": "user_hallucination.jinja2",
                "preference": "user_preference.jinja2",
                "value": "user_value.jinja2",
                "classification": "user_classification.jinja2",
            }
        }

