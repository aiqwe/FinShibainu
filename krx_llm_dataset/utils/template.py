""" fewshot, validation set, User Prompt, System Prompt 생성을 위한 모듈"""

from typing import List, Literal, Optional
import os
from pydantic import BaseModel
from openai import OpenAI
from types import SimpleNamespace
from textwrap import dedent
from jinja2 import FileSystemLoader, Environment
from .config import PathConfig, PromptConfig

path = PathConfig()
loader = FileSystemLoader(path.PROMPT_DIR)
env = Environment(loader=loader)
prompt_config = PromptConfig()


def generate_fewshot(
        user_prompt: str,
        n_options: int = 4,
        eval_type: Literal['mcqa', 'qa'] = None,
        task_type: Literal['knowledge', 'math'] = None,
        domain_type: Literal['accounting', 'market', 'law', 'academy'] = None,
        n_datasets: int = 1,
        step: int = 2,
        with_template=False,
        title: Optional[str] = None,
        contents: Optional[str] = None
):
    """ ChatGPT에서 퓨샷을 생성하기 위한 프롬프트 생성기

    Args:
        user_prompt: 입력할 유저 프롬프트
        n_options: mcqa의 선택지 갯수
        eval_type: GPT로부터 평가받을 형태를 입력. 형태에 따라 사전 입력된 System Prompt의 jinja2 템플릿 호출
          - mcqa: MCQA Instruction 생성 작업
          - qa: QA Instruction 생성 작업
        task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
          - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
          - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는...
        step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정
        with_template: user_prompt 템플릿(title, contents 사용) 사용 여부
        title: Reference의 타이틀
        contents: Reference의 내용

    Returns: fewshot 프롬프트 텍스트 출력

    """
    role = prompt_config.ROLE[domain_type]
    task = prompt_config.TASK_TYPE[task_type]
    level = prompt_config.LEVEL[step - 1]
    system_template = env.get_template(prompt_config.PROMPTS['fewshot'][eval_type])
    system_prompt = system_template.render(n_options=n_options, role=role, task_type=task, level=level)
    if with_template:
        if (title is not None) or (contents is not None):
            raise ValueError(f"if with_template is True, title and contents required\n{title=}\n{contents=}")
        user_template = env.get_template(prompt['fewshot'][eval_type][task_type][step - 1])
        user_prompt = user_template.render(title=title, contents=contents)
    else:
        user_prompt = f"사용자의 설명을 기반으로 {n_datasets}개의 데이터를 생성하세요.\n### 설명:\n" + user_prompt
    return f"{system_prompt}\n\n{user_prompt}"

def generate_validation(
        user_prompt: str,
        n_options: int = 4,
        eval_type: Literal['mcqa', 'qa'] = None,
        task_type: Literal['knowledge', 'math'] = None,
        domain_type: Literal['accounting', 'market', 'law', 'academy'] = None,
        n_datasets: int = 5
):
    """ ChatGPT에서 Validation Set을 생성하기 위한 프롬프트 생성기

    Args:
        user_prompt: 유저 프롬프트
        n_options: mcqa시 선택지 갯수
        eval_type: mcqa, qa, halluciation, preference, value 평가 등 다양한 타입
          - mcqa: MCQA Instruction 생성 작업
          - qa: QA Instruction 생성 작업
        task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
          - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
          - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는...
        with_template: user_prompt의 템플릿(title, contents 사용) 사용 여부
        title: Reference의 타이틀
        contents: Reference의 내용

    Returns: Validation Prompt Text 출력

    """
    role = prompt_config.ROLE[domain_type]
    task = prompt_config.TASK_TYPE[task_type]
    system_template = env.get_template(prompt_config.PROMPTS['validation'][eval_type])
    system_prompt = system_template.render(n_options=n_options, role=role, task_type=task)
    user_prompt = f"다음 사용자의 설명을 기반으로 {n_datasets}개의 데이터를 생성하세요.\n ### 설명\n" + user_prompt
    return f"{system_prompt}\n\n{user_prompt}"

# Orca
class OrcaTemplates:
    """ Orca 템플릿 """
    id1: str = ""
    id2: str = "당신은 AI 어시스턴트입니다. 사용자가 답변을 이해하기 위해 추가 검색을 하지 않도록 자세한 답변을 제공하세요."
    id3: str = "당신은 AI 어시스턴트입니다. 주어진 작업에 대해 상세하고 긴 답변을 생성해야 합니다."
    id4: str = "당신은 설명을 제공하여 항상 도움이 되는 어시스턴트입니다. 5살 어린이에게 답변하는 것처럼 생각해 보세요."
    id5: str = "당신은 지시 사항을 매우 잘 따르는 AI 어시스턴트입니다. 가능한 한 많이 도와주세요."
    id6: str = "당신은 정보를 찾는 데 도움을 주는 AI 어시스턴트입니다. 사용자가 답변을 이해하기 위해 추가 검색을 하지 않도록 자세한 답변을 제공하세요."
    id7: str = "당신은 AI 어시스턴트입니다. 사용자가 지시 사항을 제공하면 충실하게 수행하는 것이 목표입니다. 작업을 수행하는 동안 단계별로 생각하고, 그 과정을 증명하세요."
    id8: str = "사용자가 요청한 작업에 대한 답변을 설명합니다. 객관식 질문에 답할 때에 먼저 정답을 출력하세요. 그런 다음 다른 답변이 왜 틀렸는지 설명하세요. 다섯 살 어린이에게 답변하는 것처럼 생각해 보세요."
    id9: str = "어떻게 정의를 내렸는지, 어떻게 답변을 도출했는지 설명하세요."
    id10: str = "당신은 AI 어시스턴트입니다. 사용자가 요청한 작업과 답변을 설명해야 합니다. 객관식 질문에 답할 때 먼저 정답을 출력하세요. 그런 다음 다른 답변이 왜 틀렸는지 설명하세요. 답변하기 위해 추가적인 지식을 사용할 필요가 있을 수 있습니다."
    id11: str = "당신은 정보를 찾는 데 도움을 주는 AI 어시스턴트입니다. 사용자가 질문을 하면 충실하게 답변하는 것이 목표입니다. 답변하는 동안 단계별로 생각하고 그 답변을 정당화하세요."
    id12: str = "사용자가 지시와 함께 작업을 지시할 것입니다. 주어진 지시를 가능한 한 충실하게 따르는 것이 작업입니다. 답변하는 동안 단계별로 생각하고 답변을 증명하세요."
    id13: str = "당신은 교사입니다. 작업이 주어지면 해당 작업이 요구하는 사항, 제공되는 지침, 그리고 그 지침을 활용해 답변을 찾는 방법을 간단한 단계로 설명하세요."
    id14: str = "당신은 모든 언어를 알고 번역할 수 있는 AI 어시스턴트입니다. 작업이 주어지면 해당 작업이 요구하는 사항과 지침을 간단하게 설명하세요. 작업을 해결하고 그 해결 과정과 지침을 사용해 설명하세요."
    id15: str = "작업의 정의와 예제 입력이 주어졌을 때, 정의를 작은 부분으로 나누세요. 각 부분은 부분적인 지시 사항을 포함하게 됩니다. 지시 사항의 의미를 해당 기준에 맞는 예제로 보여주세요. 다음 형식을 사용하세요:\nPart #: 정의의 주요 부분.\nUsage: 해당 정의 기준을 충족하는 예제 응답. 왜 그 기준에 부합한다고 생각하는지 설명하세요."
    id16: str = "당신은 정보를 찾는 데 도움을 주는 AI 어시스턴트입니다."
    total: str = "\n".join([id1, id2, id3, id4, id5, id6, id7, id8, id9, id10, id11, id12, id13, id14, id15, id16])

class PromptTemplates:
    prompt = PromptConfig()
    path = PathConfig()

    def system_prompt(
            self,
            n_options: int = 4,
            example=None,
            eval_type: Literal['mcqa', 'qa', 'hallucination', 'preference', 'value', 'classification'] = None,
            task_type: Literal['common', 'knowledge', 'math'] = None,
            domain_type: Literal['common', 'accounting', 'market', 'law', 'quant'] = None,
            step: int = None,
            **kwargs
    ):
        """ System Prompt를 생성

        Args:
            n_options: MCQA에서 선택지의 갯수
            example: Fewshot 텍스트
            eval_type: mcqa, qa, halluciation, preference, value 평가 등 다양한 타입
              - mcqa: MCQA Instruction 생성 작업
              - qa: QA Instruction 생성 작업
              - hallucination: Hallucination 여부 평가
              - preference: 첫번째, 두번째 Instruction중 Preference 평가
              - value: Fineweb-edu 기반 교육적 가치 측정
            task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
              - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
              - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
              - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는...
            domain_type: 생성할 프롬프트의 도메인 형태. 도메인에 따라 페르소나(role) 프롬프트가 추가.
              - accounting: 회계. 예시) 당신은 전문 회계사 역할을 맡아...
              - market: 금융시장. 예시) 당신은 금융 및 주식 분석가 역할을 맡아...
              - law: 법률. 예시) 당신은 금융 법률 전문가 역할을 맡아...
              - quant: 금융학, 금융공학. 예시) 당신은 금융학자 역할을 맡아...
              - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 당신은 유능한 금융 전문가입니다...포괄적인 도움을 제공합니다.
            step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정

        Returns: System Prompt 텍스트 출력
        """
        if eval_type in ("mcqa", "qa"):
            role = prompt_config.ROLE[domain_type]
            task = prompt_config.TASK_TYPE[task_type]
            level = prompt_config.LEVEL[step - 1]
            system_template = env.get_template(prompt_config.PROMPTS['system_prompt'][eval_type][step - 1])
            system_prompt = system_template.render(
                example=example,
                n_options=n_options,
                role=role,
                task_type=task,
                level=level)

        if eval_type in ("hallucination", "preference", "value", "classification"):
            # 이 eval_type 들은 별도의 Parameter가 들어가지 않음
            system_template = env.get_template(prompt_config.PROMPTS['system_prompt'][eval_type])
            system_prompt = system_template.render()

        return system_prompt

    def user_prompt(
            self,
            data: dict,
            n_datasets: int = 5,
            eval_type: Literal['mcqa', 'qa', 'hallucination', 'preference'] = None,
            step: int = None,
    ):
        """ User Prompt 생성

        Args:
            data:
            n_datasets:
            eval_type: mcqa, qa, halluciation, preference, value 평가 등 다양한 타입
              - mcqa: MCQA Instruction 생성 작업
              - qa: QA Instruction 생성 작업
              - hallucination: Hallucination 여부 평가
              - preference: 첫번째, 두번째 Instruction중 Preference 평가
              - value: Fineweb-edu 기반 교육적 가치 측정
            step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정
        """
        if eval_type == 'mcqa':
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type][step - 1])
            if step == 1:
                user_prompt = user_template.render(n_datasets=n_datasets, title=data['title'], contents=data['contents'])

            if step >= 2:
                # Step 2 이상에서는 이전 Generation 결과를 Example로 입력하여 중복되지 않고 난이도를 높임
                if data['prev_example'] is None:
                    raise ValueError("if step is more than 2, the previous example is required.")
                user_prompt = user_template.render(
                    n_datasets=n_datasets,
                    title=data.get('title', None),
                    contents=data.get('contents', None),
                    prev_example=data.get('prev_example', None)
                )
        if eval_type == 'qa':
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type][step - 1])
            if step == 1:
                user_prompt = user_template.render(n_datasets=n_datasets, title=data['title'], contents=data['contents'])
            if step == 2:
                user_prompt = user_template.render(question=data['question'])

        if eval_type == "hallucination":
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type])
            if data['type'] == 'mcqa':
                # user_prompt = user_template.render(
                #     question=data['question'],
                #     options="\n".join(data['options']),
                #     reasoning_process=data['reasoning_process'],
                #     answer=data['answer']
                # )
                user_prompt = user_template.render(
                    question=data['question'],
                    options="\n".join(data['options']),
                )
            else:
                user_prompt = user_template.render(**data)

        if eval_type == "preference":
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type])
            user_prompt = user_template.render(question=data['question'], answer1=data['old_answer'], answer2=data['answer'])

        if eval_type == "value":
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type])
            if data['type'] == 'mcqa':
                user_prompt = user_template.render(
                    question=data['question'],
                    options="\n".join(data['options']),
                    reasoning_process=data['reasoning_process'],
                    answer=data['answer']
                )
            if data['type'] == 'qa':
                user_prompt = user_template.render(question=data['question'], answer=data['answer'])
            if data['type'] == 'references':
                user_prompt = user_template.render(question=data['contents'])

        if eval_type == "classification":
            user_template = env.get_template(prompt_config.PROMPTS['user_prompt'][eval_type])
            if data['type'] == 'references':
                user_prompt = user_template.render(question=data['title'], contents=data['contents'])

        return user_prompt