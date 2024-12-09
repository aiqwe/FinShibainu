""" GPT API로 Text Generation을 할 때 Output Format을 강제하기 위한 Pydantic BaseModel """

from pydantic import BaseModel
from typing import List, Dict, Union, Optional

class QAQuestionModel(BaseModel):
    question: str

class MCQAQuestionModel(BaseModel):
    question: str
    options: List[str]

class AnswerModel(BaseModel):
    reasoning_process: str
    answer: str

class QAAnswerModel(BaseModel):
    answer: str

class KnowledgeFormat(BaseModel):
    knowledge: str

class MCQAOutputFormat(BaseModel):
    questions: List[MCQAQuestionModel]
    answers: List[AnswerModel]

class QAOutputFormat(BaseModel):
    questions: List[QAQuestionModel]
    answers: List[QAAnswerModel]

class EvaluationByLLMOutputFormat(BaseModel):
    ratings: List[int]

class HallucinationOutputFormat(BaseModel):
    hallucination: str
    hallucination_desc: str

class PreferenceOutputFormat(BaseModel):
    choice: str
    preference_desc: str

class ValueOutputModel(BaseModel):
    value: int

class ClassificationOutputModel(BaseModel):
    classification: str

class EnglishMCQAFromMCQAOutputFormat(BaseModel):
    question_translated: str
    options_translated: List[str]
    addtional_options: List[str]
    reasoning_process: str

class KoreanMCQAFromMCQAOutputFormat(BaseModel):
    addtional_options: List[str]
    reasoning_process: str

class MCQAFromMCQAOutputFormat(BaseModel):
    reasoning_process: str