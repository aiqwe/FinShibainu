# Prompt Template 저장소
각 템플릿은 [`jinja2`](https://jinja.palletsprojects.com/en/stable/) 기반으로 작성되었으며 사용처는 다음과 같습니다.

- `fewshot_generator_mcqa.jinja2`: `utils.template.generate_fewshot`에서 MCQA의 System Template을 작성해주는 템플릿
- `fewshot_generator_qa.jinja2`: `utils.template.generate_fewshot`에서 QA의 System Template을 작성해주는 템플릿
- `system_classification.jinja2`: Classification 작업시 System Prompt를 작성해주는 템플릿 
- `system_hallucination.jinja2`: Hallucination 작업시 System Prompt를 작성해주는 템플릿
- `system_mcqa_default.jinja2`: MCQA 작업시 System Prompt의 Skeleton 템플릿
- `system_mcqa_step1.jinja2`: MCQA 작업시 Step1의 System Prompt을 작성해주는 템플릿(default 템플릿이 Override 됩니다)
- `system_mcqa_step2.jinja2`: MCQA 작업시 Step2이상에서 System Prompt을 작성해주는 템플릿(default 템플릿이 Override 됩니다)
- `system_preference.jinja2`: Preference 작업시 System Prompt을 작성해주는 템플릿 
- `system_qa.jinja2`: QA 작업시 첫번째 질문 답변을 생성하는 단계에서 System Prompt를 작성해주는 템플릿
- `system_qa_answer.jinja2`: QA 작업시 두번째 답변만을 생성하는 단계에서 System Prompt를 작성해주는 템플릿
- `system_value.jinja2`: Educational Value 평가를 위한 System Prompt를 작성해주는 템플릿
- `user_classification.jinja2`: Classification 작업시 User Prompt를 작성해주는 템플릿
- `user_hallucination.jinja2`: Hallucination 작업시 User Prompt를 작성해주는 템플릿
- `user_mcqa_step1.jinja2`: MCQA 작업시 Step1의 User Prompt을 작성해주는 템플릿(default 템플릿이 Override 됩니다)
- `user_mcqa_step2.jinja2`: MCQA 작업시 Step2의 User Prompt을 작성해주는 템플릿(default 템플릿이 Override 됩니다)
- `user_preference.jinja2`: Preference 작업시 User Prompt을 작성해주는 템플릿
- `user_qa.jinja2`: QA 작업시 첫번째 질문 답변을 생성하는 단계에서 User Prompt를 작성해주는 템플릿
- `user_qa_answer.jinja2`: QA 작업시 두번째 답변만을 생성하는 단계에서 User Prompt를 작성해주는 템플릿
- `user_value.jinja2`: Educational Value 평가를 위한 User Prompt를 작성해주는 템플릿