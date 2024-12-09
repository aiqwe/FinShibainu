# Fewshot 저장소
Fewshot Naming Rule은 다음과 같습니다.
`common_{task_type}_mcqa_n_options_{선택지 수}.txt`

실험 결과 회계와 금융 시장의 Fewshot을 굳이 나누지 않더라도 사용자의 Reference를 기반으로 충분히 의미 전달이 되었습니다.

따라서 회계 / 금융 시장을 나누지 않고 task_type(일반 지식, 수학 계산)과 선택지 수만으로 Fewshot을 나눠서 관리합니다.

예선에는 각 도메인별 Fewshot을 관리하였으므로 예선 때 사용한 템플릿을을 [`deprecated_fewshot.py`](deprecated_fewshot.py)를 통해 참조 자료료 제공합니다. 

특히 선택지 수는 Fewshot의 영향에 있어서 매우 큰 영향을 기칩니다. 4가지 선택지의 문제를 만들 때, Fewshot에 4가지 선택지를 만들어 제공하는 것이 그 어떤 프롬프트를 추가하는 것보다 큰 영향을 끼쳤습니다.
