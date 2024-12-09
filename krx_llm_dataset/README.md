# krx-llm-dataset
shibainu24모델의 Dataset을 생성하는 코드 레포입니다.

## Code Tree
기능은 코드 내의 Docstring을 참조해주세요.

### Code
`utils` 내 코드 트리
+ `api`: GPT Completion 등 API 요청용 코드
+ `config`: Path, Prompt Config등 설정 코드
+ `datamodel`: GPT Completion 생성시 Output을 강제하기 위한 `Pydantic` 모델
+ `dataset_manager`: 데이터셋 메타 정보를 관리하기 위한 도움용 함수
+ `graph`: 데이터셋 자동화를 수행하는 [Langgraph](https://www.langchain.com/langgraph) 기반의 Graph Pipeline 코드
+ `io`: json, jsonl, pdf 파일 읽기, 저장용 코드
+ `processing` : Graph Pipeline의 에러처리, 시각화 등 각종 데이터 가공용 코드
+ `template`: Prompt 생성에 도움을 주는 프롬프트 템플릿 생성 코드
+ `utils` : 환경변수 로드, 파일 탐색, 파일 크기 출력 등 일반적인 유틸리티 코드

### tools
`utils` 내 Fewshot 결과물 및 `jinja2` 프롬프트 템플릿
+ `fewshot`: `o1-preview`를 통해 생성한 fewshot 모음
+  `prompt_templates` : `jinja2`을 기반으로 한 각 작업별 System / User Prompt 템플릿  

## Huggingface
+ Dataset: [https://huggingface.co/datasets/aiqwe/krx-llm-competition](https://huggingface.co/datasets/aiqwe/krx-llm-competition)
+ Model: [https://huggingface.co/aiqwe/krx-llm-competition](https://huggingface.co/aiqwe/krx-llm-competition)

## 예제 파일
+ pdf 기반 데이터셋 생성 예제: [pdf_examples.ipynb](./pdf_examples.ipynb)
+ 용어사전 기반 데이터셋 생성 예제: [ref_examples.ipynb](./ref_examples.ipynb)