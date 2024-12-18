# KRX LLM Competition - shibainu24 Model(FinShibainu) / Dataset Repository
KRX 금융 언어 모델 경진대회에서 우수상을 입상한 `shibainu24` 모델의 학습 코드 및 데이터 생성 레포지토리입니다.  
한국의 금융 언어 모델 도메인에 특화시키기 위해 한국의 금융, 회계 제도와 법률을 기반으로 학습되었습니다.

+ Dataset 생성 소스코드 : [krx_llm_dataset](./krx_llm_dataset)
+ Training 소스코드 : [krx_llm_train](./krx_llm_train)
+ Model(Huggingface) : [https://huggingface.co/aiqwe/FinShibainu](https://huggingface.co/aiqwe/FinShibainu)
+ Dataset(Huggingface) : [https://huggingface.co/datasets/aiqwe/FinShibainu](https://huggingface.co/datasets/aiqwe/FinShibainu)

# 라이브러리
사용 전에 다음과 같이 라이브러리를 설치해주세요.
+ 데이터셋 생성 관련 라이브러리
```shell
pip install -r requirements-dataset.txt
```
+ 학습 코드 관련 라이브러리
```shell
pip install -r requirements-train.txt
```

# 환경변수
데이터셋 생성시 특정 기능을 사용하려면 환경변수가 입력되어야 합니다.

+ `NAVER_API_ID` : 네이버 검색 API 사용시
+ `NAVER_API_SECRET` : 네이버 검색 API 사용시
+ `PUBLIC_DATA_API_KEY` : 공공데이터 API 사용시
+ `OPENAI_API_KEY` : OpenAI API 사용시

각 API 사용을 위한 참조 사이트는 아래와 같습니다.
+ Naver: [Naver Developer](https://developers.naver.com/docs/serviceapi/search/blog/blog.md)
+ 공공데이터: [공공데이터 포털](https://www.data.go.kr/)
+ OpenAI: [OpenAI Platform](https://platform.openai.com/)

# Citation
```bibtex
@misc{jaylee2024finshibainu,
  author = {Jay Lee},
  title = {FinShibainu: Korean specified finance model},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/aiqwe/FinShibainu}
}
```
