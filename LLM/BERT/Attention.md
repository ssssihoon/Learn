# Attention mechanism

## Attention Mechanism이란?

- 관련 정보에 집중해 모델의 성능을 향상시키는데 사용되는 기술. 그래서 여러 부분에서 선택적으로 attention을 기울이고 여러 요소에 다양한 중요도 또는 가중치를 할당할 수 있다.
- 디코더에서 출력 단어를 예측하는 매시점마다 인코더에서의 전체 입력 문장을 다시 한 번 참고한다.

## 등장 배경

- RNN, LSTM, Seq2SeqModel 등의 모델 들은 고정길이의 문맥 벡터를 사용한 나머지 긴 문장을 기억할 수 없다는 단점이 있다. 이를 해결하기 위해 등장
    - 하나의 고정된 크기의 벡터에 모든 정보를 압축 →  정보 소실이 발생한다.
    - Gradient Vanishing 문제 존재

## Self Attention

- 자기 자신을 attention하겠다는 것으로 동일한 문장 내에 단어와 단어 사이의 관계성을 파악하겠다는 의미

[https://velog.io/@seven7724/Attention-Mechanism](https://velog.io/@seven7724/Attention-Mechanism)

## 작동원리

[https://h2o.ai/wiki/attention-mechanism/](https://h2o.ai/wiki/attention-mechanism/)

- 입력 데이터의 다양한 요소 또는 특징에 대한 어텐션 가중치를 생성해 작동
- 그 가중치는 각 요소가 모형의 출력에 기여하는 중요도를 결정
- 요소와 쿼리 또는 컨텍스트 벡터 간의 관련성 또는 유사성을 기반으로 계산된다.

### 구성요소

- Query : 모델의 현재 context 또는 focus를 나타낸다.
- Key : 입력 데이터의 요소 또는 특징을 나타낸다.
- Value : 요소 또는 형상과 관련된 값을 나타낸다.

## Attention Mechanism의 중요성

- 중요한 패턴이나 종속성을 포착 가능
- 가변 길이 입력의 효과적인 처리가 가능하다.
- 해석 가능성, 설명 가능성