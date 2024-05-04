# RNN

# 신경망의 기초 이해

## 인공 신경망의 기본 개념

ANN, Artificial Neural Network : 행렬 수학을 모델로 삼아 인간의 뉴런 구조를 본떠 만든 기계학습 모델.

인공지능을 구현하기 위한 기술 중 한 형태이다.

### 인공신경망의 구조

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled.png)

- 입력 계층 : 값들을 전달하기만 하는 특징을 갖는다.
    - 데이터셋으로부터 입력을 받음
    - 입력 변수의 수 = 노드의 수
    - 계층의 크기 = 노드의 개수 = 입력 변수의 수 =  입력벡터의 길이
- 은닉 계층 : 계산이 일어나는 층이다.
    - 입력층과 마지막 출력층 사이에 있는 층들을 의미
    - 계산의 결과를 사용자가 볼 수 없다.
- 출력 계층 : 신경망의 마지막층.
    - 출력층의 활성함수가 존재 → 이를 이용해 문제를 해결

---

노드 : 위의 동그라미

학습 매개변수(weight;중요도or가중치, bias;민감도or편향) : 위의 선

방향 (순방향, 역방향) : 화살표의 방향. 선의 끝부분에 방향이 표기되어 있음.

위의 층 수는 2층이다. → 은닉 계층 개수 + 출력 계층 개수

위의 노드 개수는 9개이다. → 입력층 3, 은닉층 4, 출력층 2

이진분류의 경우 입력층 노드가 2개라면 2개의 입력을 받아 1개의 출력 값을 반환해 이진분류를 해결하는 모델이 될 것임.

바이어스를 포함한 가중치는 몇 개인가?

→ 20개이다. because (입력층 노드 수 * 은닉층1) + (은닉층1 * 은닉층2) + (은닉층2 * 출력층)

→→ (3*4) + (4*2) = 20

## 뉴런과 활성화 함수

### 뉴런

- 인공지능과 뉴런의 관계
    - 뉴런은 시냅스를 거쳐서 수상돌기로 받아들인 외부의 전달물질을 세포체에 저장하다가 자신의 용량을 넘어서면 축삭돌기를 통해 외부로 전달물질을 내보낸다.
    - 인공신경망에서 뉴런 모델은 여러개의 뉴런으로부터 입력값을 받아서 일정 수준이 넘어서면 활성화되어 출력값을 내보낸다.

### 활성화 함수

- 인공 신경망에서 입력신호의 가중치 합을 출력 신호로 변환하는 함수
- Sigmoid 시그모이드 :
    
    ![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%201.png)
    
    - 0과 1사이의 값을 출력, 로지스틱회귀, ANN 에서 주로 사용
    - 입력값이 어떤 범위에 들어갈 때 출력 값이 0과 1사이에서 급격하게 변하는 특성을 갖고 있다. 그렇기에 이진 분류에서 출력 레이어의 활성화 함수로 사용됨

- tanh 하이퍼볼릭 탄젠트  :
    
    ![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%202.png)
    
    - 실수 값을 (-1, 1) 범위로 압축하는 활성화 함수. 원점을 중심으로 값이 양의 무한대로 갈수록 1에 수렴, 음의 무한대로 갈수록 -1에 수렴.
    - 함수의 출력값이 평균을 중심으로 분포하는 결과를 가져옴
    - 주로 은닉층에서 사용된다.
- Relu 렐루 :
    
    ![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%203.png)
    
    - (0, x)로 입력값이 0보다 크면 그 값을 출력, 0이하면 0을 출력.
    - 계산 효율성과 비선형성 특성으로 더 복잡한 모델링을 할 수 있음
    - 주로 은닉층에서 사용

## 가중치와 편향

*z*=*w*1×*x*1+*w*2×*x*2+...+*wn*×*xn +b*

z : 가중합

wn : 가중치

xn : 뉴런의 입력

b : 편향

1. 가중합 계산 : z = w*x+b
2. 활성화 함수 적용 : z값을 활성화 함수에 전달하여 뉴런을 출력을 생성
3. 은닉층으로 전달

### 가중치

각 입력 신호가 결과 출력에 미치는 중요도를 조절하는 매개변수

- 뉴런의 입력값에 곱해진다.
- 그 값을 활성화 함수로 들어가게 된다.

### 편향

뉴런의 활성화 조건을 결정하는 매개변수

- 하나의 뉴런으로 입력된 모든 값을 다 더한 다음 이 값에 더해주는 상수이다.
- 뉴런에서 활성화 함수를 거쳐 최종적으로 출력되는 값을 조절하는 역할

## 피드포워드 신경망의 구조와 작동 원리

FNN, Feedforward Neural Network 순방향 신경망

- 노드 간의 연결이 순환을 형성하지 않는 인공 신경망임.
- 입력값이 출력까지 **한 방향**으로 전달되는 구조.
- 이러한 특징으로 시계열 데이터와 같은 연속적인 정보를 처리하는 데 한계가 있음.
- 지도 학습에서 주로 사용, **분류, 회귀** 문제를 해결하는데 주로 사용.

[https://wikidocs.net/176148](https://wikidocs.net/176148)

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%204.png)

[https://www.researchgate.net/figure/A-simple-three-layered-feedforward-neural-network-FNN-comprised-of-a-input-layer-a_fig3_285164623](https://www.researchgate.net/figure/A-simple-three-layered-feedforward-neural-network-FNN-comprised-of-a-input-layer-a_fig3_285164623)

- 단방향(순방향)으로 구성되어 있다.
- 데이터가 신경망의 입력층에서 출력층까지 단방향으로 이동하므로 순환 또는 피드백이 없다.

# 시퀀스 데이터 이해

## 시퀀스 데이터의 특징과 종류

- 시퀀스 데이터 : 순서대로 정렬된 데이터의 연속

### 특징

- 특정 순서에 따라 배열된 항목들로 구성
- 순서가 중요함.

### 종류

- `list`, `tuple`, `range` : 인덱스를 활용해 위치에 접근이 가능하다.

## 시계열 데이터와 자연어 처리의 예

### 시계열 데이터의 예시

- 일일 주가, 분 단위 센서 데이터, 월간 판매량 등

### 자연어 처리의 예시

- 텍스트 분류, 기계 번역, 감정 분석, 챗봇 등

## 순차 데이터의 특성과 응용 분야

Sequential data 순차 데이터 : 

텍스트나 시계열 데이터와 같이 시간 및 순서에 의미가 있는 데이터

순차적 구성 요소가 복잡한 의미와 구문 규칙에 따라 상호 연관되는, 단어, 문장 또는 시계열 데이터 등의 데이터

### 순차 데이터 특성

- 데이터의 값이 전 후 값 사이에 상관관계를 가질 수 있다.
- 이전 관측에 영향을 받는다. → FNN과는 거리가 멀다(FNN은 직접적인 값을 출력으로 이용하기 때문에 이전 값을 버림)
- 대부분이 가변적인 길이로 구성
- 패턴이 있을 수 있다.
- 시간적인 순서를 가진다.

### 순차 데이터 응용 분야

- 의료 : 건강데이터로 질병 진행을 예측
- 금융 : 다음 주가가 어떻게 될 것인가
- 이상 탐지 : 정상 동작에서 벗어나는 비정상적인 패턴 감지

# RNN의 개념 이해

## 순환 신경망의 개념과 구조

### 순환 신경망이란?

- 순차 데이터 입력을 처리하고 특정 순차 데이터 출력으로 변환하도록 훈련된 딥러닝 모델.

### 구조

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%205.png)

## RNN의 순환 구조와 정보 전달 방식

### 순환 구조

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%206.png)

[https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)

### 정보 전달 방식

- 초기조건(x0 = 0 or 1 or random)이 있어야 한다.
- 이전의 값 h(t-1)과 현재의 값 h(t)를 이용해 y값을 도출한다.
- 활성화함수가 적용된 이전의 값을 이용해 y값을 도출.

## 시간에 따른 정보 처리와 활용 방법

- One-to-One : Input-Onem Out-One : 이미지 캡셔닝(이미지를 받아 이를 설명하는 문장을 만들어냄)
- Many-to-Many : Input-Many, Out-Many : 번역
- Many-to-One : Input-Many, Out-One : 예측
- One-to-Many : Input-One, Out-Many : 생성

[https://www.youtube.com/watch?v=Hn3GHHOXKCE&t=1029s](https://www.youtube.com/watch?v=Hn3GHHOXKCE&t=1029s)

# 장단기 메모리(LSTM)와 게이트 순환 유닛(GRU) 이해

## RNN의 단점과 문제점 이해

### 단점

- 장기 의존성 처리가 어렵다. 이전 타임스텝의 출력을 현재 타임스텝의 입력으로 사용하기 때문

### 문제점

- 병렬화 불가능 : 벡터가 순차적으로 입력되기 때문 → GPU연산의 장점인 병렬화를 사용못함
- 기울기 소실, 폭발 문제 :
    - 타임스텝에서 가중치가 1보다 작은경우 시간이 지날수록 기울기는 거의 0에 가까워짐. → 기울기 소실 → 장기 의존성을 학습하기 어려움
    - 타임스텝에서 가중치가 1보다 큰 경우 시간이 지날수록 지수적으로 증가 →기울기 폭발 → 모델이 불안정해짐

## LSTM과 GRU의 등장 배경과 개념

### LSTM

- 개념
    - RNN이 출력과 먼 위치에 있는 정보를 기억할 수 없다는 단점을 보완하여 장단기 기억을 가능하게 설계한 신경망 구조의 모델
- 등장 배경
    - RNN의 Vanishing Gradient 문제, 장기 의존성 문제를 해결하기 위해 등장.
    - LSTM은 forget gate, input gate 등을 이용해 정보를 기억할지 말지, 얼마큼 사용할기 등을 정하기 때문에 비교적 먼 거리의 정보를 효과적으로 전달 가능함.

### GRU

- 개념
    - 성능은 LSTM과 유사하고, 복잡했던 LSTM의 구조를 간단화 시킨 모델
- 등장 배경
    - 복잡한 구조로 인한 과적합과 계산비용을 해결하고 파라미터 단순화를 하기 위함.

## 장기 의존성 문제 해결을 위한 LSTM과 GRU의 작동 원리

### LSTM

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%207.png)

[Reference]

[https://medium.com/@anishnama20/understanding-lstm-architecture-pros-and-cons-and-implementation-3e0cca194094](https://medium.com/@anishnama20/understanding-lstm-architecture-pros-and-cons-and-implementation-3e0cca194094)

[https://medium.com/dovvie/deep-learning-long-short-term-memory-model-lstm-d4ee2f005973](https://medium.com/dovvie/deep-learning-long-short-term-memory-model-lstm-d4ee2f005973)

[https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)

1. Cell State

: 게이트를 통해 Cell State에 담길 정보를 조절해 전달한다. (전체적)

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%208.png)

1. Forget Gate

: 정보를 버릴지 사용할지를 결정하는 게이트

Sigomid 함수 사용(0~1 값을 C(t-1)과 곱함으로 이전 상태의 값을 사용할 지 결정)

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%209.png)

1. Input Gate

: 새로운 정보 중 **어떤 정보**를 Cell State에 저장할 지 결정하는 게이트

tanh layer에서 정보 후보 벡터를 정함.

input layer에서 Sigmoid 함수를 이용해 그 후보 중 어떤 정보를 사용할 지 결정

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%2010.png)

1. Cell State Update

Forget Gate에 의해 삭제되었거나 사용되는 값과 Input Gate를 통해 결정된 새로운 정보를 더함

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%2011.png)

1. Output Gate

input 값에 Sigmoid를 적용해 Cell State로부터 어떤 값을 output 값으로 도출 할 지 정함.

tanh(정한 output값 * Cell State 값) → 필요한 output 값만 내보냄.

![Untitled](RNN%204e3cf4e9ff2b4967a9957ceb0595a1e0/Untitled%2012.png)

# 시퀀스 모델링 문제에 RNN 적용

## RNN을 활용한 자연어 처리 기법

## 시계열 데이터 예측을 위한 RNN 모델 구축

## 음악 생성과 같은 창의적인 시퀀스 모델링 문제에 RNN 적용

# RNN의 훈련 및 조정

## RNN의 훈련 데이터 구성과 전처리

## 역전파 알고리즘을 통한 RNN 훈련

## 그래디언트 소실 문제 해결을 위한 기법
