- [RNN](#rnn)
- [신경망의 기초 이해](#신경망의-기초-이해)
  - [인공 신경망의 기본 개념](#인공-신경망의-기본-개념)
    - [인공신경망의 구조](#인공신경망의-구조)
  - [뉴런과 활성화 함수](#뉴런과-활성화-함수)
    - [뉴런](#뉴런)
    - [활성화 함수](#활성화-함수)
  - [가중치와 편향](#가중치와-편향)
    - [가중치](#가중치)
    - [편향](#편향)
  - [피드포워드 신경망의 구조와 작동 원리](#피드포워드-신경망의-구조와-작동-원리)
- [시퀀스 데이터 이해](#시퀀스-데이터-이해)
  - [시퀀스 데이터의 특징과 종류](#시퀀스-데이터의-특징과-종류)
    - [특징](#특징)
    - [종류](#종류)
  - [시계열 데이터와 자연어 처리의 예](#시계열-데이터와-자연어-처리의-예)
    - [시계열 데이터의 예시](#시계열-데이터의-예시)
    - [자연어 처리의 예시](#자연어-처리의-예시)
  - [순차 데이터의 특성과 응용 분야](#순차-데이터의-특성과-응용-분야)
    - [순차 데이터 특성](#순차-데이터-특성)
    - [순차 데이터 응용 분야](#순차-데이터-응용-분야)
- [RNN의 개념 이해](#rnn의-개념-이해)
  - [순환 신경망의 개념과 구조](#순환-신경망의-개념과-구조)
    - [순환 신경망이란?](#순환-신경망이란)
    - [구조](#구조)
  - [RNN의 순환 구조와 정보 전달 방식](#rnn의-순환-구조와-정보-전달-방식)
    - [순환 구조](#순환-구조)
    - [정보 전달 방식](#정보-전달-방식)
  - [시간에 따른 정보 처리와 활용 방법](#시간에-따른-정보-처리와-활용-방법)
- [장단기 메모리(LSTM)와 게이트 순환 유닛(GRU) 이해](#장단기-메모리lstm와-게이트-순환-유닛gru-이해)
  - [RNN의 단점과 문제점 이해](#rnn의-단점과-문제점-이해)
    - [단점](#단점)
    - [문제점](#문제점)
  - [LSTM과 GRU의 등장 배경과 개념](#lstm과-gru의-등장-배경과-개념)
    - [LSTM](#lstm)
    - [GRU](#gru)
  - [장기 의존성 문제 해결을 위한 LSTM과 GRU의 작동 원리](#장기-의존성-문제-해결을-위한-lstm과-gru의-작동-원리)
    - [LSTM](#lstm-1)
    - [GRU](#gru-1)
- [시퀀스 모델링 문제에 RNN 적용](#시퀀스-모델링-문제에-rnn-적용)
  - [RNN을 활용한 자연어 처리 기법](#rnn을-활용한-자연어-처리-기법)
  - [시계열 데이터 예측을 위한 RNN 모델 구축](#시계열-데이터-예측을-위한-rnn-모델-구축)
    - [모델 구축](#모델-구축)
    - [모델 컴파일](#모델-컴파일)
  - [음악 생성과 같은 창의적인 시퀀스 모델링 문제에 RNN 적용](#음악-생성과-같은-창의적인-시퀀스-모델링-문제에-rnn-적용)
- [RNN의 훈련 및 조정](#rnn의-훈련-및-조정)
  - [RNN의 훈련 데이터 구성과 전처리](#rnn의-훈련-데이터-구성과-전처리)
    - [데이터 구성](#데이터-구성)
    - [전처리](#전처리)
  - [역전파 알고리즘을 통한 RNN 훈련](#역전파-알고리즘을-통한-rnn-훈련)
    - [순방향](#순방향)
    - [역방향](#역방향)
  - [그래디언트 소실 문제 해결을 위한 기법](#그래디언트-소실-문제-해결을-위한-기법)


# RNN

# 신경망의 기초 이해

## 인공 신경망의 기본 개념

ANN, Artificial Neural Network : 행렬 수학을 모델로 삼아 인간의 뉴런 구조를 본떠 만든 기계학습 모델.

인공지능을 구현하기 위한 기술 중 한 형태이다.

### 인공신경망의 구조

![Untitled](https://github.com/IMS-STUDY/AI-Study/assets/127017020/6da074e3-27a4-4aaa-8625-c481628f3bf4)


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
    
    
    ![Untitled 1](https://github.com/IMS-STUDY/AI-Study/assets/127017020/3f7b83db-e722-41be-a356-bf5477428d23)


    
    - 0과 1사이의 값을 출력, 로지스틱회귀, ANN 에서 주로 사용
    - 입력값이 어떤 범위에 들어갈 때 출력 값이 0과 1사이에서 급격하게 변하는 특성을 갖고 있다. 그렇기에 이진 분류에서 출력 레이어의 활성화 함수로 사용됨

- tanh 하이퍼볼릭 탄젠트  :
    
    ![Untitled 2](https://github.com/IMS-STUDY/AI-Study/assets/127017020/7d9cac47-3722-4420-9dd8-cc4b84b3e31e)

    
    - 실수 값을 (-1, 1) 범위로 압축하는 활성화 함수. 원점을 중심으로 값이 양의 무한대로 갈수록 1에 수렴, 음의 무한대로 갈수록 -1에 수렴.
    - 함수의 출력값이 평균을 중심으로 분포하는 결과를 가져옴
    - 주로 은닉층에서 사용된다.
- Relu 렐루 :
    
    ![Untitled 3](https://github.com/IMS-STUDY/AI-Study/assets/127017020/cdff98be-3e26-4123-8178-73b3678d6afb)
    
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

![Untitled 4](https://github.com/IMS-STUDY/AI-Study/assets/127017020/c7c618ef-febc-4475-9a64-92b480fffab9)


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

![Untitled 5](https://github.com/IMS-STUDY/AI-Study/assets/127017020/d261d976-e98d-4811-a615-999cc72dc372)


## RNN의 순환 구조와 정보 전달 방식

### 순환 구조

![Untitled 6](https://github.com/IMS-STUDY/AI-Study/assets/127017020/4f0adc73-1825-4890-a77c-74bcbfad12da)


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

x : input

h : state

z : output

<img width="678" alt="Untitled 7" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/8ff70b57-5661-4c5e-bfcf-b18b2516aca9">


[Reference]

[https://medium.com/@anishnama20/understanding-lstm-architecture-pros-and-cons-and-implementation-3e0cca194094](https://medium.com/@anishnama20/understanding-lstm-architecture-pros-and-cons-and-implementation-3e0cca194094)

[https://medium.com/dovvie/deep-learning-long-short-term-memory-model-lstm-d4ee2f005973](https://medium.com/dovvie/deep-learning-long-short-term-memory-model-lstm-d4ee2f005973)

[https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)

[https://youtu.be/rbk9XFaoCEE?si=suxuRy_nEfCKrdRT](https://youtu.be/rbk9XFaoCEE?si=suxuRy_nEfCKrdRT)

- Input
- Forget
- Cell
- Output

Cell State

: 게이트를 통해 Cell State에 담길 정보를 조절해 전달한다. (전체적)

<img width="493" alt="Untitled 8" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/c4096899-749d-40af-9f31-4bf05ccf754b">

1. Forget Gate

: 새로운 입력과 이전 상태를 참조해 이 정보를 얼마나 비율로 **사용**할 것인가. 얼마나 잊어버릴 것인가

Sigomid 함수 사용(0~1 값을 C(t-1)과 곱함으로 이전 상태의 값을 사용할 지 결정)

<img width="695" alt="Untitled 9" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/3724c3c2-e7aa-45fe-abb3-b2286759f933">

2. Input Gate

: 새로운 입력과 이전 상태를 참조해 이 정보들을 얼마나 **활용**할 것인가

tanh layer에서 정보 후보 벡터를 정함.

input layer에서 Sigmoid 함수를 이용해 그 후보 중 어떤 정보를 사용할 지 결정

<img width="730" alt="Untitled 10" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/6970b027-adfb-4d03-82e9-2ad04ceb2ff8">

3. Cell State Update

1(Forget Gate), 2(Input Gate)를 적절히 섞는다. → Cell State

<img width="559" alt="Untitled 11" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/27884d59-bee9-4a53-bd5d-dc13bdcb062b">

4. Output Gate

정보들을 모두 종합해 **다음 상태를 결정**

<img width="689" alt="Untitled 12" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/dbfea339-e701-49a6-aff7-1b43ba9c570d">

### GRU

[https://youtu.be/rbk9XFaoCEE?si=suxuRy_nEfCKrdRT](https://youtu.be/rbk9XFaoCEE?si=suxuRy_nEfCKrdRT)

[https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2](https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2)

[https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)

Cell State가 없음

- Input
- Forget
- Output

<img width="712" alt="Untitled 13" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/f5d52491-dd75-4822-ac01-b3aa6e13ebe3">

<img width="748" alt="Untitled 14" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/b2e84aec-0be2-4243-a34e-ca2ce38e5391">

1. Update Gate

: 모델이 이전 시간 단계의 과거 정보 중 미래에 전달되어야 하는 정보의 양을 결정

→ 과거의 모든 정보를 복사하고 기울기 문제가 사라질 위험을 제거할 수 있음

- 값에 대해 자체 가중치를 곱하고, 이전 t-1에 대한 정보도 자체 가중치를 곱해 두 결과를 모두 더한다.
- 그 후 시그모이드 활성화 함수를 적용해 결과를 0과 1사이로 압축

<img width="720" alt="Untitled 15" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/6d1a4257-c923-4c4b-b3b1-d8d4dbe0ee2c">

2. Reset Gate

: 모델에서 과거 정보를 얼마나 많이 잊어버릴지 결정하는데 사용

- 전과 동일하게 xt와 ht-1 에 가중치를 곱한 다음 결과를 합산. 그 후 시그모이드 함수 적용

<img width="705" alt="Untitled 16" src="https://github.com/IMS-STUDY/AI-Study/assets/127017020/638974e2-4fe8-4da6-920c-2efd67268fe0">

3. Current memory content

: 

- 입력 값에 가중치를 곱하고 이전 값에 가중치를 곱한다.
- 리셋 게이트와 전 값 사이의 요소별 곱을 계산해 이전 단계에서 제거할 항목을 결정
- 1, 2값을 더한 후 tanh 적용
4. Final memory at current time step

: 현재 유닛에 대한 정보를 보유한 채로 벡터를 계산.

1. 요소별 곱셈을 업데이트 게이트 *z_t* 및 *h_(t-1*)에 적용
2. *(1-z_t)* 및 *h'_t*에 요소별 곱셈을 적용
3. 1단계와 2단계의 결과를 합산

# 시퀀스 모델링 문제에 RNN 적용

## RNN을 활용한 자연어 처리 기법

1. 기본변환
    1. 토큰화 : 텍스트를 단어로 분해한다.
    2. 스커밍 : 기본 단어를 얻기 위해 끝을 자르는 방법으로 파생 접두사를 제거한다. (맨발 → 발)
    3. 레미제이션 : 굴절 어미만 제거 한다.(먹어, 먹고, 먹다 → 먹)
2. 단어 임베딩
    1. 범주형 변수인 단어를 고정 차원의 연속 벡터로 변환한다.

## 시계열 데이터 예측을 위한 RNN 모델 구축

### 모델 구축

- 3차원의 배열을 입력값으로 요구한다.
    - batch_size : 하나의 텐서는 모델에 들어가는 벡터이다.(자료의 수)
    - sequence_length : 하나의 타임 스텝은 텐서에서 하나의 관측치이다. (순서열의 길이)
    - input_dim : 한 피처는 그 타임 스텝에서 하나의 관측치이다. (피처 수)
    - `[batch_size, sequence_length, input_dim]`

```c
model = Sequential([
    SimpleRNN(40, return_sequences=True, input_shape=[sequence_length, input_dim]),
    SimpleRNN(40),
    Dense(1)
])

20 -> 유닛의 수.
return_sequences = True -> 모든 시간단계에서의 출력
Dense -> 출력값의 개수.
```

### 모델 컴파일

실제로 실행할 수 있도록 설정하는 단계

- Loss
    - mse
        - 회귀 문제에 자주 활용
        - 오차 대비 큰 손실 함수의 증가폭
        - 알고리즘이 정답을 잘 맞출수록 MSE 값은 작아진다.
- Optimizer
    - Adam
        - 간단한 구현으로 효율적인 연산 가능
        - 파라미터마다 학습률 조정 가능

## 음악 생성과 같은 창의적인 시퀀스 모델링 문제에 RNN 적용

[https://www.linkedin.com/pulse/unleashing-creativity-power-recurrent-neural-networks-simon-haywood](https://www.linkedin.com/pulse/unleashing-creativity-power-recurrent-neural-networks-simon-haywood)

- 이전 입력에서 맥락을 캡처하는 상태 기록을 유지하는 능력이 있음
- context-recall(맥락-상기) 메커니즘을 통해 RNN은 일관되고 맥락에 맞는 출력을 생성한다.
- 순환 신경망은 시퀀스 처리에 탁월해 맥락과 일관성이 있는 이야기나 멜로디를 생성 가능하게 한다.

# RNN의 훈련 및 조정

## RNN의 훈련 데이터 구성과 전처리

### 데이터 구성

시계열 데이터로 할 수 있는 것으로 미래예측과 Imputation(비어 있는 값을 채우는 것)이 있다.

- 단변량 시계열 : 하나의 피처를 사용 (시간당 접속 사용자의 수, 도시의 날짜별 온도 등)
- 다변량 시계열 : 여러 피처를 사용 (기업의 분기별 재정 안정성 - 수입, 부채 )

### 전처리

1. 토큰화 : 작은 단위로 분할한다.
2. 임베딩 : 데이터를 1대1 대응이 가능하게 다른 형식으로 바꾼다.
3. 시퀀스 길이 조정 : 모든 시퀀스의 길이를 동일하게 맞춰야 한다.
4. 스케일링.(option)

## 역전파 알고리즘을 통한 RNN 훈련

[https://towardsdatascience.com/backpropagation-in-rnn-explained-bdf853b4e1c2](https://towardsdatascience.com/backpropagation-in-rnn-explained-bdf853b4e1c2)

### 순방향

특정 시간 단계에서 이전 시간 단계의 입력 벡터와 숨겨진 상태 벡터가 각각의 가중치 행렬에 의해 곱해지고 덧셈 노드에 의 해 합산. 그후 비선형 함수를 통과. 다음도 동일하게 반복

### 역방향

행렬 곱셈 노드로 역류해 가중치 행렬과 숨겨진 상태 모두에서 그래디언트를 계산.

숨겨진 상태와 이전 시간 단계의 gradient가 합산되는 복사 노드에서 만나게 된다.

## 그래디언트 소실 문제 해결을 위한 기법

### 기울기 소실

역전파 과정에서 입력층에서 진행할수록 점차 기울기가 작아지다가 나중에는 기울기의 변화가 없어지는 문제를 말한다.

[https://aikorea.org/blog/rnn-tutorial-3/](https://aikorea.org/blog/rnn-tutorial-3/)

- tanh, sigmoid 대신 ReLU 사용 → 보편적인 방법이다.
    - ReLU는 미분값이 최대치가 1로 정해져 있지 않기 때문 -> 입력이 양수일 때 미분값이 1이고, 음수일 때 미분값이 0이된다. -> 입력이 양수인 경우, 기울기가 소실 되지 않고 그대로 전달된다.
    - -> sigmoid의 경우 최대치 0.25, tanh의 경우 최대치 1
- W 가중치를 적당히 좋은 값으로 초기화
- 정규화를 적절히 해주기
    - 다음 시간 단계에 전달되는 값의 범위를 일정하게 유지 가능.
