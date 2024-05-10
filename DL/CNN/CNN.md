# CNN

# 합성곱

## 완전연결 계층의 문제점

### 완전 연결계층이란?

[https://medium.com/@vaibhav1403/fully-connected-layer-f13275337c7c](https://medium.com/@vaibhav1403/fully-connected-layer-f13275337c7c)

- 이전 레이어의 각 뉴런이나 노드가 현재 레이어의 각 뉴런에 연결되는 인공신경망에 사용되는 레이어 유형이다.
- 한 층의 모든 뉴런이 그 다음 층의 모든 뉴런과 연결된 상태

### 문제점

- 데이터의 형상이 무시된다는 것

## Convolution 이란?

합성곱으로, 수학연산이다.

[https://medium.com/latinxinai/what-is-convolution-ceb5a3bab020](https://medium.com/latinxinai/what-is-convolution-ceb5a3bab020)

![https://miro.medium.com/v2/resize:fit:720/format:webp/1*DPncpY0v-0dzc65O8ix46w.gif](https://miro.medium.com/v2/resize:fit:720/format:webp/1*DPncpY0v-0dzc65O8ix46w.gif)

이미지 처리로 예시)

- 이미지에 필터나 커널을 적용해 원본 이미지를 필터링. (슬라이딩)
- kernel(필터)의 매개변수와 각 위치의 이미지 픽셀값을 곱한다. 필터 : 가중치를 줄 수 있음
- 그 값들의 연산을 출력 피처맵에 새 값을 도출

### 문제점

filter, stride로 인해 특징맵의 크기는 입력데이터보다 작아지므로 손실이 발생

→ padding으로 손실을 줄일 수 있다.

## Padding 이란?

paidding : 입력데이터 주변에 특정 값을 채우는 것을 의미

### padding의 종류

- Pull padding : 0으로 값을 채운다. 출력크기를 입력 보다 크게 만든다.
- Valid padding : padding을 하지 않음
- Same padding : 0으로 값을 채운다. input과 output의 크기가 동일하게, half pull padding

## 3차원 데이터의 Convolution 연산

[https://medium.com/@parkie0517/3d-convolution-완전-정복하기-using-pytorch-conv3d-4fab52c527d6](https://medium.com/@parkie0517/3d-convolution-%EC%99%84%EC%A0%84-%EC%A0%95%EB%B3%B5%ED%95%98%EA%B8%B0-using-pytorch-conv3d-4fab52c527d6)

필터 수 = 출력층의 수

![Untitled](https://github.com/IMS-STUDY/AI-Study/assets/127017020/6d14084f-0dfa-4454-a592-e190ce831916)


![https://velog.velcdn.com/images/ssh00n/post/ae6cc3cc-fc27-4bff-9620-d7b1a9bf9afa/image.png](https://velog.velcdn.com/images/ssh00n/post/ae6cc3cc-fc27-4bff-9620-d7b1a9bf9afa/image.png)

블럭으로 생각해서 연산한다고 생각하면 된다.

# Conv(n)d

[https://leeejihyun.tistory.com/37](https://leeejihyun.tistory.com/37)

- conv1d : 합성곱을 진행할 입력으로 1차원 데이터, 시퀀스 데이터, 자연어처리 등
- conv2d : 합성곱을 진행할 입력으로 2차원 데이터, 컴퓨터비전
- conv3d : 합성곱을 진행할 입력으로 3차원 데이터, 영상분야

# Pooling

주로 피처맵에서 한다.

- max pooling
- average pooling

## Pooling의 사용 이유

- 특징을 통합한다

## Pooling의 특징 및 장단점

[https://gaussian37.github.io/dl-concept-stride_vs_pooling/](https://gaussian37.github.io/dl-concept-stride_vs_pooling/)

- 표현의 공간 차원을 점차적으로 축소해 네트워크의 매개변수 수와 계산을 최소화한다. → overfitting 방지 가능

### 장점

- 특정위치의 큰 역할을 하는 특징을 추출
- 전체를 대변하는 특징을 추출할 수 있다.
- 공간 차원을 줄이는 역할을 한다

### 단점

- 데이터의 손실 → 특징되는 값만 추출하기 때문에 나머지는 쓰이지 않게 됨 (average pooling의 경우 손실이 적다.)

Max pooling을 주로 사용한다. 성능이 더 좋기 때문 → 큰 특징만 유지하기 때문에

# CNN의 사용처

## 어떤 데이터에 CNN이 유리한가?

[https://seoilgun.medium.com/cnn의-stationarity와-locality-610166700979](https://seoilgun.medium.com/cnn%EC%9D%98-stationarity%EC%99%80-locality-610166700979)

- 이미지, 자연어 처리

동일한 특징이 이미지 내 여러 지역에 있을 수 있고, 작은 지역안에 픽셀 종속성이 있기 때문

## CNN이 학습하기 어려운 데이터

- 크기가 작은 데이터
- noise가 있는 데이터
