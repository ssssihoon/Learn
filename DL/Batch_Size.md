# Batch Size

# Batch?

- 일괄적

## 배치 사이즈란?

- 하나의 소그룹에 속하는 데이터 수를 의미
    - 일반적으로 배치 크기는 훈련데이터에서 사용
    - 그러나 테스트나 검증 데이터에서도 사용 될 수 있음

### 미니배치 경사 하강법

전체 데이터셋을 한 번에 모델에 입력하지 않고, 데이터를 작은 배치로 나누어 모델에 주입하여 학습을 수행하는 방법

← 여기서 배치사이즈와 관련이 있음

## 큰 배치 사이즈

### 장점

- 빠른 학습 속도

### 단점

- 메모 요구 사항이 높아짐
- 많은 계산 비용

## 작은 배치 사이즈

### 장점

- 메모 사용량을 줄임
- 안정적인 학습 과정을 제공

### 단점

- 학습 속도가 상대적으로 느림