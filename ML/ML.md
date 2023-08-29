# ML

# 머신러닝 다가가기

### 준비

- matplotlib : 과학계산용 그래프를 그리는 패키지
- scatter() : 산점도를 그리는 그래프
- import : 따로 만들어준 파이썬 패키지를 사용하기 위한 명령

### 예제

도미와 빙어에 대한 정보가 있는 리스트 fish_data 데이터

도미를 찾기 위해 도미를 1, 빙어를 0으로 둔 리스트  fish_target 데이터

사이킷런 패키지에서 KNN을 임포트

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
```

- 도미를 찾기 위한 모델을 훈련 메서드 → fit()
- 평가하는 메서드 → score()

```python
kn.fit(fish_data, fish_target) #훈련

print(kn.score(fish_data, fish_target))
'''
1.0.   -> 정확도
'''
```

### KNN (K - ****Nearest Neighbor) K-최근접 이웃****

KNeighborsClassifier : 가장 가까운 이웃을 참고해 정답을 예측하는 알고리즘이 구현된 사이킷런 클래스

데이터를 보고 비교해 다수를 차지하는 것을 정답으로 사용한다.

그래프를 보았을 때 이웃한 위치선상에 있으면 정답이라고 본다.

- predict() 메서드 : 새로운 데이터를 ()에 넣어 정답인지 예측한다.

단점 : 많은 양의 데이터를 사용하기 힘들다.

KNeighborsClassifier는 기본적으로 5개의 가까운 데이터를 참고한다. 이를

KNeighborsClassifier(n_neighbors=20) 20개 처럼 데이터 개수를 올리면, 정확도에서 떨어질 수도 있다.

그래서 for문을 통해 n값을 변경해 가면서 score(정확도)가 가장높게 되는 것을 채택하면 된다.

# 데이터 다루기

data set 

도미, 빙어의 크기가 담긴 fish_data 입력

도미(정답=1), 빙어(오답=0)이 담긴 fish_target 타깃

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14
```

## 훈련 세트와 테스트 세트

- 지도 학습에서 타깃을 이용해 학습한다.
- 비지도 학습에서는 타깃데이터가 없다.
- 테스트 세트는 훈련 데이터에서 사용하지 않은 데이터를 가지고 사용한다.
    - ex: 알고리즘 T 테스터세트 추가하는 요소와 같다.
    - 현재 data에서는 도미가 훈련세트이고, 오답인 빙어가 테스트세트이다.

우선 사이킷런의 KNeighborsClassifier 클래스를 import해서 모델의 객체를 만든다.

```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
```

### 샘플링 편향

훈련을 앞에서 35개의 데이터를 사용, 

테스트를 36~45까지를 하는 경우

정확도는 0이 나온다.

→ 이러한 현상을 **샘플링 편향**이라고 한다.

→ 훈련을 할 경우에는 데이터가 잘 섞여있어야 한다. (+빙어까지)

```python
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

'''
0.0
'''
```

### 넘파이

- np.array()

넘파이를 사용해 배열로 만들 수 있다.

→ 고차원 배열

```python
import numpy as np

a = [1, 2, 3, 4, 5]
b = [7, 7, 7, 7, 7]

arr = np.array([a, b])
print(arr)

'''
[[1 2 3 4 5]
 [7 7 7 7 7]]
'''
2x5
```

랜덤 시드를 지정해 배열을 무작위로 섞는다

- np.random.seed(42) : 난수를 발생시켜준다.

```python
import numpy as np

np.random.seed(42)
index = np.arange(49) #0~48까지 1씩 증가하는 인덱스
np.random.shuffle(index)
```

## 데이터 전처리