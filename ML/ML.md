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

### 넘파이로 데이터 준비

넘파이로 훈련세트와 테스트 세트 나누기

- 넘파이로 배열 만들기

```python
import numpy as np

print(np.column_stack(([1, 2, 3], [4, 5, 6])))

'''
[[1 4]
 [2 5]
 [3 6]]
'''
```

- fish_length, fish_weight 합치기

```python
fish_data = np.column_stack(([fish_length], [fish_weight]))
print(fish_data[:5])

'''
[[  25.4   26.3   26.5   29.    29.    29.7   29.7   30.    30.    30.7
    31.    31.    31.5   32.    32.    32.    33.    33.    33.5   33.5
    34.    34.    34.5   35.    35.    35.    35.    36.    36.    37.
    38.5   38.5   39.5   41.    41.     9.8   10.5   10.6   11.    11.2
    11.3   11.8   11.8   12.    12.2   12.4   13.    14.3   15.   242.
   290.   340.   363.   430.   450.   500.   390.   450.   500.   475.
   500.   500.   340.   600.   600.   700.   700.   610.   650.   575.
   685.   620.   680.   700.   725.   720.   714.   850.  1000.   920.
   955.   925.   975.   950.     6.7    7.5    7.     9.7    9.8    8.7
    10.     9.9    9.8   12.2   13.4   12.2   19.7   19.9]]
'''
```

- 타깃 데이터를 만들 때 numpy라이브러리로 간단하게 만들 수 있다.
    - np.ones()
    - np.zeros()

```python
print(np.ones(5))
'''
[1. 1. 1. 1. 1.]
'''
print(np.zeros(5))
'''
[0. 0. 0. 0. 0.]
'''
```

- concatenate()
    - 가로로 길게 배열 할 수 있다.

```python
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

'''
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0.]
'''
```

### 사이킷런으로 훈련세트 & 테스트 세트 나누기

- train_test_split()
    
    리스트나 배열을 비율에 맞게 훈련세트와 테스트 세트로 나누어준다.(+섞는 것 까지)
    

테스트 세트의 비율이 편향될 가능성이 있기 때문에 밑에 참고

- stratify
    
    매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눈다.
    

```python
from sklearn.model_selection import train_test_split
```

```python
train_input, test_input, train_traget, test_target = train_test_split(fish_data, stratify=fish_target, random_state=42)
```

### KNN 훈련

df

```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
import numpy as np
from sklearn.model_selection import train_test_split

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
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

```python
print(kn.predict([[25, 150]]))

'''
[0.]
'''
```

당연히 도미(1)로 예측해야할 것이 빙어(0)으로 예측됐다.

→ 이러한 모순을 해결하기 위해 우선 그래프로 확인 

- numpy 배열에서 ..
    - a[:,0] : 모든 행에 대해서 첫번째 열의 정보를 가져다 달라 = 가로
    - a[:,1] : 모든 행에 대해서 두번째 열의 정보를 가져다 달라 = 세로
    - plt.scatter(a[:,0],a[:,1]) → x축 가로, y축 세로로 보여달라는 의미

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

'''
marker = '^'
을 이용해 세모로 표기
'''
```

사진

KNN은 이웃한 5개를 기준으로 평가한다.

그래프를 보면 x축과 y축이 1:1로 되어있지 않아서 발생하는 문제점인데, 

그래프만 놓고 봤을 때는 당연히 도미와 가깝게 보인다. 하지만 x축 y축을 1:1로 놓고 보게 된다면

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

'''
'D' -> 마름모
x축의 범위를 동일하게 1000으로 맞춰준다.
'''
```

사진

이 그래프를 보게 된다면 세모는 바로 위의 네모 1개와 밑에 빙어(네모)4개를 이웃하고 있다.

이렇게 작업하는 것은 데이터 전처리라고 한다.

---

데이터를 표현할 때 샘플 간의 거리에 영향을 받으므로 일정한 기준으로 맞춰 줘야 한다.

- 전처리 방법
    - 표준점수 z : 특성값이 0에서 **표준편차**의 몇 배만큼 떨어져 있는지를 나타낸다.
        
        표준 편차 : 분산의 제곱근
        
        분산 : 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구함
        
- mean = np.mean(train_input, axis = 0)
    - 평균을 계산
- std = np.std(train_input, axis = 0)
    - 표준편차를 계산
- axis = 0 : 행을 따라 각 열의 통계 값을 계산

```python
평균 빼기 → 표준편차 나누기

train_scaled = (train_input - mean) / std
```

### 전처리 모델로 훈련하기

```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
import numpy as np
from sklearn.model_selection import train_test_split

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
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
train_scaled = (train_input - mean) / std

import matplotlib.pyplot as plt
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
print(plt.show())
```

train_scaled → 라는 전처리 모델로 그래프를 그려보면

사진

이렇게 샘플 하나로는 평균과 표준편차를 구할 수 없다. → 다시 산점도 구하기

new = ([25, 150] - mean) / std

```python
new = ([25, 150] - mean) / std

import matplotlib.pyplot as plt
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
print(plt.show())
```

사진

x축 y축의 범위가 같고, 세모는 도미와 가까운 범위에 있다.

이제 KNN으로 다시 훈련

```python
kn.fit(train_scaled, train_target)
tset_scaled = (test_input - mean) / std
```

# 회귀 알고리즘과 모델 규제

## K-최근접 이웃 회귀

예측하려는 샘플에 가장 가까운 샘플 k개를 선택, 클래스를 확인하여 다수클래스를 새로운 샘플의 클래스로 예측

그 샘플의 타깃값의 평균을 구해 예측 타깃값을 구한다.

### 데이터 준비

df

농어의 데이터이다.(길이, 무게)

```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
```

- 산점도 그래프

```python
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel("length")
plt.ylabel("weight")
print(plt.show())
```

사진

이 산점도 그래프를 가지고 보면

길이가 증가함에 따라 무게도 증가하는 것을 볼 수 있다.

- 머신러닝 모델에 사용하기 전
    
    훈련세트와 테스트세트로 나누기
    

```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, tset_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
```

- 사이킷런을 사용하려면 2차원 배열 형태여야한다.
    - 넘파이 배열 reshape() : 1차원배열 → 2차원배열
        
        배열의 크기를 지정할 수 있다.
        

```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)

'''
(42, 1) (14, 1)
'''

(-1, 1) 을 사용한 이유 ->
크기에 -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미
```

2차원 배열 완료

### 결정계수 R^2

KNeighborsRegressor을 사용한다.

```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
```

- k 최근접 이웃 회귀 모델 훈련

```python
knr.fit(train_input, test_input)
```

- 정확도 점수

```python
print(knr.score(test_input, test_target))

'''
0.992809406101064
'''
```

ㅅ

수식

사진

- 타깃과 예측한 값 사이의 차이 구하기
    - sklearn.metrics
    - mean_absolute_error()

```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input) # 테스트 세트에 대한 예측을 만든다.

mae = mean_absolute_error(test_target, test_prediction)
# 테스트 세트에 대한 평균 절댓값 오차를 계산
print(mae) 

'''
19.157142857142862
'''
```

만약 훈련세트를 사용해 평가해 본다면?

```python
print(knr.score(train_input, train_target))

'''
0.9698823289099254
'''
```

### 과대적합 / 과소적합

훈련 세트 점수 > 테스트 세트 점수 = 과대적합

훈련 세트 점수 < 테스트 세트 점수 = 과소적합

과소 적합 : 작은 데이터를 사용하는 경우 나타날 수 있다. → 모델을 더 복잡하게 만들자 ( k = 3, 7 ```)

훈련세트점수 =.. 테스트세트점수 → GOOD

---

## 선형 회귀

50cm인 농어의 무게를 예측하려한다.

### K-최근접 이웃의 한계

아무리 떨어져 있는 이웃이라도 가장 가까운 이웃의 샘플의 타깃을 평균하여 예측한다.

df

```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
```

- 데이터를 훈련세트와 테스트세트로 나누기

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
```

- 훈련세트와 테스트세트 2차원 배열로 바꾸기

```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

- KNN (k=3), 모델 훈련

```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

knr.fit(train_input, train_target)
```

- 50cm인 농어의 무게를 예측

```python
print(knr.predict([[50]]))

'''
[1010.]
'''
```

1010g이라고 예측했지만 실제로는 이것보다 더 나간다고 한다.

왜 예측을 못했을까?

→ 산점도그래프를 그려보기

```python
import matplotlib.pyplot as plt

distances, indexes = knr.kneighbors([[50]]) # 50cm인 농어의 이웃 구하기
plt.scatter(train_input, train_target) # 훈련세트 산점도그래프
 
plt.scatter(train_input[indexes], train_target[indexes], marker='D') # 이웃샘플은 마름모로
plt.scatter(50, 1010, marker='^') # 50cm 에 무게 1010g으로 예측한 값을 세모로
print(plt.show())
```

사진

- 이웃샘플 (마름모) 타깃의 평균구하기

```python
print(np.mean(train_target[indexes]))

'''
1010.0
'''
```

이웃샘플 타깃의 평균 == 모델 예측값

결론 → 새로운 샘플이 훈련세트의 범위를 벗어나면 엉뚱한 값을 예측한다. 

ex:) 10000cm인 농어의 몸무게도 1010g이라고 표현된다. 이것이 **KNN한계**

→ 선형회귀 알고리즘을 쓰자!

---

### 선형회귀

- 사이킷런 선형회귀

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
```

- 모델 훈련과 예측

```python
lr.fit(train_input, train_target)
print(lr.predict([[50]]))

'''
[1241.83860323]
'''
```

선형회귀의 직선은 기울기와 절편이 있어야한다. a, b

y = ax + b

y = 무게, x = 길이

- a, b = lr.coef_, lr.intercept_

```python
print(lr.coef_, lr.intercept_)

'''
[39.01714496] -709.0186449535474
'''
```

- 산점도 그래프

```python
import matplotlib.pyplot as plt
plt.scatter(train_input, train_target)

#15에서 50까지 1차 방정식 그래프 그리기
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])

plt.scatter(50, 1241.8, marker='^')
print(plt.show())
```

사진

이처럼 직선 상에 농어의 데이터가 있음을 알 수 있는데

- R^2를 확인해보면

```python
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

'''
0.9398463339976041
0.824750312331356

->과소적합
'''
```

그래프의 왼쪽아래를 보면 데이터가 직선에서 이탈한 것도 문제가 된다.

### 다항회귀

직선이 아닌 최적의 곡선을 그린다.

y = ax^2 + bx + c

y = 무게, x = 길이

2차 방정식의 그래프를 그리기 위해서는 제곱한 항이 훈련세트에 있어야한다.

+ 타깃값은 그대로

```python
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
```

이렇게하면 제곱한항과 일반항이 나열된다.

- train_poly 모델로 훈련

```python
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

'''
[1573.98423528]
'''
```

→ 선형 회귀 예측값 보다 높은값을 예측

- 훈련계수와 절편 출력

```python
print(lr.coef_, lr.intercept_)

'''
[  1.01433211 -21.55792498] 116.05021078278338
'''
```

→ 무게 = 1.01 * 길이^2 - 21.6 * 길이 + 116.05

- 산점도 그래프

```python
point = np.arange(15, 50) # 구간별 직선을 그리기위해 정수 배열 15 ~ 49 

plt.scatter(train_input, train_target)

plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

plt.scatter([50], [1574], marker = '^')
print(plt.show())
```

사진

- R^2 점수

```python
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

'''
0.9706807451768623
0.9775935108325122
'''
```

점수 차이가 거의 안나지만 과소적합이 조금 남아있다.
