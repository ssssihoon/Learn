# ML2

# 머신러닝의 이해

## Numpy

`import numpy as pd`

- **ndarray**
    - 배열 구조를 뜻한다.

```python
array1 = np.array([1, 2, 3])
print('array1 type:', type(array1))
print('array1 array 형태 :', array1.shape)

array2 = np.array([[1, 2, 3],
                  [2, 3, 4]])
print('array2 type:', type(array1))
print('array2 array 형태 :', array2.shape)

array3 = np.array([[1, 2, 3]])
print('array3 type:', type(array1))
print('array3 array 형태 :', array3.shape)

'''
array1 type: <class 'numpy.ndarray'>
array1 array 형태 : (3,)
array2 type: <class 'numpy.ndarray'>
array2 array 형태 : (2, 3)
array3 type: <class 'numpy.ndarray'>
array3 array 형태 : (1, 3)
'''
```

- array의 차원 확인하기
    - `.ndim`

```python
print(array1.ndim, array2.ndim, array3.ndim)

'''
1 2 2
'''
```

- ndarray의 데이터 타입 확인

```python
list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)

'''
<class 'list'>
<class 'numpy.ndarray'>
[1 2 3] int64
'''
```

- array 자료형 변환
    - `.astype(’자료형’)`

```python
array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

'''
[1. 2. 3.] float64
```

- ndarray를 생성
    - `arange()`
    - `zeros()`
    - `ones()`

```python
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

'''
[0 1 2 3 4 5 6 7 8 9]
int64 (10,)
'''
```

```python
zero_array = np.zeros((3, 2), dtype ='int64')
print(zero_array)
print(zero_array, zero_array.shape)

print()

one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)

'''
[[0 0]
 [0 0]
 [0 0]]
[[0 0]
 [0 0]
 [0 0]] (3, 2)

[[1. 1.]
 [1. 1.]
 [1. 1.]]
float64 (3, 2)
'''
```

- ndarray의 차원과 크기를 변경
    - `reshape()`

```python
array1 = np.arange(10)
print('array1 : \n', array1)

array2 = array1.reshape(2, 5)
print('array2 : \n', array2)

array3 = array1.reshape(5, 2)
print('array3 : \n', array3)

'''
array1 : 
 [0 1 2 3 4 5 6 7 8 9]
array2 : 
 [[0 1 2 3 4]
 [5 6 7 8 9]]
array3 : 
 [[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
'''
```

지정하는 사이즈로 변경이 불가능하면 오류를 발생한다.

- -1을 인자로 사용하면 자동으로 ndarray에 호환되는 shape으로 변환해준다.

```python
array1 = np.arange(10)
print(array1)
print()
array2 = array1.reshape(-1, 5)
print(array2)

'''
[0 1 2 3 4 5 6 7 8 9]

[[0 1 2 3 4]
 [5 6 7 8 9]]
**********'''**********
```

- 리스트 자료형으로 변환
    - `tolist()`

```python
array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array3d:\n', array3d.tolist())

'''
array3d:
 [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
'''
```

- 데이터 추출
    - `[row, col]`

```python
array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)
print(array2d)

print(array2d[1, 0])

'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
4
'''
```

- 팬시인덱싱(Fancy indexing)
    
    지정한 위치부터 인덱싱해준다.
    

```python
array = array2d[[0, 1], 2]
print('array2d[[0, 1], 2] =>', array3.tolist())

'''
array2d[[0, 1], 2] => [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
'''
```

- 불린 인덱싱
    
    조건 필터링과 검색을 동시에 할 수 있다.
    

```python
array1d = np.arange(1, 10)
array3 = array1d[array1d > 5]
print(array3)
array_bool = array1d > 5
print(array_bool)

'''
[6 7 8 9]
[False False False False False  True  True  True  True]
'''
```

- 행렬 정렬
    - sort()
    - argsort()

```python
org_array = np.array([3, 1, 9, 5])
print('원본 행렬:', org_array)
sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬', sort_array1)
print('np.sort() 호출 후 원본 행렬 :', org_array)
sort_array2 = org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬:', sort_array2)
print('org_array.sort() 호출 후 원본 행렬:', org_array)

'''
원본 행렬: [3 1 9 5]
np.sort() 호출 후 반환된 정렬 행렬 [1 3 5 9]
np.sort() 호출 후 원본 행렬 : [3 1 9 5]
org_array.sort() 호출 후 반환된 행렬: None
org_array.sort() 호출 후 원본 행렬: [1 3 5 9]
'''
```

- 내림차순 정렬
    - np.sort(array)[::-1]

---

- 2차원 행렬 정렬

```python
array2d = np.array([[8, 12], 
                    [7, 1]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)
sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)
# axis=0 : 로우
# axis=1 : 컬럼

'''
로우 방향으로 정렬:
 [[ 7  1]
 [ 8 12]]
컬럼 방향으로 정렬:
 [[ 8 12]
 [ 1  7]]
'''
```

- 정렬된 행렬의 인덱스 반환
    - argsort()

```python
org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)

'''
<class 'numpy.ndarray'>
행렬 정렬 시 원본 행렬의 인덱스: [1 0 3 2]
'''
```

- ex

```python
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스 :', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력 :', name_array[sort_indices_asc])

'''
성적 오름차순 정렬 시 score_array의 인덱스 : [0 2 4 1 3]
성적 오름차순으로 name_array의 이름 출력 : ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']
'''
```

### 선형대수 연산

- 행렬내적
    - np.dot()

```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
B = np.array([[7, 8], 
              [9, 10], 
              [11, 12]])
dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)

'''
행렬 내적 결과:
 [[ 58  64]
 [139 154]]
'''
```

- 전치행렬
    - np.transpose()

```python
A = np.array([[1, 2], 
              [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)

'''
A의 전치 행렬:
 [[1 3]
 [2 4]]
'''
```

## Pandas

- DataFrame → ndarray / List / Dictionary
    - ary = df_dict.values
    - li = df_dict.values.tolist()
    - dict = df_dict.values.to_dict(’리스트명’)
- DataFrame 데이터삭제
    - df.drop(’컬럼명’, axis=1) → 컬럼 삭제
    - df.drop([인덱스], axis=0) → 로우 삭제
- DataFrame Index객체 → array로 변환
    - indexes = df.index
    - indexes.values
- 새로운 인덱스를 재할당 (기존 인덱스는 컬럼으로 들어간다)
    - reset_index()
- 정렬(오름차순)
    - df.sort_values(by=[’컬럼명’], ascending=True) # False → 내림차순
    
- 데이터 결손처리
    - df.isna() : 결측값을 확인한다.
    - df.isna().sum() : 결측값의 빈도 확인
    - df[컬럼명].fillna(’값2’) : 결측값을 값2로 대체한다.
- apply lambda로 데이터 가공
    - df.apply(lambda x : 수행)

```python
def get_square(a):
	return a**2

print('3의 제곱은 : ', get_square(3))
```

```python
lambda_square = lambda x : x**2
print('3의 제곱은:', lambda_square(3))
```

```python
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))

'''
이름의 길이가 나온다.
'''
```

# 사이킷런, 머신러닝

## 붓꽃 품종 예측하기

- df로드
- df설정
- df의 결정값을 target으로 설정
- teain_test_split()을 이용해 훈련세트와 테스트세트로 나누기
- 학습 수행
- 학습이 된 객체에서 테스트 데이터 세트로 예측 수행
- 정확도 구하기

붓꽃 데이터 세트로 붓꽃의 품종을 분류하기

- 붓꽃 데이터 세트(피쳐) : 꽃잎의 길이와 너비, 꽃받침의 길이와 너비

```python
iris = load_iris()

iris_data = iris.data # iris의 데이터를 변수에 저장(numpy이용)

iris_label = iris.target # iris의 레이블(결정 값)데이터를 변수에 저장(numpy이용)
print('irist target값:', iris_label)
print('iris target명:', iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)

'''
irist target값: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
iris target명: ['setosa' 'versicolor' 'virginica']

sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)	label
0	5.1	3.5	1.4	0.2	0
1	4.9	3.0	1.4	0.2	0
2	4.7	3.2	1.3	0.2	0
'''
```

- 피처에는 sepal length, petal length, petal width
- 레이블(결정값)에는 0(Setosa), 1(versicolor), 2(virginica) 품종들

---

- 학습용 데이터와 테스트용 데이터 분리

```python
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11) 
# 테스트 데이터 20%, 학습 데이터 80%
```

- DecisionTreeClassifier 객체 생성

```python
dt_clf = DecisionTreeClassifier(random_state=11)
```

- 학습 수행

```python
dt_clf.fit(X_train, y_train)
```

- 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행

```python
pred = dt_clf.predict(X_test)
```

- 예측 정확도

```python
from sklearn.metrics import accuracy_score
print("예측 정확도: {0:4f}".format(accuracy_score(y_test, pred)))
'''
예측 정확도: 0.933333

'''
```

## 사이킷런 기반 프레임워크

- `fit()` : 학습
- `predict()` : 예측

---

- data : 피처의 데이터 세트
- target : 분류시 레이블 값, 회귀일 때는 숫자 결과값 데이터 세트
- target_names : 개별 레이블의 이름
- feature_names : 피처의 이름
- DESCR : 데이터 세트에 대한 설명과 각 피처의 설명

```python
from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))

'''
<class 'sklearn.utils.Bunch'>
'''
```

Bunch는 딕셔너리 자료형과 유사하다.

```python
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)

'''
붓꽃 데이터 세트의 키들: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
'''
```

```python
print('\n feature_names 의 type:', type(iris_data.feature_names))
print('feature_names의 shape:', len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names의 type:', type(iris_data.target_names))
print('target_names의 shape:', len(iris_data.target_names))
print(iris_data.target_names)

print('\n data 의 type:', type(iris_data.data))
print('data의 shape:', iris_data.data.shape)
print(iris_data['data'])

print('\n target 의 type:', type(iris_data.target))
print('target 의 shape:', iris_data.target.shape)
print(iris_data.target)

'''
feature_names 의 type: <class 'list'>
feature_names의 shape: 4
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

 target_names의 type: <class 'numpy.ndarray'>
target_names의 shape: 3
['setosa' 'versicolor' 'virginica']

 data 의 type: <class 'numpy.ndarray'>
data의 shape: (150, 4)
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 ~~~~~~~~~~~~~~~~
 [6.2 3.4 5.4 2.3]
 [5.9 3.  5.1 1.8]]

 target 의 type: <class 'numpy.ndarray'>
target 의 shape: (150,)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
```

## Model_Selection

- `train_test_split()`

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

pred = dt_clf.predict(train_data)
print('예측 정확도:', accuracy_score(train_label, pred))

'''
예측 정확도: 1.0
'''
```

예측 정확도가 1인 이유는 이미 학습한 학습 데이터 세트를 기반으로 예측했기 때문.(이상함)

---

- test_size : 테스트 데이터 세트 크기를 얼마로 샘플링할 것인가(%)
- train_size : 학습용 데이터 세트 크기를 얼마로 생플링할 것인가(%)
- shuffle : 데이터를 분리하기 전 데이터를 미리 섞을지를 결정
- random_state : 난수 값
- train_test_split() : 반환값은 튜플 형태, 순서대로 → 학습용 데이터의 피처 데이터 세트, 테스트용 데이터의 피처 데이터 세트, 학습용 데이터의 레이블 데이터 세트, 테스트용 데이터의 레이블 데이터 세트가 반환

붓꽃 데이터 세트를 train_test_split()을 이용해 테스트 데이터 세트를 전체의 30%로, 학습 데이터 세트를 70%

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.3, random_state=121)

dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:4f}'.format(accuracy_score(y_test, pred)))

'''
예측 정확도: 0.955556
'''
DecisionTreeClassifier를 이용해 fit()으로 학습용 피처, 레이블 데이터학습 ->
predict로 테스트용 데이터의 피처 데이터 세트를 예측 ->
예측한 것을 테스트용 데이터의 레이블 데이터를 예측성능 평가
```

테스트 데이터가 30%인 45개밖에 되지 않아 예측 성능을 평가하기에는 아쉽다.

→ 교차 검증

### 교차 검증

학습 데이터 세트 내에서 학습데이터 세트와 검증 데이터 세트로 분할 이후 검증을 하고나서 테스트 데이터 세트에 적용해 평가하는 프로세스이다.

- K 폴드 교차 검증
    - K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴트 세트에 학습과 검증 평가를 반복적으로 수행하는 방법이다.
        1. 데이터 세트를 K등분
        2. 1~(K-1)은 학습 데이터 세트, 마지막 K번째 하나를 검증 데이터 세트로 설정
        3. 학습 데이터 세트에서 학습 수행, 검증 데이터 세트에서 평가를 수행
        4. K-1번째를 검증 데이터 세트로 설정 → K-2 ``` 반복
        5. 1번째를 검증데이터, K번째를 검증데이터로 예측 평가
        6. 모든 예측 평가를 구해 평균을 구한다.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits = 5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:', features.shape[0])

'''
붓꽃 데이터 세트 크기: 150
'''
```

KFold 객체의 split()을 호출해 교차 검증 수행 시마다 학습과 검증을 반복해 예측 정확도를 측정

```python
n_iter = 0

#KFold 객체의 split()을 호출하면 폴드별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
  # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
  X_train, X_test = features[train_index], features[test_index]
  y_train, y_test = label[train_index], label[test_index]
  # 학습 및 예측
  dt_clf.fit(X_train, y_train)
  pred = dt_clf.predict(X_test)
  n_iter += 1
  # 반복 시마다 정확도 측정
  accuracy = np.round(accuracy_score(y_test, pred), 4)
  train_size = X_train.shape[0]
  test_size = X_test.shape[0]
  print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기:{3}'.format(n_iter, accuracy, train_size, test_size))
  print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
  cv_accuracy.append(accuracy)

#개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))

'''
#1 교차 검증 정확도 :1.0, 학습 데이터 크기: 120, 검증 데이터 크기:30
#1 검증 세트 인덱스:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]

#2 교차 검증 정확도 :0.9667, 학습 데이터 크기: 120, 검증 데이터 크기:30
#2 검증 세트 인덱스:[30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
 54 55 56 57 58 59]

#3 교차 검증 정확도 :0.8667, 학습 데이터 크기: 120, 검증 데이터 크기:30
#3 검증 세트 인덱스:[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
 84 85 86 87 88 89]

#4 교차 검증 정확도 :0.9333, 학습 데이터 크기: 120, 검증 데이터 크기:30
#4 검증 세트 인덱스:[ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119]

#5 교차 검증 정확도 :0.7333, 학습 데이터 크기: 120, 검증 데이터 크기:30
#5 검증 세트 인덱스:[120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
 138 139 140 141 142 143 144 145 146 147 148 149]

## 평균 검증 정확도: 0.9
'''
```

- Stratified K 폴드
    - 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K폴드 방식
    - 특정한 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한쪽으로 치우지는 것

```python
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()

'''
0    50
1    50
2    50
Name: label, dtype: int64
'''
레이블의 값이 0(Setosa), 1(Versicolor), 2(Virginica) 각각 50개이다.
```

3개의 폴드 세트를 KFold로 생성 후 학습과 검증 레이블 데이터 값의 분포도 확인

```python
kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
  n_iter += 1
  label_train = iris_df['label'].iloc[train_index]
  label_test = iris_df['label'].iloc[test_index]
  print('## 교차 검증 : {0}'.format(n_iter))
  print('학습 레이블 데이터 분포:\n', label_train.value_counts())
  print('검증 레이블 데이터 분포:\n', label_test.value_counts())

'''
## 교차 검증 : 1
학습 레이블 데이터 분포:
 1    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    50
Name: label, dtype: int64
## 교차 검증 : 2
학습 레이블 데이터 분포:
 0    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    50
Name: label, dtype: int64
## 교차 검증 : 3
학습 레이블 데이터 분포:
 0    50
1    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    50
Name: label, dtype: int64
'''
```

이런식으로 하면 정확도는 0이된다. → StratifiedKFold 사용

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
  n_iter += 1
  label_train = iris_df['label'].iloc[train_index]
  label_test = iris_df['label'].iloc[test_index]
  print('## 교차 검증 : {0}'.format(n_iter))
  print('학습 레이블 데이터 분포:\n', label_train.value_counts())
  print('검증 레이블 데이터 분포:\n', label_test.value_counts())

'''
## 교차 검증 : 1
학습 레이블 데이터 분포:
 2    34
0    33
1    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    17
1    17
2    16
Name: label, dtype: int64
## 교차 검증 : 2
학습 레이블 데이터 분포:
 1    34
0    33
2    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    17
2    17
1    16
Name: label, dtype: int64
## 교차 검증 : 3
학습 레이블 데이터 분포:
 0    34
1    33
2    33
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    17
2    17
0    16
Name: label, dtype: int64
'''
```

StratifiedKFold 교차검증

```python
df_clf = DecisionTreeClassifier(random_state = 156)

skfold = StratifiedKFold(n_splits = 3)
n_iter = 0
cv_accuracy = []

#StratifiedKFold의 split() 호출시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index in skfold.split(features, label):
  # split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
  X_train, X_test = features[train_index], features[test_index]
  y_train, y_test = label[train_index], label[test_index]
  #학습 및 예측
  dt_clf.fit(X_train, y_train)
  pred = dt_clf.predict(X_test)

  #반복 시마다 정확도 측정
  n_iter += 1
  accuracy = np.round(accuracy_score(y_test, pred), 4)
  train_size = X_train.shape[0]
  test_size = X_test.shape[0]
  print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기:{3}'.format(n_iter, accuracy, train_size, test_size))
  print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
  cv_accuracy.append(accuracy)

  # 교차 검증별 정확도 및 평균 정확도 계산
  print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
  print('## 평균 검증 정확도:', np.round(np.mean(cv_accuracy), 4))

'''
#1 교차 검증 정확도 :0.98, 학습 데이터 크기: 100, 검증 데이터 크기:50
#1 검증 세트 인덱스:[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  50
  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66 100 101
 102 103 104 105 106 107 108 109 110 111 112 113 114 115]

## 교차 검증별 정확도: [0.98]
## 평균 검증 정확도: 0.98

#2 교차 검증 정확도 :0.94, 학습 데이터 크기: 100, 검증 데이터 크기:50
#2 검증 세트 인덱스:[ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  67
  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82 116 117 118
 119 120 121 122 123 124 125 126 127 128 129 130 131 132]

## 교차 검증별 정확도: [0.98 0.94]
## 평균 검증 정확도: 0.96

#3 교차 검증 정확도 :0.98, 학습 데이터 크기: 100, 검증 데이터 크기:50
#3 검증 세트 인덱스:[ 34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  83  84
  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 133 134 135
 136 137 138 139 140 141 142 143 144 145 146 147 148 149]

## 교차 검증별 정확도: [0.98 0.94 0.98]
## 평균 검증 정확도: 0.9667
'''
```

- cross_val_score() : 교차검증을 훨씬 더 간편하게 할 수 있는 사이킷런 API
    - estimator : 사이킷런의 분류 알고리즘 클래스인 Classifier, 회귀 알고리즘 클래스인 Regression
    - scoring : 예측 성능 평가 지표를 기술해줌
    - cv : 교차 검증 폴드 수

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
df_clf = DecisionTreeClassifier(random_state = 156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도, 교차 검증 3개
scores = cross_val_score(dt_clf, data, label, scoring = 'accuracy', cv = 3)
print('교차 검증별 정확도 :', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))

'''
교차 검증별 정확도 : [0.98 0.94 0.98]
평균 검증 정확도: 0.9667
'''
```

cross_val_score()는 cv로 지정된 횟수만큼 scoring 파라미터로 지정된 평가 지표로 평가 결과값을 배열로 반환한다.

### GridSearchCV

- 교차검증과 최적 하이퍼 파라미터 튜닝을 한번에 해준다.

```python
grid_parameters = {'max_depth': [1, 2, 3], 
                   'min_samples_split': [2, 3]}

-> 12 13 22 23 32 33 이렇게 조합한다.
총 6개.
CV값 * 조합수 => 횟수(의 학습/평가)
```

- param_grid : key + 리스트 값을 가지는 딕셔너리가 주어짐. estimator 튜닝을 위한 파라미터명과 사용될 여러 파라미터를 지정
- regit : 디폴트가 True, True로 생성 시 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 파라미터로 재학습 시킴

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state=121)
dtree = DecisionTreeClassifier()

# 파라미터를 딕셔너리 형태로 설정
parameters = {'max_depth': [1, 2, 3], 
                   'min_samples_split': [2, 3]}

import pandas as pd
# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트
# refit=True로 재학습
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv = 3, refit=True)

#붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습, 평가
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과로 추출해 DF로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]

'''
params	mean_test_score	rank_test_score	split0_test_score	split1_test_score	split2_test_score
0	{'max_depth': 1, 'min_samples_split': 2}	0.700000	5	0.700	0.7	0.70
1	{'max_depth': 1, 'min_samples_split': 3}	0.700000	5	0.700	0.7	0.70
2	{'max_depth': 2, 'min_samples_split': 2}	0.958333	3	0.925	1.0	0.95
3	{'max_depth': 2, 'min_samples_split': 3}	0.958333	3	0.925	1.0	0.95
4	{'max_depth': 3, 'min_samples_split': 2}	0.975000	1	0.975	1.0	0.95
5	{'max_depth': 3, 'min_samples_split': 3}	0.975000	1	0.975	1.0	0.95
'''
```

- rank_test_score : 하이퍼 파라미터별로 성능이 좋은 score순위를 나타낸다. 1이 제일 좋음
- mean_test_score : CV의 폴딩 테스트 세트에 대한 평가 평균값

```python
print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도:{0:4f}'.format(grid_dtree.best_score_))

'''
GridSearchCV 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
GridSearchCV 최고 정확도:0.975000
'''
```

```python
# GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습 필요없음
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

'''
테스트 데이터 세트 정확도: 0.9667
'''
```
