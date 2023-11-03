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
