# Pandas

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
