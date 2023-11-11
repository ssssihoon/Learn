# Sklearn

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

### 데이터 인코딩

- 레이블 인코딩
    - 피처를 숫자 값으로 변환하는 것
    - LabelEncoder를 객체로 생성
    - fit()
    - encoder.transform()

```python
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder를 객체로 생성한 후, fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값:', labels)

'''
인코딩 변환값: [0 1 4 5 3 3 2 2]
'''
```

다시 확인하는 방법

```python
print(encoder.classes_)

'''
['TV' '냉장고' '믹서' '선풍기' '전자레인지' '컴퓨터']
'''
```

```python
print(encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3])

'''
['TV' '냉장고' '믹서' '선풍기' '전자레인지' '컴퓨터']
'''

```

레이블 인코딩은 숫자값에따라 ML알고리즘의 가중치 적용 때문에 문제가 발생할 수 있다. 이를 해결하기 위한 원-핫 인코딩

- 원핫 인코딩
    - 피처에 따른 고유 값을 1로 적용, 아닐 시 0

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# 2차원 ndarray로 변환
items = np.array(items).reshape(-1, 1)

# 원핫 인코딩 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)

# OneHotEncoder로 변환한 결과는 희소행렬이므로 toarray()를 이용해 밀집 행렬로 변환
print('원 핫 인코딩 데이터')
print(oh_labels.toarray())
print('원 핫 인코딩 데이터 차원')
print(oh_labels.shape)

'''
원 핫 인코딩 데이터
[[1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]]
원 핫 인코딩 데이터 차원
(8, 6)
'''
```

- get_dummies()를 이용해 원핫인코딩을 빠르게

```python
import pandas as pd

df = pd.DataFrame({'items' : ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
pd.get_dummies(df)

'''
items_TV	items_냉장고	items_믹서	items_선풍기	items_전자레인지	items_컴퓨터
0	1	0	0	0	0	0
1	0	1	0	0	0	0
2	0	0	0	0	1	0
3	0	0	0	0	0	1
4	0	0	0	1	0	0
5	0	0	0	1	0	0
6	0	0	1	0	0	0
7	0	0	1	0	0	0
'''
```

### 피처 스케일링(정규화, 표준화)

- 스케일링
    - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
- 표준화
    - 피처의 평균을 0, 표준편차를 1로 만든다.
- 정규화
    - 피처 값을 [0, 1]로 조정 최소 0 최대 1

---

- StandardScaler
- 표준화를 쉽게 지원하기 위한 클래스
    - 평균→0, 분산→1 로 표현

```python
from sklearn.datasets import load_iris
import pandas as pd

#붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환한다.
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)

print('feature 들의 평균')
print(iris_df.mean())
print('\nfeature 들의 분산')
print(iris_df.var())

'''
feature 들의 평균
sepal length (cm)    5.843333
sepal width (cm)     3.057333
petal length (cm)    3.758000
petal width (cm)     1.199333
dtype: float64

feature 들의 분산
sepal length (cm)    0.685694
sepal width (cm)     0.189979
petal length (cm)    3.116278
petal width (cm)     0.581006
dtype: float64
'''
```

각 피처를 표준화해 변환

```python
from sklearn.preprocessing import StandardScaler

# StandardScaler객체 생성
scaler = StandardScaler()
# StrandardScaler로 데이터 세트 변환 fit과 transform()호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform()시 스케일 변환된 데이터 세트가 NumPy, ndarray로 변환돼 이를 DF로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 들의 평균')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산')
print(iris_df_scaled.var())
```

- MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환. fit(), transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform()시 스케일 변환된 데이터 세트가 NumPy, ndarray로 변환돼 이를 DF로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature 들의 최댓값')
print(iris_df_scaled.max())

'''
feature 들의 최솟값
sepal length (cm)    0.0
sepal width (cm)     0.0
petal length (cm)    0.0
petal width (cm)     0.0
dtype: float64

feature 들의 최댓값
sepal length (cm)    1.0
sepal width (cm)     1.0
petal length (cm)    1.0
petal width (cm)     1.0
dtype: float64
'''
```
