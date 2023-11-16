# Evaluation

- 머신러닝
    - 데이터 가공/변환
    - 모델 학습/예측
    - 평가
- 분류의 성능 평가 지표
    - 정확도
    - 오차행렬
    - 정밀도
    - 재현율
    - F1 스코어
    - ROC AUC

## 정확도

정확도 = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수

직관적으로 모델 예측 성능을 나타내는 평가 지표

---

- 예제
    - 사이킷런의 BaseEstimator 클래스를 상속
    - 아무런 학습을 하지 않고 성별에 따라 생존자를 예측하는 단순한 Classifier 생성
    - 그러므로 fit() 메서드는 아무것도 수행하지 않는다.
    - predict() 메서드로 Sex피처가 1이면 0, 그렇지 않으면 1로 예측하는 Classifier

```python
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator):
  #fit() 메서드는 아무것도 학습하지 않음.
  def fit(self, X, y=None):
    pass
  # predict() 메서드는 단순히 Sex 피처가 1 이면 0, 그렇지 않으면 1로 예측
  def predict(self, X):
    pred = np.zeros( (X.shape[0], 1))
    for i in range(X.shape[0]):
      if X['Sex'].iloc[i] == 1:
           pred[i] = 0
      else :
        pred[i] = 1

    return pred
```

- MyDummyClassifier을 이용해 앞 장의 타이타닉 생존자 예측
    - 타이타닉 데이터를 추가
    - 데이터를 가공
    - Classifier을 이용해 학습, 예측, 평가 적용

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#원본 데이터를 재로딩, 데이터 가공, 학습 데이터/테스트 데이터 분할
titanic_df = pd.read_csv('/content/drive/MyDrive/train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis = 1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.2, random_state = 0)

#위에서 생성한 Dummy Classifier를 이용해 학습/ 예측 / 평가 수행
myclf = MyDummyClassifier()
myclf.fit(X_train, y_train)

mypredictions = myclf.predict(X_test)
print(accuracy_score(y_test, mypredictions))

'''
0.7877
'''
```

---

### MNIST

- MNIST 데이터 세트를 변환해 불균형한 데이터 세트를 변환해 불균형한 데이터 세트로 만든 뒤
- 정확도 지표 적용 시 어떤 문제가 발생할 수 있는지 확인
- MNIST 데이터 세트는 0~9의 숫자 이미지 픽셀 정보이다.
    - 7을 타겟으로 해 7만 True, 나머지 False → 10%만 True, 90% False

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
  def fit(self, X, y):
    pass

    # 입력값으로 들어오는 X 데이터 세트의 크기만큼 모두 0값으로 만들어서 반환
    def predict(self, X):
      return np.zeros( (len(X), 1), dtype = bool)

# 사이킷런의 내장 데이터 세트인 load_digits()를 이용해 MNIST 데이터 로딩
digits = load_digits()

# digits 번호가 7번이면 True, 이를 astype(int)로 1로 변환, 7번이 아니면 False이고 0으로 변환
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split( digits.data, y, random_state = 11)
```

- y_test데이터 분포도를 확인하고 MyFakesClassifier을 이용해 예측과 평가를 수행

```python
# 불균형한 레이블 데이터 분포도 확인
print("레이블 테스트 세트 크기 : ', y_test.shape)
print("테스트 세트 레이블 0과 1의 분포도")
print(pd.Series(y_test).value_counts())

# Dummy Classifier로 학습/ 예측 / 정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train)
fakepred = fakeclf.predict(X_test)
print('모든 예측을 0으로 하여도 정확도는 : {:.3f}'.format(accuracy_score(y_test, fakepred)))

'''
레이블 테스트 세트 크기 : (450,)
테스트 세트 레이블 0 과 1의 분포도
0    405
1     45
dtype: int64
모든 예측을 0으로 하여도 정확도는:0.900
'''
```

## 오차 행렬
