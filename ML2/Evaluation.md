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

- 이진 분류의 예측 오류가 얼마인지, 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표
    - TN FP FN TP
        - TN 예측값 Negative(0) → 실제값 Negative(0)
        - FP 예측값 Positive(1) → 실제값 Negative(0)
        - FN 예측값 Negative(0) → 실제값 Positive(1)
        - TP 예측값 Positive(1) → 실제값 Positive(1)

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, fakepred)

'''
array([[405,   0],
       [ 45,   0]])
'''
# [0, 0]의 경우, 예측으로 7이 아닌 Digit이. 실제 7이 아닌 Digit
# [0, 1]의 경우, 예측으로 7이 아닌 Digitdl. 실제 7인 Digit
```

- 이 값을 조합 해 정확도, 정밀도, 재현율을 알 수 있다.
- 정확도 = (TN+TP)/(TN + FP + FN + TP)

## 정밀도와 재현율

- 정밀도 = TP / (FP + TP)
    - 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
    - precision_score()
- 재현율 = TP / (FN + TP)
    - 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
    - 민감도 or TPR
    - recall_score()
- matirx, accuracy, precision, recall등의 평가를 한꺼번에 호출하는 get_clf_eval() 함수

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def get_clf_eval(y_test, pred):
  confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  print('오차행렬')
  print(confusion)
  print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:4f}'.format(accuracy, precision, recall))
```

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, \
                                                    test_size=0.20, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test , pred)

'''
오차 행렬
[[104  14]
 [ 13  48]]
정확도: 0.8492, 정밀도: 0.7742, 재현율: 0.7869
'''
```

### 정밀도 / 재현율 트레이드 오프(상충관계)

정밀도와 재현율은 서로 상충관계이다.

- Positive 예측값이 많아지면 상대적으로 재현율 값이 높아진다.

---

- 정밀도가 100%가 되는 법
    - 확실한 기준이 되는 경우만 Positive, 나머지를 Negative
- 재현율이 100%가 되는 법
    - 모든 경우를 Positive

## F1스코어

- 정밀도와 재현율을 결합한 지표
    - f1_score()

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
print('F1 스코어 : {0:.4f}'.format(f1))

'''
F1 스코어 : 0.7966
'''
```

타이타닉 생존자 예측의 임곗값을 변화시키면서 F1 스코어를 포함한 평가 지표를 구하기

```python
def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    # F1 스코어 추가
    f1 = f1_score(y_test,pred)
    print('오차 행렬')
    print(confusion)
    # f1 score print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))

thresholds = [0.4 , 0.45 , 0.50 , 0.55 , 0.60]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)

'''
임곗값: 0.4
오차 행렬
[[99 19]
 [10 51]]
정확도: 0.8380, 정밀도: 0.7286, 재현율: 0.8361, F1:0.7786
임곗값: 0.45
오차 행렬
[[103  15]
 [ 12  49]]
정확도: 0.8492, 정밀도: 0.7656, 재현율: 0.8033, F1:0.7840
임곗값: 0.5
오차 행렬
[[104  14]
 [ 13  48]]
정확도: 0.8492, 정밀도: 0.7742, 재현율: 0.7869, F1:0.7805
임곗값: 0.55
오차 행렬
[[109   9]
 [ 15  46]]
정확도: 0.8659, 정밀도: 0.8364, 재현율: 0.7541, F1:0.7931
임곗값: 0.6
오차 행렬
[[112   6]
 [ 16  45]]
정확도: 0.8771, 정밀도: 0.8824, 재현율: 0.7377, F1:0.8036
'''
```

⇒임곗값이 0.6일 때가 가장 좋은 F1스코어

그러나 재현율이 크게 감소하고 있으니 주의

## ROC곡선과 AUC

- 이진 분류의 예측 성능 측정에서 중요하게 사용되는 지표
- ROC 곡선
    - roc_curve()
    - 수신자 판단 곡선 : FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선.
        - TPR : 재현율
        - TNR : 특이성
    - ROC 곡선이 가운데 직선에 가까울수록 성능이 떨어지는 것이며, 멀어질수록 성능은 뛰어나다.

## 피마 인디언 당뇨병 예측

데이터 세트

- Pregnancies : 임신 횟수
- Glucose : 포도당 부하 검사 수치
- BloodPressure : 혈압
- SkinThickness : 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
- Insulin : 혈청 인슐린(mu U/ml)
- BMI : 체질량지수
- DiabetesPedigreeFunction : 당뇨 내력 가중치 값
- Age
- Outcome : 클래스 결정 값

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

diabetes_data = pd.read_csv('/content/drive/MyDrive/diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
diabetes_data.head(3)

'''
0    500
1    268
Name: Outcome, dtype: int64
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
'''
```

```python
diabetes_data.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
'''
```

→ Null은 없고, 타입은 숫자형이다.

---

- 로지스틱 회귀를 이용한 예측 모델 생성
    - 데이터 세트를 피처 데이터 세트와 클래스 데이터 세트로 나누고
    - 학습 데이터 세트와 테스트 데이터 세트로 분리
    - 로지스틱 회귀를 이용해 예측을 수행하고
    - 성능평가 지표를 출력 (get_clf_eval, get_eval_by_threshold, precision_recall_curve_plot)
    - 재현율 곡선을 시각화
