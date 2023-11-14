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

```
