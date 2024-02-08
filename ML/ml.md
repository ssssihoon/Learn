# 머신러닝

분석 프로세스

1. 입력 데이터 구조 만들기 → 데이터 프레임 1. 피처, 2.레이블
2. 전체 데이터 → train / test (train_test_split)
3. feature와 label 분리 : train_X, train_Y, train_X, train_Y
4. KNN model → KNeighborsClassifier
5. 머신러닝 모델 생성 → model = KNeighborsClassifier()
6. 생성된 모델을 학습 → model.fit(train_X, train_Y)
7. 학습 모델 테스트 → pred_knn = model.predict(test_X)
8. 학습 모델 정확도 산출 → metrics.accuracy_score(pred_knn, test_Y)

- KNN 알고리즘 : 데이터로부터 거리가 가까운 데이터 k개의 레이블을 참조하여 많은 수에 해당하는 클래스로 분류
- SVM 알고리즘 : 두 클래스를 구분하는 가상의 결정경계면을 계싼하여 클래스를 분류
- 의사결정 트리 : 스무고개를 하듯이 데이터를 나무형태로 분류

와인 경작자 분류

```python
from sklearn.datasets import load_wine
wine = load_wine() # 와인데이터 불러오기
print(wine.DESCR) # 무엇이 있는지 확인
```

와인데이터를 데이터프레임형식으로 만들기

그 중 df에 레이블 추가

```python
import pandas as pd
import numpy as np

wine_feature = wine.data
wine_label = wine.target

df_wine = pd.DataFrame(data=wine_feature, columns=[wine.feature_names])
df_wine['label'] = wine_label
df_wine
```

레이블을 정수형으로 변환

데이터셋을 학습과 테스트로 분할

테스트 데이터는 학습데이터에서 사용하지 않은 것으로 구성

stratify 변수는 학습데이터와 테스트 데이터에 각 레이블이 적절한 비율로 포함되도록 데이터 구성

train과 test데이터셋을 다시 특성과 레이블로 분리

```python
from sklearn.model_selection import train_test_split

df_wine = df_wine.astype({'label' : 'int'})
train, test = train_test_split(df_wine, test_size = 0.3, random_state = 0, stratify = df_wine['label'])

train_X = train[train.columns[:13]]
train_Y = train[train.columns[13:]]

test_X = test[test.columns[:13]]
test_Y = test[test.columns[13:]]
```

KNN분류

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 학습
model = KNeighborsClassifier()
model.fit(train_X, train_Y)

#테스트, 평가
pred_knn = model.predict(test_X)
print('KNN알고리즘 분류 정확도 :', metrics.accuracy_score(pred_knn, test_Y))
```