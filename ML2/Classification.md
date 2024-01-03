# Classification

지도학습의 대표적인 유형인 분류는 기존데이터가 어떤 레이블에 속하는지 패턴을 알고리즘으로 인지한 뒤에 새롭게 관측된 데이터에 대한 레이블을 판별하는 것이다

다양한 머신러닝 알고리즘

- 나이브 베이즈 : 베이즈 통계와 생성 모델에 기반
- 로지스틱 회귀 : 독립변수와 종속변수의 선형 관계성에 기반
- 결정트리 : 데이터 균일도에 따른 규칙 기반의 결정 트리
- 서포트 벡터 머신 : 개별 클래스 간의 최대 분류 마진을 효과적으로 찾아준다
- 신경망 : 심층 연결 기반
- 앙상블 : 서로 다른 or 같은 머신러닝 알고리즘을 결합

## 결정 트리

ML알고리즘 중 직관적으로 이해하기 쉬운 알고리즘이다.

많은 노드(깊음)를 생성한다면 그만큼 복잡해 진다는 얘기고, 그 결과로 과적합이 발생한다.

- 규칙 노드 : 규칙 조건
- 리프 노드 : 결정된 클래스
- 서브 노드 : 균일한 데이터 세트로 분할

균일도가 높은 세트(같은 모집단)를 순서대로 나열하면 된다.

---

### 균일도 측정 by 엔트로피

- 정보 이득 : 엔트로피라는 개념을 기반으로 한다.
    - 엔트로피는 주어진 데이터 집합의 혼잡도를 의미
    - 서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞여 있으면 엔트로피가 낮다
    - 정보 이득 지수 = (1 - 엔트로피 지수)
    - 결정 트리는 이 정보 이득 지수로 분할 기준을 정한다.
    - 정보 이득이 높은 속성을 기준으로 분할
- 지니 계수 : 0이 가장 평등, 1이 가장 불평등
    - 지니 계수가 낮을수록 데이터 균일도가 높은 것으로 해석해 지니 계수가 낮은 속성을 기준으로 분할

### 결정 트리 모델의 특징

- 균일도를 기반
- 전처리 필요 x
- 과적합으로 정확도가 떨어진다.

### 결정 트리 파라미터

- min_samples_split :
    - 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는 데 사용
    - 디폴트 = 2, 작게 설정할수록 분할되는 노드가 많아져서 과적합 가능성 증가
- min_samples_leaf
    - 분할이 될 경우 왼쪽과 오른쪽의 브랜치 노드에서 가져야 할 최소한의 샘플 데이터 수
    - 큰 값으로 설정될수록, 분할된 경우 왼쪽과 오른쪽의 브랜치 노드에서 가져야 할 최소한의 샘플 데이터 수 조건을 만족시키기 어려움
    - 과적합 제어 용도
- max_features
    - 최적의 분할을 위해 고려할 최대 피처 개수, 디폴트는 None로 데이터 세트의 모든 피처를 사용해 분할 수행
    - int형으로 지정하면 대상 피처의 개수, float형으로 지정하면 전체 피처 중 대상 피처의 퍼센트
- max_depth
    - 트리의 최대 깊이를 규정
- max_leaf_nodes
    - 말단 노드의 최대 개수

### 결정 트리 모델의 시각화

- Graphiz 패키지 사용
    - export_graphiz()

---

- 붓꽃 데이터 세트를 이용한 규칙트리

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state = 156)

# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 세트로 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 11)

#DecisionTreeClassifier 학습
dt_clf.fit(X_train, y_train)
```

- Graphviz를 이용해 export_graphviz()함수 사용으로 시각화할 수 있는 출력파일 생성

```python
from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일 생성
export_graphviz(dt_clf, out_file = "tree.dot", class_names = iris_data.target_names, feature_names = iris_data.feature_names, impurity=True, filled = True)
```

- 시각화

```python
import graphviz

with open("tree.dot") as f:
  dot_graph = f.read()
graphviz.Source(dot_graph)
```

사진

- petal length(cm) ≤ 2.45와 같이 피처의 조건이 있는 것은 자식 노드를 만들기 위한 규칙 조건, 조건이 없다면 리프 노드
- gini는 value = []로 주어진 데이터 분포에서의 지니 계수
- samples는 현 규칙에 해당하는 데이터 건수
- value = [] 는 클래스 값 기반의 데이터 건수. ex) Value = [41, 40, 39]라면 왼쪽부터 붓꽃의 클래스가 0 인, 1인, 2인 클래스의 개수.(Setosa, ```)
- 각 노드의 색깔은 붓꽃 데이터의 레이블 값을 의미
- 색깔이 짙어질수록 지니 계수가 낮고 해당 레이블에 속하는 샘플 데이터가 많다는 의미

.

규칙 생성 로직을 미리 제어하지 않으면 완벽하게 클래스 값을 구별하기 위해 트리노드를 계속해서 만들어 간다.

→ 복잡한 규칙트리가 만들어져 과적합 문제가 발생

→ 그러므로 max_depth 하이퍼 파라미터를 변경해 최대 트리 깊이를 제어

min_samples_leaf의 값을 키우게 되면 더 이상 분할되지 않고 리프 노드가 될 수 있는 가능성이 높아진다.

그러면 자연스럽게 브랜치 노드가 줄어들고 결정트리가 더 간결하게 만들어진다.

.

결정 트리는 균일도에 기반해 어떠한 속성을 규칙 조건으로 선택하느냐가 중요한 요건

결정 트리 알고리즘이 학습을 통해 규칙을 정하는 데 있어 피처의 중요한 역할 지표를 feature_importances_ 속성 사용

- feature_importances_ :
    - ndarray 형태로 값을 반환
    - 피처 순서대로 값이 할당
    - 피처가 트리 분할 시 정보 이득이나 지니 계수를 얼마나 효율적으로 잘 개선시켰는지를 정규화된 값으로 표현 한 것.
    - 값이 높을수록 해당 피처의 중요도가 높다

.

- 붓꽃 데이터 세트에서 피처별로 결정 트리 알고리즘에서 중요도를 추출
    - fit()으로 학습된 DecisionTreeClassifier 객체 변수인 df_clf에서 feature_importances_ 속성을 가져와 피처별로 중요도 값을 매핑하고 이를 막대그래프로 표현

```python
import seaborn as sns
import numpy as np
%matplotlib inline

# feature importance 추출
print(np.round(dt_clf.feature_importances_, 3), "\n")

# feature별 importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
  print('{0} : {1:3f}'.format(name, value))

# feature importance를 column 별로 시각화
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)

'''
[0.025 0.    0.555 0.42 ] 

sepal length (cm) : 0.025005
sepal width (cm) : 0.000000
petal length (cm) : 0.554903
petal width (cm) : 0.420092
'''
```

피처들 중 petal_length가 가장 피처 중요도가 높음을 알 수 있다.

### 결정 트리 과적합(Overfitting)

- 시각화를 통해 과적합을 알아보기

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
%matplotlib inline

plt.title("3 Class values with 2 Features Sample data creation")

# 2차원 시각화를 위해서 피처2개, 클래스 3가지 유형의 분ㄹ 샘플 데이터 생성
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state = 0)

# 그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색깔로 표기
plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', c=y_labels, s=25, edgecolor='k')
```

사진

- visualize_boundary() : 어떠한 결정 기준을 가지고 분할하면서 데이터를 분류하는지 확인 가능한 함수

```python
from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약 없는 결정 트리의 학습과 결정 경계 시각화
dt_clf = DecisionTreeClassifier(random_state=156).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)
```

사진

일부 이상치 데이터까지 분류하기 위해 분할이 자주 일어나서 결정 기준 경계가 매우 많아졌다.

이렇게 복잡한 모델은 예측 정확도가 떨어지게 된다.

- min_samples_leaf를 이용해 제한

```python
# min_sample_leaf=6 로 트리 생성 조건을 제약한 결정 경계 시각화
dt_clf = DecisionTreeClassifier(min_samples_leaf=6, random_state=156).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labes)
```

사진

## 앙상블 학습

여러 개의 분류기를 생성하고 그 예측을 결합함으로써 정확한 최종 예측을 도출하는 기법

- 종류
    - 랜덤 포레스트
    - 그래디언트 부스팅 알고리즘
- 앙상블 학습의 유형
    - 보팅
    - 배깅
    - 부스팅

### 보팅 유형

- Hard Voting : 다수의 분류기들 간 다수결로 결과값을 선정
- Soft Voting : 다수의 분류기들의 결과값을 평균내어 결과값을 선정, 주로 사용

### 보팅 분류기

- 우선 로지스틱 회귀와 KNN 기반 분류기를 만들기

```python
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head(3)
```

- 소프트 보팅 방식으로 새롭게 보팅 분류기 만들기
    - VotingClassifier 클래스
        - estimators : 리스트 값으로 보팅에 사용될 여러 개의 Classifier 객체들을 튜플 형식으로 입력 받음
        - voting : hard or soft

```python
# 개별 모델은 로지스틱 회귀, KNN
lr_clf = LogisticRegression(solver='liblinear')
knn_clf = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기
vo_clf = VotingClassifier( estimators=[('LR', lr_clf), ('KNN', knn_clf)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

# VotingClassifier 학습/예측/평가
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print("Voting 분류기 정확도 : {0:4f}".format(accuracy_score(y_test, pred)))

# 개별 모델의 학습/예측/평가
classifiers = [lr_clf, knn_clf]
for classifier in classifiers:
  classifier.fit(X_train, y_train)
  pred = classifier.predict(X_test)
  class_name = classifier.__class__.__name__
  print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))

'''
Voting 분류기 정확도 : 0.956140
LogisticRegression 정확도: 0.9474
KNeighborsClassifier 정확도: 0.9386
'''
```

## 랜덤 포레스트

배깅은 같은 알고리즘으로 여러 개의 분류기를 만들어서 보팅으로 최종 결정하는 알고리즘

배깅의 대표적인 알고리즘

여러 개의 결정 트리 분류기가 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정

- 부트스트래핑 분할 방식 : 여러 개의 데이터 세트를 중첩되게 분리하는 것
- 랜덤 포레스트의 서브세트 데이터는 부트스트래핑으로 데이터가 임의로 만들어진다.
    - n_estimators=3 이라면 부트스트래핑으로 3개의 서브세트로 분할해준다.
- RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 결정 트리에서 사용한 get_human_dataset()을 이용해 학습/테스트용 DataFrame 반환
X_train, X_test, y_train, y_test = get_human_dataset()

# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0, max_depth=8)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))

'''
랜덤 포레스트 정확도: 0.9196
'''
```

### 랜덤 포레스트 하이퍼 파라미터 및 튜닝

트리 기반의 앙상블 알고리즘의 단점은 하이퍼 파라미터가 너무 많고 그로 인해 튜닝을 위한 시간이 많이 소모된다는 것이다.

- n_estimators : 랜덤 포레스트의 결정 트리 개수를 지정
- max_features : 파라미터

랜덤 포레스트의 파라미터를 튜닝

앞의 사용자 행동 데이터 세트를 그대로 이용

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[8, 16, 24],
    'min_samples_leaf' : [1, 6, 12], 
    'min_samples_split' : [2, 8, 16]
}

rf_clf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=2, n_jobs=1 )
grid_cv.fit(x_train , y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
```
