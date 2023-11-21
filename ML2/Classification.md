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
