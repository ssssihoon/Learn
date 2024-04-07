# week5

## 주성분 분석

- 고유값과 고유벡터 계산 : `eigenvalues, eigenvectors = np.linalg.eig(A)`
- pca

```python
import numpy as np

def pca(X, n_components):
    # 원본 데이터 출력
    print("원본 데이터:")
    print(X)

    # 공분산 행렬 계산
    cov_matrix = np.cov(X.T)
    print("\n공분산 행렬:")
    print(cov_matrix)

    # 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("\n고유값:")
    print(eigenvalues)
    print("\n고유벡터:")
    print(eigenvectors)

    # 고유값을 내림차순으로 정렬하고, 해당하는 고유벡터 선택
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    principal_components = eigenvectors[:, idx]

    return principal_components

# 테스트용 데이터 생성 (10x5 크기의 정수 데이터)
X = np.random.randint(1, 10, size=(10, 5))
n_components = 2  # 주성분 개수 지정

# PCA 수행
principal_components = pca(X, n_components)
print("\n주성분:")
print(principal_components)

```

- 주성분 개수 설정 : pca = PCA(n_components=50) ; 50개의 주성분을 사용
- pca.components_: PCA 모델에서 추출된 주성분(고유벡터)을 나타내는 속성입니다.
- pca.transform(데이터.shape) : 데이터의 차원으로 데이터를 변환한다.
