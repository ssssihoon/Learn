# Leave-One-Out Cross Validation (LOOCV)

Leave-One-Out 교차검증과 k-fold 교차검증간의 비교.

k-fold의 경우 a, b, c, d, e, f 라는 데이터셋이 있다면 3-fold라면 {a, b}, {c, d}, {e, f} 이렇게 세 개의 데이터로 분할 해 검증을 한다.

그러나 LOOCV의 경우 모든 데이터셋을 순환한다고 생각하면 된다. {a}, {b}, {c}, {d}, {e}, {f} 이렇게 데이터셋의 len만큼을 검증한다. ; len(data)-fold라고 생각하면 됨.
