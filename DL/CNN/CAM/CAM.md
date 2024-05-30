# CAM

## 인공지능 모델의 설명 가능성을 확인하기 위해 CAM 기법을 사용하여 InputData의 weight 형태를 확인하자.

### CAM(Class Activation maps)이란?

CNN이 특정 클래스를 식별하는 데 사용하는 식별 이미지 영역을 얻는 기술

- CAM의 배경
    - 마지막 Conv층을 FC층으로 Flatten할 때 피처맵의 정보를 잃게 된다. → 어떻게 분류를 했는 지 알 수 없음. → CAM을 통해 정보들을 얻어 낼 수 있게 됐다.
- 왜 사용하는가?
    - 이미지의 어느 영역이 이 클래스와 관련이 있는지 확인이 가능

### CAM의 중요 알고리즘

[https://velog.io/@albert0811/DL-Class-Activation-MapCAM-GradCAM](https://velog.io/@albert0811/DL-Class-Activation-MapCAM-GradCAM)

- CNN & CAM 비교
    - CNN : Conv연산 후 Flatten과정으로 벡터화를 한다.
    - CAM : Flatten과정을 거치지 않고 GAP(Global Average Pooling) 구조를 사용한다.
- GAP : 피처맵의 가중치값들의 평균

![Untitled](CAM%2090e47b03065346aab2cfd8fd0d50bbe6/Untitled.png)

GAP값(a,b,c) 으로 w1, w2, w3 학습 → 피처맵에 곱해줌 → 전체 sum
