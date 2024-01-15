


>#### **합성곱 신경망은 크게 합성곱층과(Convolution layer)와 풀링층(Pooling layer)으로 구성**
</br>
Convolution Layer : 특징을 추출하는 층
</br>
Pooling Layer : 정보를 압축하는 층

![](https://velog.velcdn.com/images/ssssihoon/post/f6150dd3-ead2-437c-99c1-bcd10032536d/image.png)

- 앞단 : Conv, ReLU, Pool(선택) : 특징 추출

- 뒷단 : FC(Fully Connected), SM(SoftMax) : 추출된 특징을 사용해 분류 또는 회귀를 수행하는 다층 퍼셉트론 부분




# Convolution (합성곱)


>## Matrix Convolution
![](https://velog.velcdn.com/images/ssssihoon/post/0753aec3-0d9e-4f49-b869-6a324beb5151/image.png)

>## Tensor Convolution
3차원 텐서 합성곱의 경우 입력데이터와 필터를 곱해서 나온 값들을 모두 더해 피처맵에 추가하는 방식이다.

>### Padding (패딩)
가장자리를 0으로 픽셀을 추가하는 것. (Zero- Padding)
일반적인 경우, 정보가 너무 축소되는데 (4x4 \* 3x3 -> 2x2) 이를 방지 해준다. (5x5 \* 3x3 -> 4x4) </br>
; 가장자리의 픽셀 정보까지 잘 이용할 수 있으며, 아웃풋을 이미지를 좀 더 유지할 수 있다.
![](https://velog.velcdn.com/images/ssssihoon/post/eb0d493f-7164-447f-b3f3-6a2f4f576e7e/image.png)




# Pooling (풀링)
- Max pooling
- Average Pooling

![](https://velog.velcdn.com/images/ssssihoon/post/0f7e2234-6f5d-4eae-a613-1c22eb1f0c34/image.png)

### Pooling 하는 이유
1. parameter를 줄이기 때문에, 해당 network의 표현력이 줄어들어 Overfitting을 억제

2. Parameter를 줄이므로, 그만큼 비례하여 computation이 줄어들어 hardware resource(energy)를 절약하고 speedup

[출처](https://technical-support.tistory.com/65)
