# Face_Recognition

얼굴 탐지와 얼굴 인식은 다르다.

얼굴 탐지는 얼굴임을 판별하는 것이고 얼굴 인식은 인물이 누구인지 파악하는 것이 목표이다.

1. Face recognition with LBPH and OpenCV : 얼굴 인식에 있어서 가장 유명한 알고리즘, 지역 이진 패턴 히스토그램
2. Face recognition with Dlib, CNN and distance calculation
3. Face recognition using the webcam

## LBPH (Local Binary Patterns Histograms)

픽셀 행렬의 중앙값을 기준으로 다시 값을 재조정한다.

```python
12 15 18
5  8. 3
8. 1. 2

이렇게 있다면 기준값 이상이면 1, 아니면 0으로 재조정
if >= 8 : 1
if <  8 : 0

1 1 1
0 8 0
1 0 0
이렇게 재조정 된다.

좌측 상단을 기준으로 시계방향으로 2진수화를 해준다. (기준값 제외)
Binary -> 11100010
Decimal(10진수) -> 226
; 중앙값을 둘러싸고 있는 픽셀이 226임을 뜻한다.

이미지에 빛을 비추면 
픽셀의 값이 올라간다. -> 흰색 (255, 255, 255)
그래도 이진수의 값을 유지된다. -> 11100010 , 어둡게 해도 동일

 이를 히스토그램을 통해 비교
```

## Load Face Dataset

```python
# 파이썬에서 이미지를 작업할 수 있도록 해줌
from PIL import Image
import cv2
import numpy as np

# zip형식의 압축을 해제할 필요가 있기 때문에 사용
import zipfile
path = '/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Datasets/yalefaces.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r') # read
zip_object.extractall('./') # 앞으로 모든 파일을 이 루트 디렉토리에 저장한다는 뜻
zip_object.close()
```

## Pre-processing the images

```python
import os

print(os.listdir('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/yalefaces/train'))

#train폴더에 속한 파일명을 모두 볼 수 있다.
```