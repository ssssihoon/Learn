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

## LBPH (Local Binary Patterns Histograms)

- 저장된 히스토그램과 분류하고자 하는 새 이미지의 히스토그램을 비교할 때 사용

```python
pip install opencv-python
pip install opencv-contrib-python

import cv2

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml') # 분류기를 저장할 때 기본형식 yml

-> lbph 분류기파일 생성됨.

```

## Recognizing faces

```python
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('/Users/sihoon/Desktop/ComputerVision/Face_Recognition/lbph_classifier.yml')
```

```python
test_image = '/Users/sihoon/Desktop/ComputerVision/Face_Recognition/yalefaces/test/subject10.sad.gif'
```

```python
image = Image.open(test_image).convert('L') # 흑백이미지로 변환
image_np = np.array(image, 'uint8')
```

```python
image_np.shape

'''
(243, 320) -> 흑백 이미지 픽셀
'''
```

```python
prediction = lbph_face_classifier.predict(image_np)
prediction

'''
(10, 6.384336446373091)
'''
10 -> 클래스 분류 (맞음)
6.384336446373091 -> 신뢰도. 높을수록 성능이 좋음
```

```python
expected_output = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
expected_output

'''
10
'''
```

```python
cv2.putText(image_np, 'PredL: ' + str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(image_np, 'EXP: ' + str(expected_output), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

# 10, 30 은 글자의 위치를 나타낸다.

cv2.imshow('Image', image_np)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Recognition%204205d5fc40494d46aef1a59681d148a3/Untitled.png)

## Evaluating the face classifier

```python
# 이미지 전처리
paths = [os.path.join('/Users/sihoon/Desktop/ComputerVision/Face_Recognition/yalefaces/test', f) for f in os.listdir('/Users/sihoon/Desktop/ComputerVision/Face_Recognition/yalefaces/test')]
predictions = []
expected_outputs = []
for path in paths:
    #print(path)
    image = Image.open(path).convert('L')
    image_np = np.array(image, 'uint8')
    prediction, _ = lbph_face_classifier.predict(image_np)
    expected_output = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

    predictions.append(prediction)
    expected_outputs.append(expected_output)
```

```python
predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(expected_outputs, predictions)
'''
0.6666666666
'''
```

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_outputs, predictions)
cm
```

```python
# 오차행렬 시각화
import seaborn as sns

sns.heatmap(cm, annot=True);
```

![Untitled](Face_Recognition%204205d5fc40494d46aef1a59681d148a3/Untitled%201.png)

## LBPH Parameter

1. Radius(반지름)
    1. 높아질수록 더 많은 패턴을 찾을 수 있음. 그러나 모서리를 찾기 힘듦
2. Neighbors(이웃 수)
    1. 반지름이 1인 경우 이웃은 8, 사용되는 픽셀의 개수를 나타냄.
3. grid_x and grid_y(가로, 세로 축의 셀 수)
    1. 정사각형 마다 히스토그램이 있음. 셀이 더 많을수록 히스토그램의 수도 많아져 이미지에서 더 많은 패턴을 찾아낼 수 있음.
4. Threshold(임계값) ; 감지의 신뢰도
    1. 값이 높을수록 얼굴 인식의 품질이 높아짐.

## Detecting facial points
