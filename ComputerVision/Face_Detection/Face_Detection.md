# Face_Detection

1. Face detection with Haarcascade and OpenCV
2. Face detection with HOG and Dlib
3. Face detection with CNN and Dlib
4. Face detection using webcam

## Images, Pixels

32 x 32 = 1024 pixels     → 흑백 ; 흑백의 경우 R, G, B가 동일한 값이기 때문이다.

32 x 32 x 3(R, G, B) = 3072    → 컬러

## Cascade Classifier

- 얼굴감지, 객체감지의 기초적인 방법
- 흰 픽셀의 합 - 검은 픽셀의 합
- 왼쪽에서 오른쪽, 위에서 아래
- 1번 특성이 이미지상에 존재하지 않으면 F 작업 종료, 아무런 감지 없음
- 특성을 찾아 T 반복

## **Image loading, preprocessing**

OpenCV 라이브러리 : `import cv2`

이미지 가져오기 : `변수명 = cv2.imread(’경로’)`

이미지 크기 : `변수명.shape()`

- 이미지 보는법

```python
cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

이미지 크기 재조정 : `변수명 = cv2.resize(image, (800, 600)) *# 800x600*`

컬러 → 흑백 : `변수명_흑백 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled.png)

## Use Haarcascade and OpenCV to Face Detection

*# 얼굴 탐지기(분류기) 설정*

`face_detector = cv2.CascadeClassifier('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Cascades/haarcascade_frontalface_default.xml')`

*# 모든 감지값을 저장하는 변수 설정*

`detections = face_detector.detectMultiScale(image_gray)`

detections

‘’’

387 233 73 73
92 239 66 66
115 124 53 53
475 123 59 59
677 72 68 68
390 323 56 56

‘’’

*# 열 row 의 개수 : 감지된 얼굴의 수*

*# 컬럼 idx 첫 번째와 두 번째는 각각 x축, y축을 뜻함*

*# 마지막 두 개는 얼굴의 크기(너비 width, 높이 height)를 나타 낸다. 73x73*

*# 즉 얼굴의 위치를 나타냄.*

```python
for (x, y, w, h) in detections:
    print(x, y, w, h)
    # 1 : 이미지, 2 : 직사각형이 시작되는 지점 3 : 선의 색깔, 4 : 선의 굵기
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%201.png)

→ 긍정오류 (1)

## Haarcascade Parameter1

- 긍정오류를 보완
- `scaleFactor=1.1 (기본값)`

`detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.09)`

→ scaleFactor을 조절해 이미지를 확대, 축소한 이미지를 감지한다.

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%202.png)

얼굴이 클 수록 매개변수를 더 크게 조절해야한다.

## Haarcascade Parameter2

scaleFactor로 보완이 안될 때

- `minNeighbors` : 얼굴을 감지하는 알고리즘은 주어진 이미지에서 각 위치에서 얼굴이 있을 가능성이 있는 사각형을 생성합니다. 이러한 사각형은 후보 사각형이라고도 합니다. 그런 다음 **minNeighbors** 매개변수는 이러한 후보 사각형을 최종적으로 얼굴로 감지할지 여부를 결정하는 데 사용됩니다.
    - 높아질수록 감지성능도 높아짐

```python
image = cv2.imread('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Images/people2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=7)
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%203.png)

두 명의 얼굴을 더 감지해야함.

- `minSize=(10,10)` : 너비10, 높이10 사이즈의 감지해야 하는 최소경계사이즈이다. 기본값은 30x30
- `maxSize=(100,100)` : 너비100, 높이100 사이즈의 감지해야 하는 최대경계사이즈이다.

## Use Haarcascade to Eye detection

```python
eye_detector = cv2.CascadeClassifier('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Cascades/haarcascade_eye.xml')
```

```python
image = cv2.imread('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Images/people1.jpg')
image = cv2.resize(image, (800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.09)
for (x, y, w, h) in face_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

eye_detections = eye_detector.detectMultiScale(image_gray)
for (x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%204.png)

하나의 긍정오류

→ scaleFactor, minNeighbors 설정 , 크기는 원래대로 재조정

```python
image = cv2.imread('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Images/people1.jpg')
#image = cv2.resize(image, (800, 600)) # 소요시간을 줄이기 위해 크기 축소
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.3, minSize=(30, 30))
for (x, y, w, h) in face_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, maxSize=(70, 70))
for (x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%205.png)

## HOG - Histograms of Oriented Gradients

- cascade보다 더 나은 결과를 산출한다.
- Derivative allows to measure the rate of change
- Gradient vector

## Detecting faces with HOG

- 흑백 사진이 필요없음.
- 분류기 파일이 필요없음.
- 더 복잡한 계산을 하기 때문임
- `face_detector_hog = dlib.get_frontal_face_detector()`
- `detections = face_detector_hog(image, 1)`

```python
face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 1) # 1의 의미는 scaleFactor(이미지의 크기)와 같은 파라미터라고 보면 된다. -> 높은 값일수록 작은 경계 박스
detections

'''
rectangles[[(429, 38) (465, 74)], [(665, 90) (701, 126)], [(717, 103) (760, 146)], [(909, 70) (952, 113)], [(828, 98) (871, 142)], [(605, 70) (641, 106)], [(777, 62) (813, 98)], [(485, 78) (521, 114)], [(386, 60) (429, 103)], [(170, 41) (213, 84)], [(93, 89) (136, 132)], [(237, 50) (280, 94)], [(323, 50) (367, 94)], [(544, 65) (588, 108)]]
'''
```

```python
len(detections)

'''
14
'''
-> 모든 사람을 감지해냄
```

```python
for face in detections:
    # print(face.left())
    # print(face.top())
    # print(face.right())
    # print(face.bottom())
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
```

![Untitled](Face_Detection%20cd3a9aac109248159d2bedfe6b7bdc5b/Untitled%206.png)

## Detecting faces with CNN

## Haarcascade x HOG x CNN
