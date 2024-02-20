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

![Untitled](https://github.com/ssssihoon/Learn/assets/127017020/93184248-ed09-4ad1-af80-54005dae4fa7)


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

![Untitled 1](https://github.com/ssssihoon/Learn/assets/127017020/7154154d-55d6-4622-a01c-c0a69d588a51)


→ 긍정오류 (1)

## Haarcascade Parameter1

- 긍정오류를 보완
- `scaleFactor=1.1 (기본값)`

`detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.09)`

→ scaleFactor을 조절해 이미지를 확대, 축소한 이미지를 감지한다.

![Untitled 2](https://github.com/ssssihoon/Learn/assets/127017020/5eb54752-35dc-4b61-9f57-3152df06dc2a)


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

![Untitled 3](https://github.com/ssssihoon/Learn/assets/127017020/f169380d-9ad7-4119-ad01-a1a86f4fdf89)


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

![Untitled 4](https://github.com/ssssihoon/Learn/assets/127017020/1a9c1dc4-b724-4851-8325-946034826d60)


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

![Untitled 5](https://github.com/ssssihoon/Learn/assets/127017020/6cc490b1-dda4-402d-bac5-29e21f464eb3)


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

![Untitled 6](https://github.com/ssssihoon/Learn/assets/127017020/fd548965-ab4f-43e2-9220-b97294686099)


## Detecting faces with CNN

```python
image = cv2.imread('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Images/people2.jpg')

# cnn 감지기 사용
cnn_detector = dlib.cnn_face_detection_model_v1('/Users/sihoon/Desktop/ComputerVision/Computer Vision Masterclass/Weights/mmod_human_face_detector.dat')
```

```python
detections = cnn_detector(image, 1)
for face in detections:
    l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    print(c)
    cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)  # 아무 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기

'''
1.144068717956543
1.137050986289978
1.127898931503296
1.1200220584869385
1.1149381399154663
1.1131561994552612
1.0975712537765503
1.094212293624878
1.0853136777877808
1.0801897048950195
1.0800751447677612
1.0784766674041748
1.066402554512024
1.06417977809906

-> 신뢰도. 높을수록 좋음
'''
```

![Untitled 7](https://github.com/ssssihoon/Learn/assets/127017020/2a0ef7e3-5415-40c8-8c05-691c0e7bef17)


## Face detection in Webcam

```python
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
```
