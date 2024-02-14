# Face_Detection

1. Face detection with Haarcascade and OpenCV
2. Face detection with HOG and Dlib
3. Face detection with CNN and Dlib
4. Face detection using webcam

# Images, Pixels

32 x 32 = 1024 pixels     → 흑백 ; 흑백의 경우 R, G, B가 동일한 값이기 때문이다.

32 x 32 x 3(R, G, B) = 3072    → 컬러

# Cascade Classifier

- 얼굴감지, 객체감지의 기초적인 방법
- 흰 픽셀의 합 - 검은 픽셀의 합
- 왼쪽에서 오른쪽, 위에서 아래
- 1번 특성이 이미지상에 존재하지 않으면 F 작업 종료, 아무런 감지 없음
- 특성을 찾아 T 반복