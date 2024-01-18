

# Collider

콜라이더는 충돌 감지를 위한 컴포넌트이다.

- 박스 콜라이더
- 스피어 콜라이더
- 캡슐 콜라이더
- 메쉬 콜라이더

![](https://velog.velcdn.com/images/ssssihoon/post/456d1e1a-87d4-4e69-a4d5-a30ae6787244/image.png)


## Box Collider

| 이름 | 설명 |
| --- | --- |
| Is Trigger | 트리거 사용 여부 (체크 시 물리적인 충돌이 발생하지 않음) |
| Material | 콜라이더에 사용되는 재질을 나타냄 |
| Center | 콜라이더 중심 (로컬 기준) |
| Size | 콜라이더 크기 (로컬 기준) |

## **Sphere Collider**

| 이름 | 설명 |
| --- | --- |
| Is Trigger | 트리거 사용 여부 (체크 시 물리적인 충돌이 발생하지 않음) |
| Material | 콜라이더에 사용되는 재질을 나타냄 |
| Center | 콜라이더 중심 (로컬 기준) |
| Radius | 콜라이더 반지름 (로컬 기준) |

## **Capsule Collider**

| 이름 | 설명 |
| --- | --- |
| Is Trigger | 트리거 사용 여부 (체크 시 물리적인 충돌이 발생하지 않음) |
| Material | 콜라이더에 사용되는 재질을 나타냄 |
| Center | 콜라이더 중심 (로컬 기준) |
| Radius | 콜라이더 반지름 (로컬 기준) |
| Height | 캡슐의 높이 |
| Direction | 캡슐의 방향 |

## **Mesh Collider**

| 이름 | 설명 |
| --- | --- |
| Convex | 충돌 감지 메쉬 오브젝트 사용 여부 (체크 시 충돌 감지) |
| Is Trigger | 트리거 사용 여부 (체크 시 물리적인 충돌이 발생하지 않음) |
| Material | 콜라이더에 사용되는 재질을 나타냄 |
| Mesh | 충돌 감지 메쉬 오브젝트 |

## 콜라이더 충돌

콜라이더가 붙은 게임오브젝트가 서로 충돌할 경우 OnCollision or OnTrigger가 호출됨

- 충돌 감지 조건
    - 두 게임오브젝트 모두 콜라이더 컴포넌트가 추가 되어야 함
    - 움직이는 게임오브젝트에는 반드시 리지드바디 컴포넌트를 추가해야 함
    
    → 리지드바디 컴포넌트가 붙어 있을 경우 물리 동작을 수행 가능
    

# Collision

- 나랑 물리적으로 부딪친 오브젝트에 대한 정보가 담겨있다.
- 나랑 부딪친 오브젝트의 Transform, Collider, GameObject, Rigidbody, 상대 속도 등등

### OnCollision

- OnCollisionEnter : 두 게임오브젝트가 충돌이 일어날 경우 호출
- OnCollisionStay : 두 게임오브젝트가 충돌이 지속될 경우 호출
- OnCollisionExit : 두 게임오브젝트가 충돌이 끝난 경우 호출

[https://ssabi.tistory.com/45](https://ssabi.tistory.com/45)

# Trigger

- 물리적으로 부딪치지 않더라도 내 Collider 범위 안에 들어온 오브젝트에 대한 정보가 담겨 있다.

### OnTrigger

충돌하는 두 오브젝트가 물리적인 충돌은 일으키지 않고 서로 통과. 하지만 충돌 자체는 감지함. 콜라이더의 IsTrigger 를 체크할 경우 OnTrigger 발생

- OnTriggerEnter : 두 게임오브젝트가 충돌이 일어날 경우 호출
- OnTriggerStay : 두 게임오브젝트가 충돌이 지속될 경우 호출
- OnTriggerExit : 두 게임오브젝트가 충돌이 끝난 경우 호출

# UGUI

[https://ansohxxn.github.io/unity lesson 1/chapter10-1/](https://ansohxxn.github.io/unity%20lesson%201/chapter10-1/)

- UI요소를 게임 오브젝트를 게임 오브젝트 & 컴포넌트처럼 다루고 편집한다.

## 캔버스

- 모든 UI 오브젝트들을 쥐고 있는 스크린

### Render Mode

캔버스 컴포넌트

1. Render Mode 에는 "Screen Space - Overlay", "Screen Space - Camera", "World Space" 가 있다.
2. Screen Space - Overlay : 크기가 화면(디스플레이)에 맞게 고정이 되고 "Pixel Perfect", "Sort Order", "Target Display" 속성이 표시된다.
    - Pixel Perfect : 요소의 픽셀을 선명하게 유지하고 블러(Blur)를 방지하는 효과를 볼 수 있다.
    - Sort Order : 캔버스의 정렬 순서이다. 숫자가 작을 수록 먼저 렌더링 된다.
    - Target Display : 다중모니터를 사용할 경우 표시할 모니터를 지정한다.
3. Screen Space - Camera  : 아래의 "Screen Space - Camera 예시" 이미지는 카메라가 캔버스를 비추고 있는 모습이다. 카메라가 있는 위치에서 일정 거리에 자동으로 위치시킬 수 있다. Screen Space- Camera를 선택하면 하위 속성으로 "PIxel Perfect", "Render Camera", "Plance Distance", "Sorting Layer", "Order in Layer" 가 나타난다.
    
    Screen Space - Camera
    
    - Pixel Perfect : 요소의 픽셀을 선명하게 유지하고 블러(Blur)를 방지하는 효과를 볼 수 있다.
    - Render Camera : Canvas를 비출 카메라를 지정한다.
    - Plance Distance : 카메라로 부터 거리를 지정한다. 크기는 카메라 영역에 꽉 차게 조절된다.
    - Sorting Layer : 정렬 레이어를 지정한다.
    - Order in Layer : 레이어에서 렌더링 될 우선 순위를 지정한다.
4. World Space : 지정된 카메라가 비추는 영역을 렌더링한다. 아래의 "World Space 예시" 이미지는 카메라가 캔버스를 비추고 있는 모습니다. 일반적으로 카메라는 캔버스를 렌더링하지 않지만 Canvas 컴포넌트에서 지정된 카메라는 Canvas를 렌더링 한다. 하위 속성으로 "Pixel Perfect", "Event Camera", "Sortng Layer", "Order in Layer" 가 있다.
    - Pixel Perfect : 요소의 픽셀을 선명하게 유지하고 블러(Blur)를 방지하는 효과를 볼 수 있다.
    - Event Camera : Canvas를 비출 카메라를 지정한다.
    - Sorting Layer : 정렬 레이어를 지정한다.
    - Order in Layer : 레이어에서 렌더링 될 우선 순위를 지정한다.

[https://benxen.tistory.com/48](https://benxen.tistory.com/48)

## Rect Transform

### 앵커

- Anchor 는 부모 RectTansform 의 범위 안에서 UI 객체가 고정될 위치를 지정해 줍니다.
- 만약 상위 RectTransform 객체가 Canvas 이면 Anchor는 화면 크기의 범위 안에서 현재 UI 객체가 고정될 위치를 지정해 주는 것이다.
- ex→화면 해상도 변경시 객체의 위치도 변경

### 피벗

- Pivot 은 UI 객체의 기준점이다.
- ex → 객체 자체의 중심점

### 포지션

- Position 은 객체의 위치를 나타낸다.

# 좌표계

[https://timeboxstory.tistory.com/128](https://timeboxstory.tistory.com/128)

- 유니티는 왼손 좌표계를 사용
    - x축 : 엄지 방향 : 오른쪽
    - y축 : 검지 방향 : 위쪽
    - z축 : 중지 방향 : 앞쪽

## Global 좌표

- 유니티의 가상 공간의 기준 좌표

## Local 좌표

- 오브젝트 기준 시점의 좌표
- 이동된 값 = 부모 좌표값 + 자식 좌표값
