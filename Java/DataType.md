# Data Type

# Primitive Type(기본형 타입)

- 계산을 위해 실제 값을 저장

### 특징

- 모두 소문자로 시작된다
- 비객체 타입이므로 null 값을 가질 수 없다. (기본값이 정해져 있음)
- 변수의 선언과 동시에 메모리 생성모든 값 타입은 메모리의 스택(stack)에 저장됨
- 저장공간에 실제 자료 값을 가진다

출처:

[https://inpa.tistory.com/entry/JAVA-☕-변수의-기본형-참조형-타입](https://inpa.tistory.com/entry/JAVA-%E2%98%95-%EB%B3%80%EC%88%98%EC%9D%98-%EA%B8%B0%EB%B3%B8%ED%98%95-%EC%B0%B8%EC%A1%B0%ED%98%95-%ED%83%80%EC%9E%85)

|  |  타입 |  할당되는 메모리 크기 |  기본값 |  데이터의 표현 범위 |
| --- | --- | --- | --- | --- |
| 논리형 |  boolean |  1 byte |  false |  true, false |
| 정수형 | byte |  1 byte |  0 |  -128 ~ 127 |
| 정수형 | short |  2 byte |  0 |  -32,768 ~ 32,767 |
| 정수형 | int(기본) |  4 byte  |  0 |  -2,147,483,648 ~ 2,147,483,647 |
| 정수형 | long |  8 byte |  0L |  -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807 |
| 실수형 |  float |  4 byte |  0.0F |  (3.4 X 10-38) ~ (3.4 X 1038) 의 근사값 |
| 실수형 | double(기본) |  8 byte |  0.0 |  (1.7 X 10-308) ~ (1.7 X 10308) 의 근사값 |
| 문자형 | char  | 2 byte (유니코드) '\u0000'  0 ~ 65,535 | '\u0000'  |  0 ~ 65,535 |

# ****Reference type(참조형 타입)****

- 객체의 주소를 저장

### 특징

- 기본형 과는 달리 실제 값이 저장되지 않고, 자료가 저장된 공간의 주소를 저장한다.
- 즉, 실제 값은 다른 곳에 있으며 값이 있는 주소를 가지고 있어서 나중에 그 주소를 참조해서 값을 가져온다.
- 메모리의 힙(heap)에 실제 값을 저장하고, 그 참조값(주소값)을 갖는 변수는 스택에 저장
- 참조형 변수는 null로 초기화 시킬 수 있다

출처:

[https://inpa.tistory.com/entry/JAVA-☕-변수의-기본형-참조형-타입](https://inpa.tistory.com/entry/JAVA-%E2%98%95-%EB%B3%80%EC%88%98%EC%9D%98-%EA%B8%B0%EB%B3%B8%ED%98%95-%EC%B0%B8%EC%A1%B0%ED%98%95-%ED%83%80%EC%9E%85)

|  타입 |  기본값 | 할당되는 메모리 크기  |
| --- | --- | --- |
|  배열(Array) |  Null | 4 byte  |
| 열거(Enumeration) |  Null | 4 byte  |
| 클래스(Class) |  Null | 4 byte  |
| 인터페이스(Interface) |  Null | 4 byte  |

# 메모리에서 두 변수 비교

[https://velog.io/@yh20studio/Java-기본형-변수와-참조형-변수](https://velog.io/@yh20studio/Java-%EA%B8%B0%EB%B3%B8%ED%98%95-%EB%B3%80%EC%88%98%EC%99%80-%EC%B0%B8%EC%A1%B0%ED%98%95-%EB%B3%80%EC%88%98)

<img width="767" alt="Untitled" src="https://github.com/ssssihoon/Learn/assets/127017020/c85fc8c5-eed8-45bd-8e43-b12e6f590fc7">

---

[https://zangzangs.tistory.com/107](https://zangzangs.tistory.com/107)

<img width="501" alt="Untitled 1" src="https://github.com/ssssihoon/Learn/assets/127017020/123095ed-8c35-4619-a96c-b62770dabe7e">


Code

: 실행할 프로그램의 코드가 저장되는 영역으로 텍스트(code) 영역

Data

: 프로그램의 전역 변수와 정적(static) 변수가 저장되는 영역

Heap

: 사용자가 직접 관리할 수 있는 '그리고 해야만 하는' 메모리 영역

Stack 

: 메모리의 스택(stack) 영역은 함수의 호출과 관계되는 지역 변수와 매개변수가 저장되는 영역

[https://tcpschool.com/c/c_memory_structure](https://tcpschool.com/c/c_memory_structure)
