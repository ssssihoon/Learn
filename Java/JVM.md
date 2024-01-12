# 제목 없음

# JVM

Java Virtual Machine : 자바 가상 머신

각자의 운영체제에 맞는 실행파일로 바꿔주는 역할

Java로 개발한 프로그램을 컴파일하여 만들어지는 바이트코드(.class)를 실행시키기 위한 가상머신 → 모든 플랫폼에서 동작하도록 할 수 있다.

[출처](https://coding-factory.tistory.com/827)

<img width="642" alt="Untitled" src="https://github.com/ssssihoon/Learn/assets/127017020/90622f54-1034-4353-b0e6-f12f090eec58">


- 컴파일러 : 특정 프로그래밍 언어로 쓰여 있는 문서를 다른 프로그래밍 언어로 옮기는 언어 번역 프로그램을 말한다.
- 바이트 코드 : 고급 언어로 작성된 소스 코드를 가상머신이 이해할 수 있는 중간 코드로 컴파일한 것

---

## JVM 동작 방식

[출처](https://coding-factory.tistory.com/828)

<img width="578" alt="Untitled 1" src="https://github.com/ssssihoon/Learn/assets/127017020/a5c751da-1830-4842-9755-7446f556dbfb">

**1.** 자바로 개발된 프로그램을 실행하면 JVM은 OS로부터 메모리를 할당합니다.

**2.** 자바 컴파일러(javac)가 자바 소스코드(.java)를 자바 바이트코드(.class)로 컴파일합니다.

**3.** Class Loader를 통해 JVM Runtime Data Area로 로딩합니다.

**4.** Runtime Data Area에 로딩 된 .class들은 Execution Engine을 통해 해석합니다.

**5.** 해석된 바이트 코드는 Runtime Data Area의 각 영역에 배치되어 수행하며 이 과정에서 Execution Engine에 의해 GC의 작동과 스레드 동기화가 이루어집니다.

- Garbage Collector : 불필요한 메모리를 알아서 정리해주는 역할
- Thread : 운영체제로부터 자원을 할당받아 소스 코드를 실행하여 데이터를 처리하는 역할
    - 두 개 이상의 스레드가 공유 데이터에 동시에 접근하면 결과가 좋지 않을 수 있기 때문에 동기화를 사용

---

## JVM 구성요소

- Class Loader
- Execution Engine
- Grabage Collectors
- Runtime Data Area

[출처]([https://doozi0316.tistory.com/entry/1주차-JVM은-무엇이며-자바-코드는-어떻게-실행하는-것인가](https://doozi0316.tistory.com/entry/1%EC%A3%BC%EC%B0%A8-JVM%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B4%EB%A9%B0-%EC%9E%90%EB%B0%94-%EC%BD%94%EB%93%9C%EB%8A%94-%EC%96%B4%EB%96%BB%EA%B2%8C-%EC%8B%A4%ED%96%89%ED%95%98%EB%8A%94-%EA%B2%83%EC%9D%B8%EA%B0%80))

### Class Loader

클래스를 참조할 때, 그 클래스를 로드하고 링크하는 역할

### Execution Engine

클래스를 실행시키는 역할

- Interpreter : 바이트 코드를 명령어 단위로 읽어서 실행
- JIT(Just-in-Time) : 인터프리터 방식으로 실행하다가 적절한 시점에 바이트 코드 전체를 컴파일하여 기계어로 변경하고, 이후에는 해당 더 이상 인터프리팅 하지 않고 기계어로 직접 실행하는 방식

### Garbage Collector

불필요한 메모리를 알아서 정리해주는 역할

### Runtime Data Area

프로그램을 수행하기 위해 OS에서 할당받은 메모리 공간
