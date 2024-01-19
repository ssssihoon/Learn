# 클래스 정의하는 방법

```java
public class 클래스명 {
	private String modelName;
	private int ---
	----
	----
}
```

# 객체 만드는 방법 (new 키워드), 메소드 정의하는 방법

```java
클래스명 변수명;
// 클래스의 객체를 참조하기 위한 참조변수 선언
변수명 = new 클래스명();
// 클래스의 객체를 생성 후, 해당 객체의 주소를 참조변수에 저장.
```

클래스란? : 객체를 정의해 놓은 것.

인스턴스란? : 클래스로부터 객체를 만드는 과정을 인스턴스화라고 하는데. 어떤 클래스로부터 만들어진 객체를 그 클래스의 인스턴스라고 한다.

→ 객체 ~= 인스턴스

객체 : 클래스의 모든 인스턴스

인스턴스 : 특정 객체

- ex : TV
    - 속성 : 크기, 길이, 높이, 색상, 볼륨, 채널 , 기능 : 켜기, 끄기, 볼륨 높이기, 볼륨 낮추기, 채널 변경하기
        - 속성 → 멤버변수, 기능 → 메서드

```java
class TV {
	// TV의 속성(멤버변수)
	String color;
	boolean power;
	int channel;

	// TV의 기능(메서드) // 메서드 정의 
	void power()
	{
	   power = !power; // 티비 전원을 끄거나 킴
	}
	void channelup()
	{
		++channel; // 채널을 높임
	}

class TVTest {
	public static void main(String args[]) {
		TV t; // 변수 t를 선언
		t = new TV(); // TV 클래스의 객체를 생성하여 변수 t에 할당
		t.channel = 7; // t를 통해 객체 멤버변수에 접근
		t.channelup();
	}
```

# 생성자 정의하는 방법

생성자란? : 인스턴스가 생성될 때 호출되는 인스턴스 초기화 메서드

1. **생성자의 이름은 클래스의 이름과 같아야 함**
2. 생성자를 통해서 초기 멤버변수의 값 세팅이 가능하다.
3. 생성자는 리턴 값이 없다.

```java
class Bicycle {
	int weight_kg = 10 // 무게 초기화

    
    public Bicycle() {           // 생성자 정의
			int weight_kg = 20 // 무게 변경
    }

}
```

# this 키워드

- 객체 내에서 현재 객체를 참조하는 데 사용된다.
- 주로 멤버 변수와 메서드의 매개변수의 이름이 동일한 경우 현재 객체를 가리키기 위해 사용된다.

```java
class Car {
    String color;

    // 생성자
    public Car(String color) {
        // 여기서 this.color는 객체의 멤버 변수를 나타냅니다.
        this.color = color;
    }

    // 메서드
    public void printColor() {
        // 여기서 this.color는 객체의 멤버 변수를 나타냅니다.
        System.out.println("Car color: " + this.color);
    }
}

public class CarTest {
    public static void main(String[] args) {
        // Car 객체 생성
        Car myCar = new Car("Blue");

        // printColor 메서드 호출
        myCar.printColor();
    }
}
```

# 자바 상속의 특징

[https://velog.io/@donglee99/자바-상속의-특징-임시](https://velog.io/@donglee99/%EC%9E%90%EB%B0%94-%EC%83%81%EC%86%8D%EC%9D%98-%ED%8A%B9%EC%A7%95-%EC%9E%84%EC%8B%9C)

**상속 방법**

- extends를 사용하면 부모의 클래스 객체를 물려받게 된다.
1. 상속 방법

```java
class 자식클래스 extends 부모클래스{

}
```

1. 부모 클래스

```java
public class Tv {

	String
```

1. 자식 클래스

```java
public class Samsung extends Tv {

	String tvType;
}
```

## 장점

- 코드의 확장성, 재사용성 상승
- 중복된 코드 제거가능
- 객체지향 프로그래밍에서의 다형성

## 단점

- 캡슐화를 깨뜨린다.

## 특징

- 자바의 클래스는 단일상속이 원칙이다. 다중상속이 불가능하다.
- 자바에서 계층구조의 최상위에는 java.lang.Object 클래스가 있다.

## super 키워드

**`super`** 키워드를 사용하여 부모 클래스의 **생성자**나 **메서드**에 접근할 수 있다.

- **부모 클래스의 멤버 호출**
- **부모 클래스의 생성자 호출**

**`*extends와 상속 super 구분 할 것.*`**

- extends(상속) : 부모클래스의 ***객체***를 물려받음 ( 객체 = 모든 메서드, 멤버변수 )
- super : 부모클래스의 ***생성자***나 ***메서드***에 접근

## 메소드 오버라이딩

- 부모클래스에게 상속받는 메서드를 재정의하여 사용

# 추상 클래스

[https://velog.io/@ung6860/JAVA추상클래스에-대하여-알아보자](https://velog.io/@ung6860/JAVA%EC%B6%94%EC%83%81%ED%81%B4%EB%9E%98%EC%8A%A4%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

- 추상 클래스를 선언 할 때에는 abstract 키워드를 사용해야한다.
- 추상 클래스를 상속받는 모든 서브 클래스들은 추상 메소드를 반드시 재정의(강제구현)해야한다.
- new 연산자 사용을 통한 인스턴스화가 불가능하다.

메소드의 구현부가 미완성인 채로 남겨지기 때문에 이를 오버라이드하는 서브 클래스에서 실행부에 세부적인 로직을 구현해야한다.

```java
//abstract 키워드 사용
public abstract class Test{
	/*
     -추상 메소드(abstract 키워드 생략가능)
     -{}블록이 없기 때문에 상속받은 서브클래스에서 재정의(강제구현)
    */
    public abstract void testMethod();
    //일반 메소드
    public void testMethod2(){
    	실행문 작성...
    };
}
```

# 접근지시자

[https://nuemeel.tistory.com/19](https://nuemeel.tistory.com/19)

[https://beststar-1.tistory.com/18](https://beststar-1.tistory.com/18)

- 객체 생성을 막기 위해 생성자를 호출하지 못하게 함
- 객체의 특정 데이터를 보호하기 위해 해당 필드에 접근하지 못하게 함
- 특정 메서드를 호출할 수 없도록 제한하는 기능을 함

### 접근지시자의 종류

접근지시자는 public, protected, default, private 이렇게 네 가지로 나뉩니다. 각 개념에 대해 더 구체적으로 알아봅시다.

- public

모든 접근을 허용하는 지정자입니다. 같은 패키지 내에서도 / 다른 패키지에서도 해당 클래스에 접근할 수 있습니다.

- protected

같은 패키지 내에 있는 모든 클래스들은 접근이 가능합니다. 다른 패키지에 있더라도 상속관계라면 접근 가능합니다.

- default

접근지정자를 설정하지 않았을 때 가장 기본이 되는 접근지시자입니다. 같은 패키지 내에 있는 클래스들만 접근이 가능합니다.

- private

현재 객체 내에서만 접근이 허용됩니다. 같은 패키지에 있더라도 다른 객체에서는 접근이 불가합니다.
