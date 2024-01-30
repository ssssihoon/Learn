# Java

```java
class 클래스이름 {
    **public static void main(String[] args) // main메서드의 선언부**
 {
        System.out.println("Hello, World.");
    }
}
```

## 변수

### 변수 선언

- `변수타입 변수이름;`
- `int age;`

### 변수 초기화

- `변수타입 변수이름 = 값;`
- `int age = 25;`

### 변수 타입

- 기본형
    - 실제 값을 저장
    - 논리형, 문자형, 정수형, 실수형 등

| 자료형 | 크기(바이트) |
| --- | --- |
| boolean | 1 |
| char | 2 |
| byte | 1 |
| short | 2 |
| int | 4 |
| long | 8 |
| float | 4 |
| double | 8 |

float의 정밀도 : 7자리

double의 정밀도 : 15자리

- 참조형
    - 주소 값을 가진다
    - `클래스이름 변수이름;`

### 상수(리터럴)

- `final 변수타입 변수 = 값(리터럴);` (변수의 이름은 대문자{관례}, final : 상수 변수 선언)
- 접미사

| 종류 | 접미사 |
| --- | --- |
| 정수형(long) | L |
| 실수형(float, double) | f, d |

문자열에는 어떤 타입을 더해도 문자열이 된다.

### 지시자

| 지시자 | 설명 |
| --- | --- |
| %b | 불리언 형식으로 출력 |
| %d | 10진 정수의 형식으로 출력 |
| %o | 8진 정수의 형식으로 출력 |
| %x, %X | 16진 정수의 형식으로 출력 |
| %f | 부동 소수점의 형식으로 출력 |
| %e, %E | 지수 표현식의 형식으로 출력 |
| %c | 문자로 출력 |
| %s | 문자열로 출력 |
- printf() : 줄바꿈을 하지 않는다.

### Scanner

- `import java.util.*;` : Scanner 클래스를 사용하기 위해 추가
- `Scanner scanner = new Scanner(System.in);` : Scanner 클래스의 객체를 생성
- `String input = scanner.nextLine();` : 입력받은 내용을 input에 저장
- `int 변수 = Integer.parseInt(input);` : 입력받은 내용을 int 타입의 값으로 변환
    - 바로 입력받을 수 있는 메서드 : `int 변수 = scanner.nextInt();`

```java
import java.util.*;

class hello {
    public static void main(String[] args)
    {
        Scanner scanner = new Scanner(System.in);
        String input = scanner.nextLine();
        
        System.out.println(input);
    }
}
```

### 형변환

```java
// 아스키코드를 위한 형변환(정수형 표기)

import java.util.*;

class hello {
    public static void main(String[] args)
    {
        char ch = 'A';
        int code = (int)ch;

        System.out.print(code);
    }
}
```

### 특수문자

| 특수문자 | 문자 리터럴 |
| --- | --- |
| tab | \t |
| backspace | \b |
| form feed | \f |
| new line | \n |
| carriage return | \r |
| 역슬래쉬 | \\ |
| 작은따옴표 | \’ |
| 큰따옴표 | \” |
| 유니코드문자 | \u유니코드 |

### 형변환(캐스팅)

- `타입(피연산자);`

## 연산자

반올림

- `Math.round(피연산자);`

### 난수

- 최솟값 ≤ `Math.random()` ≤ 최댓값

## 배열

### 배열 선언

| 선언방법 | 선언 예 |
| --- | --- |
| 타입[] 변수이름; | int[] score; |
|  | String[] name; |
| 타입 변수이름[]; | int score[]; |
|  | String name[]; |
- `int[] 배열명 = new int [n];` : 길이가 n인 정수형 배열

### `배열명.length;`

- 배열의 길이를 알 수 있다.

### 배열의 초기화

- 하나하나 대입하기보다는 for문 사용

### 배열의 복사

1. `int [] arr = new int[5];` // 원래 배열의 초기화
2. `int [] tmp = new int [arr.length*2];` // 새로운 배열 초기화(길이 2배)
3. `for (int i=0; i<arr.length; i++){` // for문을 돌려 값 대입
    1. `tmp[i] = arr[i];}`
- 이것보다 효율적인 방법
    - `System.arraycopy(복사할 배열명, 의 인덱스 시작점, 복사받을 배열명, 의 인덱스 시작점);`
    

## String 배열

### String 배열 선언

- `String[] 문자열명 = new String[n];` // n개를 담을 수 있는 배열 선언

### String 배열 초기화

- `String[] 문자열명 = new String[] {”value1”, “value2”, ‘’’};`
- 또는
- `String[] 문자열명 = {”value1”, “value2”, ‘’’};`

### String 메서드

| 메서드 | 설명 |
| --- | --- |
| char charAt(int index) | 문자열에서 해당 위치에 있는 문자를 반환 |
| int length() | 문자열의 길이를 반환 |
| String substring(int from, int to) | 문자열에서 해당 범위에 있는 문자열을 반환 |
| boolean equals(Object obj) | 문자열의 내용이 obj와 같은지 확인. 같으면 true, 다르면 false |
| char[] toCharArray() | 문자열을 문자배열로 변환해서 반환 |