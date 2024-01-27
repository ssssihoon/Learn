# 문법1

# 조건문

## if ~ else 문

```java
if (조건식)
{
	조건식에 해당될 때 실행될 문장
}
else if (조건식)
{
	조건식에 해당될 때 실행될 문장
}
else
{
	어느 조건식에도 만족하지 않을 때 실행될 문장
}
```

## switch문

```java
switch (조건식)
{
	case 값1 :
		조건식의 결과가 값1과 같은 경우 실행될 문장
		break;
	case 값2 :
		조건식의 결과가 값2과 같은 경우 실행될 문장
		break;
	case 값3 :
		조건식의 결과가 값3과 같은 경우 실행될 문장
		break;
}
```

# 반복문

## for 문

```java
for (초기화;조건식;증감식)                            // 초기화 : 자료형 변수 = 값
{
	수행될 문장
}
```

## while 문

```java
while (조건식)
{
	조건식이 참일 동안 수행될 문장
}
```

## do - while 문

```java
do {
	조건식이 참일 동안 수행될 문장
} while (조건식)
```

# 배열

## 배열 선언

| 선언방법 | 선언 예 |
| --- | --- |
| 타입[] 변수이름; | int[] score; |
|  | String[] name; |
| 타입 변수이름[]; | int score[]; |
|  | String name[]; |
- `int[] 배열명 = new int [n];` : 길이가 n인 정수형 배열 생성

### `배열명.length;`

- 배열의 길이를 알 수 있다.

### 배열의 초기화

- 하나하나 대입하기보다는 for문 사용

```java
int[] score = new int[5];

for(int i = 0; i < score.length; i++)
	score[i] = i * 10 + 50;
```

## 2차원 배열 선언

| 선언방법 | 선언 예 |
| --- | --- |
| 타입[][] 변수이름; | int[][] score; |
| 타입 변수이름[][]; | int score[][]; |
| 타입[] 변수이름[]; | int[] score[]; |

### 초기화

```java
int [][] arr = new int [][]{ {1, 2, 3}, {4, 5, 6} };
```
