- [Unity\_C#](#unity_c)
	- [메세지 출력](#메세지-출력)
	- [변수](#변수)
		- [변수 선언](#변수-선언)
	- [그룹형 변수](#그룹형-변수)
		- [배열](#배열)
		- [리스트](#리스트)
	- [연산자](#연산자)
	- [조건문](#조건문)
		- [if ~ else](#if--else)
		- [switch ~ case](#switch--case)
	- [반복문](#반복문)
		- [While](#while)
		- [for](#for)
		- [foreach](#foreach)
	- [함수](#함수)
	- [클래스](#클래스)
	- [상속](#상속)

# Unity_C#

## 메세지 출력

- `Debug.Log(”출력할 메세지”);`
    - Debug.Log(”Hello Unity!”);

## 변수

| 이름 | 자료형 |
| --- | --- |
| int | 정수형 |
| float | 숫자형 |
| string | 문자열 |
| bool | 논리형 |

### 변수 선언

- `자료형 변수명 = 값;`
    - int level = 5;
    - float strength = 15.5f;
    - string playername = “홍길동”;
    - bool Fulllevel = false;

## 그룹형 변수

### 배열

고정형 그룹형 변수

- `string[] 배열명 = {”value1”, “value2”, “ ‘’’ “}`
- 인덱싱 가능
- `int[] 배열명 = new int[3];`
    - 정수형 배열을 초기화 하고, 배열의 크기를 초기화(new)
- 길이 조회
    - `.Length;`

### 리스트

가변형 그룹형 변수

- `List<자료형> 리스트명 = new List<자료형>();`

리스트 값 초기화

- `리스트명.add(’값’);`

⇒ 가변이기 때문에 add를 사용한만큼 리스트의 크기 조정

- 리스트 값 제거
    - `리스트명.RemoveAt(인덱스);`
- 길이 조회
    - `.Count;`

## 연산자

- 문자열 붙이기 : +로 가능
- `? A : B`  : True면 A 출력 False면 B출력

## 조건문

### if ~ else

```csharp
if (조건식) 
	{ 
	참일 때 실행 내용 
	}
else if (조건식)
	{
	참일 때 실행 내용
	}
else
	{
	참일 때 실행 내용
	}
```

### switch ~ case

```csharp
switch(변수) 
{
	case 값1;

		break;
	case 값2;

		break;
	case 값3;

		break;
	default;

		break;
}

```

## 반복문

### While

```csharp
while (조건식)
{
	실행내용
}
```

### for

```csharp
for (연산될 변수; 조건식; 증감)
{
	실행내용
}
```

### foreach

```csharp
foreach (자료형 리스트명2 in 리스트명1)
{
	실행내용
}
```

## 함수

```csharp
자료형 함수명(받는자료형 받는변수명)
{
	함수정의부
	return 반환값;      // 생략가능
}
```

```csharp
int heal(int health)
{
	health += 10;
	Debug.Log("힐을 받았습니다. " + Health);
	return health;
}
```

## 클래스

- class를 이용할 파일을 만들어야 함.

ex

변수 설정시 (private) 접근자가 생략이 되어있는데

이를 public접근자로 변경해 줌으로써 외부 클래스에 공개

`player.` 을 사용할 수 있다.

```csharp
// Actor.cs 

public class Actor {

	public int id;
	public string name;
	public string title;
	public string weapon;
	public float strength;
	public int level;

	public string Talk()
	{
		return "대화를 걸었습니다.";
	}
	
	public string HasWeapon()
	{
		return weapon;
	}
```

```csharp
// main.cs

Actor player = new Actor();

player.id = 12;
player.name = "홍길동";
```

↑인스턴스화 : 정의된 클래스를 변수 초기화로 실체화

- `클래스명 변수명 = new 클래스명();`

## 상속

```csharp
// Player.cs

public class Player : Actor 
{

	public string move()
	{
		return "플레이어는 움직입니다.";
	}
}
```

```csharp
// main.cs

Player player = new Player();

player.id = 12;
player.name = "홍길동";
```

⇒ Actor 클래스는 부모클래스

⇒ Player 클래스는 자식클래스
