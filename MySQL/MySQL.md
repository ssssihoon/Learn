# MySQL

대소문자를 구분하지 않는다.

## 데이터 분석 과정

문제 인식 → 데이터 수집과 가공 → 데이터 분석 → 분결 결과 실행

데이터 수집과 가공, 데이터 분석은 SQL 적용 가능

### 데이터 전처리

데이터를 분석하기위해 데이터를 수집, 가공, 처리 하는 과정을 전처리라고 한다.

# 데이터베이스

- DB (DataBase)
- DBMS (DataBase Management System) : 데이터베이스를 관리한다.

## 관계형 데이터베이스

열과 컬럼을 갖는 2차원 테이블을 중심으로 설계

하나의 열에는 문자와 숫자 타입의 데이터를 섞어 쓸 수 없다.

- 오라클
- DB2
- MySQL

### 구성요소

- 데이터를 저장하는 개체와 관계들의 집합
- 일관성, 정확성, 신뢰성을 위한 트랜잭션, 무결성, 동시성 제어 등의 개념 존재

### 객체

| TABLE | 행과 열로 구성된 데이터의 저장 단위 |
| --- | --- |
| VIEW | 부분 집합이자 가상의 테이블 |
| INDEX | 주소를 사용해 행을 검색 |
| SEQUENCE | 고유한 번호를 자동 생성 |
| SYNONYM | 객체에 별칭을 부여 |

## 계층형 데이터베이스

개인 컴퓨터의 저장 장치에서 주로 사용되는 방식( like 폴더와 파일.. )

## 객체 지향 데이터베이스

객체 지향 데이터 모델을 지원하는 데이터베이스

Java, C++ 등 객체 지향 언어의 객체 지향 프로그래밍에 적합한 데이터베이스

## XML 데이터베이스

XML 표준 문서 구조를 계층형 트리 형태로 저장하거나 관리하기 위해 만든 데이터베이스

# SQL

DBMS와 통신하기 위한 언어

사용자 → SQL → DBMS → 데이터베이스

사용자 ←———— DBMS ← 데이터베이스

- 사용하기 쉽다.
- 비절차적 언어
- 데이터를 정의, 검색, 조작 가능
- 표준 언어

## 명령어

| DML | Data Manipulation Language | 데이터 조작어 |
| --- | --- | --- |
| DDL | Data Definition Language | 데이터 정의어 |
| DCL | Data Control Language | 데이터 제어어 |
| TCL | Transaction Control Language | 트랜잭션 제어어 |
- DML : SELECT, INSERT, UPTATE, DELETE
- DDL : CREATE, ALTER, DROP, RENAME, TURNCATE
- DCL : GRANT, REVOKE
- TCL : COMMIT, ROLLBACK, SAVEPOINT

# 주석

```sql
-- 이 부분은 주석입니다.

 (--)하이픈 두개 이후로는 주석을 달 수 있다.
```

# SELECT

## Collumn

- SELECT 컬럼명 FROM 테이블이름;

### 모든 컬럼 선택

```sql
SELECT * FROM Tablename;

-- *은 asterisk로 모든~ 의 뜻 -> 모든 컬럼을 가져온다.
```

### 특정 컬럼 선택

```sql
SELECT CustomerName FROM Customers;

-- Customers 에서 CustomerName라는 컬럼을 가져온다.
```

### 다중 컬럼 선택

```sql
SELECT CustomerName, ContactName FROM Customers;

-- Customers에서 CustomerName, ContactName 두개 를 추출
-> 두개 이상의 컬럼을 선택 할 수 있다.
```

### 컬럼이 아닌 값 선택

```sql
SELECT
	customerName, 1, 'Hello', NULL
FROM Customers;

-- CustomerName, 1, Hello, Null 컬럼명으로 이루어진 데이터프레임 생성
```

## Row

- WHERE : 조건문 (if와 같다)

```sql
SELECT 컬럼명 FROM 데이터명
WHERE 카테고리컬럼명 = "원하는 데이터";

-- 원하는 데이터 "str" or int
```

## 특정기준 정렬 ORDER BY

```sql
select * from Customers
order by ContactName;

-- ContactName을 기준으로 오름차순 정렬을 해준다.

```

### 내림차순 _ desc

```sql
select * from Customers
order by ContactName desc;

-- desc는 내림차순 정렬이다.
asc -> 오름차순
```

### 두 가지 이상의 기준 정렬

```sql
select * from Customers
order by ContactName, Address desc;

-- ContactName은 오름차순, Address는 내림차순이다.

order by ContactName desc, Address desc;
-- 두가지 다 내림차순이다.
```

## 원하는 만큼의 데이터 가져오기

```sql
limit (건너뛸 개수), 가져올 개수;
-- 건너뛸 개수는 생략 가능하다.
```

## 원하는 별명으로 데이터 가져오기

ex:

```sql
select
customerid as id,
customername as name,
from customers;
```

# 연산자

| + | 더하기 |
| --- | --- |
| - | 빼기 |
| * | 곱하기 |
| / | 나누기 |
| %, mod | 나머지 |

```sql
select 5 - 2 as difference;

-- 3
```

```sql
select 'abc' + 3;
select 'abc' * 3;
select '1' + '002' * 3

-- 3
-- 0
-- 7

-- 문자열을 0으로 인식한다.
-- 숫자로 구성된 문자열은 숫자로 자동인식
```

## 논리 연산자

| TRUE | 참 |
| --- | --- |
| FALSE | 거짓 |
| NOT, ! | 반대 |
| AND | 모두 |
| OR | 한 쪽만 |
- <> 는 ≠ 과 같다

| BETWEEN [MIN] AND [MAX] | 두 값 사이에 있다. |
| --- | --- |

```sql
select 5 between 2 and 10;
-- 1      <- true
```

| LIKE’…%…’ | 0~N개 문자를 가진 패턴 |
| --- | --- |
| LIKE’…_…’ | _개수 만큼의 문자를 가진 패턴 |

```sql
SELECT
  'HELLO' LIKE 'hel%',
  'HELLO' LIKE 'H%',
  'HELLO' LIKE 'H%O',
  'HELLO' LIKE '%O',
  'HELLO' LIKE '%HELLO%',
  'HELLO' LIKE '%H',
  'HELLO' LIKE 'L%'

-- 1
1
1
1
1
0
0
```

```sql
select * from employees
where notes like '%economics%'
```

```sql
SELECT * FROM OrderDetails
WHERE OrderID LIKE '1025_'

'''
1025다음 한 글자가 온다.
```

# 함수

## 숫자 관련

| ROUND | 반올림 |
| --- | --- |
| CEIL | 올림 |
| FLOOR | 내림 |

```sql
select round(0.012);
-- 0
```

| ABS | 절대값 |
| --- | --- |

```sql
select abs(5.3);
-- 5
```

| GREATEST | 괄호 안의 가장 큰 값 |
| --- | --- |
| LEAST | 괄호 안의 가장 작은 값 |

```sql
select greatest(1, 5, 7);
-- 7
select least(6, 2, 1);
-- 1
```

| MAX | 최대값 |
| --- | --- |
| MIN | 최소값 |
| COUNT | 갯수 |
| SUM | 합 |
| AVG | 평균 |

| POW(A, B), POWER(A, B) | A를 B만큼 제곱 |
| --- | --- |
| SQRT | 제곱근 |

```sql
select
	pow(2, 3) # 8
	sort(16) # 4
```

| TRUNCATE(N,n) | N을 소숫점 n자리까지 선택 |
| --- | --- |

```sql
select
	truncate(3.1415, 2)
-- 3.14
```

## 문자열 관련

| UCASE, UPPER | 모두 대문자로 |
| --- | --- |
| LCASE, LOWER | 모두 소문자로 |

```sql
select
	upper("hello");
-- HELLO
```

| CONCAT(...) | 괄호 안의 내용 이어붙임 |
| --- | --- |
| CONCAT_WS(S, ...) | 괄호 안의 내용 S로 이어붙임 |

```sql
select
	concat('hello', 'my age', 'is', 24);
-- hellomy ageis24

select
	concat_ws('+', 'hello', 'my age', 'is', 24);
-- hello+my age+is+24
```

| SUBSTR, SUBSTRING | 주어진 값에 따라 문자열 자름 |
| --- | --- |
| LEFT | 왼쪽부터 N글자 |
| RIGHT | 오른쪽부터 N글자 |

```sql
select
	substr('abcde', 2);
select
	substr('abcde', 2, 2);
select
	left('abcde', 2);

-- bcde # 앞에서 2-1 자름
-- bc #앞에서 2-1, 그후 2칸 까지 출력
-- ab # 앞에서 2번째 까지만 출력
```

| LENGTH | 문자열의 바이트 길이 |
| --- | --- |
| CHAR_LENGTH, CHARACTER_LEGNTH | 문자열의 문자 길이 |
- CHAR_LENGTH, CHARACTER_LEGNTH 이것이 흔히 아는 len()이다.
- LENGTH는 한글에서 바이트 길이가 나오기 때문
    
    
    | TRIM | 양쪽 공백 제거 |
    | --- | --- |
    | LTRIM | 왼쪽 공백 제거 |
    | RTRIM | 오른쪽 공백 제거 |

```sql
select
	concat('|, 'Hello', '|'),
	concat('|, trim('Hello'), '|');

-- | Hello |
-- |Hello|

```

| LPAD(S, N, P) | S가 N글자가 될 때까지 P를 이어붙임 |
| --- | --- |
| RPAD(S, N, P) | S가 N글자가 될 때까지 P를 이어붙임 |

```sql
select
	rpad('abc', 5, '__');

-- abc__
```

| REPLACE(S, A, B) | S중 A를 B로 변경 |
| --- | --- |

```sql
select
	replace('맥도날드에서 햄버거를 먹었다', '맥도날드', '버거킹');

-- 버거킹에서 햄버거를 먹었다
```

| INSTR(S, s) | S중 s의 첫 위치 반환, 없을 시 0 |
| --- | --- |

```sql
select
	instr('abcde', 'abc')
	instr('abcde', 'cd')
	instr('abcde', 'dg')
	instr('abcde', 'f');

-- 1
-- 3
-- 0 
-- 0
```

| CAST(A AS T) | A를 T 자료형으로 변환 |
| --- | --- |
| CONVERT(A, T) | A를 T 자료형으로 변환 |

```sql
select
	'01' = '1'
	convert('01', decimal), = convert('1', decimal);
 
-- 1
```

## 기타
