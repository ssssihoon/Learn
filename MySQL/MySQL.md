- [참고한 사이트](https://www.yalco.kr/)

- [MySQL](#mysql)
	- [데이터 분석 과정](#데이터-분석-과정)
		- [데이터 전처리](#데이터-전처리)
- [데이터베이스](#데이터베이스)
	- [관계형 데이터베이스](#관계형-데이터베이스)
		- [구성요소](#구성요소)
		- [객체](#객체)
	- [계층형 데이터베이스](#계층형-데이터베이스)
	- [객체 지향 데이터베이스](#객체-지향-데이터베이스)
	- [XML 데이터베이스](#xml-데이터베이스)
- [SQL](#sql)
	- [명령어](#명령어)
- [주석](#주석)
- [SELECT](#select)
	- [Collumn](#collumn)
		- [모든 컬럼 선택](#모든-컬럼-선택)
		- [특정 컬럼 선택](#특정-컬럼-선택)
		- [다중 컬럼 선택](#다중-컬럼-선택)
		- [컬럼이 아닌 값 선택](#컬럼이-아닌-값-선택)
	- [Row](#row)
	- [특정기준 정렬 ORDER BY](#특정기준-정렬-order-by)
		- [내림차순 \_ desc](#내림차순-_-desc)
		- [두 가지 이상의 기준 정렬](#두-가지-이상의-기준-정렬)
	- [원하는 만큼의 데이터 가져오기](#원하는-만큼의-데이터-가져오기)
	- [원하는 별명으로 데이터 가져오기](#원하는-별명으로-데이터-가져오기)
- [연산자](#연산자)
	- [논리 연산자](#논리-연산자)
- [함수](#함수)
	- [숫자 관련](#숫자-관련)
	- [문자열 관련](#문자열-관련)
	- [시간 관련](#시간-관련)
		- [DATE\_FORMAT](#date_format)
	- [기타 함수](#기타-함수)
- [그룹](#그룹)
	- [GROUP BY](#group-by)
		- [활용](#활용)
	- [WITH ROLLUP](#with-rollup)
	- [HAVING](#having)
	- [DISTINCT](#distinct)
		- [GROUP BY, DISTINCT 함께 사용](#group-by-distinct-함께-사용)
- [쿼리 안에 서브쿼리](#쿼리-안에-서브쿼리)
	- [비상관 서브쿼리](#비상관-서브쿼리)
	- [상관 서브쿼리](#상관-서브쿼리)
	- [(NOT)EXISTS](#notexists)
- [JOIN](#join)
	- [JOIN - INNERJOIN(기본값)](#join---innerjoin기본값)
		- [JOIN한 테이블 GROUP 하기](#join한-테이블-group-하기)
		- [SELF JOIN](#self-join)
	- [(LEFT, RIGHT) JOIN](#left-right-join)
	- [CROSS JOIN](#cross-join)
- [UNION](#union)
	- [합집함](#합집함)
	- [교집합](#교집합)
	- [차집합](#차집합)
- [MYSQL 사용](#mysql-사용)
	- [터미널 실행](#터미널-실행)
	- [Workbench 사용하기](#workbench-사용하기)
		- [Schema 생성](#schema-생성)
		- [데이터베이스 삭제 명령어](#데이터베이스-삭제-명령어)
- [테이블 생성 / 수정 / 삭제](#테이블-생성--수정--삭제)
	- [테이블 생성](#테이블-생성)
		- [CREATE TABLE](#create-table)
	- [테이블 변경](#테이블-변경)
		- [ALTER TABLE](#alter-table)
		- [DROP TABLE](#drop-table)
	- [데이터 삽입](#데이터-삽입)
		- [INSERT INTO](#insert-into)
	- [테이블 생성시 제약 넣기](#테이블-생성시-제약-넣기)
- [자료형](#자료형)
	- [숫자 자료형](#숫자-자료형)
		- [고정 소수점](#고정-소수점)
		- [부동 소수점](#부동-소수점)
	- [문자 자료형](#문자-자료형)
		- [문자열](#문자열)
		- [텍스트](#텍스트)
	- [시간 자료형](#시간-자료형)
- [데이터 (변경 / 삭제) 하기](#데이터-변경--삭제-하기)
	- [DELETE](#delete)
		- [주어진 조건의 행 삭제하기](#주어진-조건의-행-삭제하기)
		- [전체 행 삭제하기](#전체-행-삭제하기)
	- [TRUNCATE](#truncate)
		- [테이블 초기화](#테이블-초기화)
	- [UPDATE](#update)
		- [주어진 조건의 행 수정하기](#주어진-조건의-행-수정하기)
		- [여러 컬럼 수정하기](#여러-컬럼-수정하기)
		- [컬럼 데이터 활용하여 수정하기](#컬럼-데이터-활용하여-수정하기)
		- [****조건문 없이는 모든 행 변경****](#조건문-없이는-모든-행-변경)


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
| FALSE | 거짓 |
| NOT, ! | 반대 |
| AND | 모두 |
| OR | 한 쪽만 |
- <> 는 ≠ 과 같다

| BETWEEN [MIN] AND [MAX] | 두 값 사이에 있다. |

```sql
select 5 between 2 and 10;
-- 1      <- true
```

| LIKE’…%…’ | 0~N개 문자를 가진 패턴 |
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
| CEIL | 올림 |
| FLOOR | 내림 |

```sql
select round(0.012);
-- 0
```

| ABS | 절대값 |

```sql
select abs(5.3);
-- 5
```

| GREATEST | 괄호 안의 가장 큰 값 |
| LEAST | 괄호 안의 가장 작은 값 |

```sql
select greatest(1, 5, 7);
-- 7
select least(6, 2, 1);
-- 1
```

| MAX | 최대값 |
| MIN | 최소값 |
| COUNT | 갯수 |
| SUM | 합 |
| AVG | 평균 |

| POW(A, B), POWER(A, B) | A를 B만큼 제곱 |
| SQRT | 제곱근 |

```sql
select
	pow(2, 3) # 8
	sort(16) # 4
```

| TRUNCATE(N,n) | N을 소숫점 n자리까지 선택 |

```sql
select
	truncate(3.1415, 2)
-- 3.14
```

## 문자열 관련

| UCASE, UPPER | 모두 대문자로 |
| LCASE, LOWER | 모두 소문자로 |

```sql
select
	upper("hello");
-- HELLO
```

| CONCAT(...) | 괄호 안의 내용 이어붙임 |
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
| CHAR_LENGTH, CHARACTER_LEGNTH | 문자열의 문자 길이 |
- CHAR_LENGTH, CHARACTER_LEGNTH 이것이 흔히 아는 len()이다.
- LENGTH는 한글에서 바이트 길이가 나오기 때문
    
    
    | TRIM | 양쪽 공백 제거 |
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
| RPAD(S, N, P) | S가 N글자가 될 때까지 P를 이어붙임 |

```sql
select
	rpad('abc', 5, '__');

-- abc__
```

| REPLACE(S, A, B) | S중 A를 B로 변경 |

```sql
select
	replace('맥도날드에서 햄버거를 먹었다', '맥도날드', '버거킹');

-- 버거킹에서 햄버거를 먹었다
```

| INSTR(S, s) | S중 s의 첫 위치 반환, 없을 시 0 |

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
| CONVERT(A, T) | A를 T 자료형으로 변환 |

```sql
select
	'01' = '1'
	convert('01', decimal), = convert('1', decimal);
 
-- 1
```

## 시간 관련

| CURRENT_DATE, CURDATE | 현재 날짜 반환 |
| CURRENT_TIME, CURTIME | 현재 시간 반환 |
| CURRENT_TIMESTAMP, NOW | 현재 시간과 날짜 반환 |

```sql
select
	now();

--2023-08-07 12:01:37
```

| DATE | 문자열에 따라 날짜 생성 |
| TIME | 문자열에 따라 시간 생성 |

```sql
SELECT
  '2021-6-1' = '2021-06-01',   # 0
  DATE('2021-6-1') = DATE('2021-06-01'),   # 1
  '1:2:3' = '01:02:03',  # 0
  TIME('1:2:3') = TIME('01:02:03');  # 1

-- date, time 함수는 날짜, 시간으로 문자를 받는다.
```

| YEAR | 주어진 DATETIME값의 년도 반환 |
| MONTHNAME | 주어진 DATETIME값의 월(영문) 반환 |
| MONTH | 주어진 DATETIME값의 월 반환 |
| WEEKDAY | 주어진 DATETIME값의 요일값 반환(월요일: 0) |
| DAYNAME | 주어진 DATETIME값의 요일명 반환 |
| DAYOFMONTH, DAY | 주어진 DATETIME값의 날짜(일) 반환 |

---

| HOUR | 주어진 DATETIME의 시 반환 |
| MINUTE | 주어진 DATETIME의 분 반환 |
| SECOND | 주어진 DATETIME의 초 반환 |

```sql
select
	hour(now()), minute(now());
```

| ADDDATE, DATE_ADD | 시간/날짜 더하기 |
| SUBDATE, DATE_SUB | 시간/날짜 빼기 |

시간/날짜 끼리의 덧,뺄셈이 가능한 함수이다.

---

| DATE_DIFF | 두 시간/날짜 간 일수차 |
| TIME_DIFF | 두 시간/날짜 간 시간차 |

| LAST_DAY | 해당 달의 마지막 날짜 |

### DATE_FORMAT

- 시간/날짜를 지정한 형식으로 반환

| %Y | 년도 4자리 |
| %y | 년도 2자리 |
| %M | 월 영문 |
| %m | 월 숫자 |
| %D | 일 영문(1st, 2nd, 3rd...) |
| %d, %e | 일 숫자 (01 ~ 31) |
| %T | hh:mm:ss |
| %r | hh:mm:ss AM/PM |
| %H, %k | 시 (~23) |
| %h, %l | 시 (~12) |
| %i | 분 |
| %S, %s | 초 |
| %p | AM/PM |

```sql
select
	date_format(now(), "%Y년 %m월 %d일")
```

| STR _ TO _ DATE(S, F) | S를 F형식으로 해석하여 시간/날짜 생성 |

str -> date형식으로 생성 가능하다.

## 기타 함수

| IF(조건, T, F) | 조건이 참이라면 T, 거짓이면 F 반환 |

```sql
select
	if(1 > 2, "참이다", "거짓이다")

# 거짓이다.
```

```sql
select
case
	when -1 > 0 then "-1은 양수이다."
	when -1 = 0 then "-1은 0이다."
	else "-1은 음수다."
end;
```

| IFNULL(A, B) | A가 NULL일 시 B 출력 |

```sql
select
	ifnull('A', 'B')
	ifnull(NULL, 'B')
```

# 그룹

- 다음 함수들은 group by에서 사용된다.

| MAX | 가장 큰 값 |
| MIN | 가장 작은 값 |
| COUNT | 갯수 (NULL값 제외) |
| SUM | 총합 |
| AVG | 평균 값 |

## GROUP BY

```sql
select Country from Customers
group by country;

-- 같은 국가끼리 묶고 선택
```

### 활용

```sql
select
	count(*), OrderDate
from Orders
group by OrderDate;

-- Orders 테이블에서 각 OrderDate의 개수를 가져오겠다.
```

## WITH ROLLUP

- 전체의 집계값
    
    ORDER BY와 함께 사용 불가
    

```sql
select
	Country, count(*)
from Suppliers
group by Country
with rollup; # count* 값 이후, sum(count*)값이 하나 추가됨.
```

## HAVING

- 그룹화된 데이터를 걸러낸다.(Like if)
- ** WHERE은 그룹하기 전 데이터, HAVING은 그룹한 후 집계에 사용

```sql
select
	Country, count(*) as count
from Suppliers
group by Country
having count >= 3;

-- 3이상만 선택된다.
```

## DISTINCT

- 중복된 값들을 제거
- 집계함수와는 같이 사용 불가

```sql
select distinct CategoryID
from Products;

--기본적인 정렬이 안됨
--ORDER BY 를 사용해 수동으로 정렬 가능

select distinct CategoryID
from Products
order by CategoryID
```

### GROUP BY, DISTINCT 함께 사용

```sql
select
	Country,
	count(distinct CITY)
from Customers
group by Country;
```

# 쿼리 안에 서브쿼리

## 비상관 서브쿼리

```sql
select * from Products
where Price < (
	select avg(Price) from Products
);

-- select avg(Price) from Products 이 쿼리 값은 28.866~ 값을 가지게 되는데 이 값보다 작은 값만 선택하겠다 라는 뜻
```

```sql
select
	SupplierID, Unit
from Products
where
	SupplierID in
	(SELECT SupplierID FROM Products where SupplierID > 3);
```

| ~ ALL | 서브쿼리의 모든 결과에 대해 ~하다 |
| ~ ANY | 서브쿼리의 하나 이상의 결과에 대해 ~하다 |

```sql
select * from Products
where Price > ALL (
	select Price from Products
    where CategoryID = 2
);

-- CatrgoryID 가 2인 상품의 가격들 보다 비싼 상품들 출력
-- MAX(Price(ID=2 인 값 그 이상을 출력한다는 뜻
-- any라면 min이 되겠죠?
```

## 상관 서브쿼리

```sql
select
	ProductID, ProductName,
    (
    select CategoryName from Categories as C
    where C.CategoryID = P.CategoryID
    ) as CategoryName
from Products as P;

-- Categories의 Name과 Products의 Name이 일치할 때 그것을 CategoryName이라고 부르고
-- ProductID와 ProductName과 CategoryName 출력
```

## (NOT)EXISTS

```sql
select
	CategoryID, CategoryName
from Categories as C
where exists (
	select * from Products as P
    where P.CategoryID = C.CategoryID
    and P.Price > 80
);

-- CategoryID의 Price가 80이 넘는 것이 있는가?를 출력
```

# JOIN

## JOIN - INNERJOIN(기본값)

- on : 어떤 컬럼을 기준으로 ?

```sql
select * from Categories as C
join Products as P
	on C.CategoryID = P.CategoryID;

-- C.CategoryID 와 P.CategoryID 가 동일한 것을 기준으로 병합하겠다.
```

- JOIN할 때 ambiguous를 주의해야 한다.
    
    CategoryID는 두개이므로(같은 컬럼이 있는 경우) 이를 구별할 수 있는 P. or C. 등으로 별칭을 사용해 테이블의 컬럼을 정의 해야함
    

### JOIN한 테이블 GROUP 하기

```sql
SELECT 
  C.CategoryName,
  MIN(O.OrderDate) AS FirstOrder,
  MAX(O.OrderDate) AS LastOrder,
  SUM(D.Quantity) AS TotalQuantity
FROM Categories C
JOIN Products P 
  ON C.CategoryID = P.CategoryID
JOIN OrderDetails D
  ON P.ProductID = D.ProductID
JOIN Orders O
  ON O.OrderID = D.OrderID
GROUP BY C.CategoryID;
```

### SELF JOIN

- 같은 테이블 끼리도 가능하다.

## (LEFT, RIGHT) JOIN

- NULL값을 채워준다.

## CROSS JOIN

모든 경우의 수를 확인할 수 있다.

- 조건 없이 모든 조합 반환 A*B

```sql
select
	E1.LastName, E2.FirstName
from Employees as E1
cross join Employees as E2
order by E1.EmployeeID;

-- 모든 조합이 나온다.
```

# UNION

| UNION | 중복을 제거한 집합 |
| UNION ALL | 중복을 제거하지 않은 집합 |

## 합집함

```sql
select CustomerName as Name, City, Country, 'CUSTOMER'
from Customers
union
select SupplierName as Name, City, Country, 'SUPPLIER'
from Suppliers;

-- 데이터들이 union으로 인해 합쳐진다.
```

## 교집합

```sql
select CategoryID as ID
from Categories as C ,Employees as E
where
	C.CategoryID > 4
	and E.EmployeeID % 2 = 0
	and C.CategoryID = E.EmployeeID;
```

## 차집합

```sql
select CategoryID as ID
from Categories
where
	C.CategoryID > 4
	and CategoryID not in (
	select Employees
	from Employees
	where EmployeeID % 2 = 0
	);
```

# MYSQL 사용

## 터미널 실행

• `mysql.server start`

## Workbench 사용하기

### Schema 생성

```sql
create schema schema_name collate utf8mb4_general_ci;
```

| utf8mb4 | 한글을 포함한 전세계 문자 + 이모티콘 사용 가능 |
| utf8mb4 _ general _ ci | 가장 정확하지는 않지만 정렬 속도 빠름 |

### 데이터베이스 삭제 명령어

```sql
drop database 'schema_name';
```

# 테이블 생성 / 수정 / 삭제

## 테이블 생성

### CREATE TABLE

```sql
create table people (
    person_id int,
    person_name varchar(10),
    age tinyint,
    birthday date
);

-- people 이라는 TABLE 생성됨.
```

## 테이블 변경

### ALTER TABLE

```sql
alter table people rename to friends,  -- 테이블 변경
change column person_id person_id tinyint, -- 컬럼 자료형 변경
change column person_name person_nickname varchar(10),  -- 컬럼명 변경
drop column birthday,  -- 컬럼 삭제
add column is_married tinyint after age; -- 컬럼 추가
```

### DROP TABLE

```sql
drop table friends; -- 테이블 삭제
```

## 데이터 삽입

### INSERT INTO

```sql
insert into people
(person_id, person_name, age, birthday)
VALUES
  (1, '홍길동', 21 , '2000-01-31')
	(2, '황시훈', 24 , '2000-06-20')

```

```sql
insert into people
VALUES
	(2, '전우치', 18 , '2003-05-12') -- 모든 컬럼명에 넣을 경우 컬럼명을 생략할 수 있다.
```

```sql
insert into people
(person_id, person_name, age, birthday)
VALUES
  (1, '홍길동', '2000-01-31') -- 자료형에 맞지 않으면 오류가 발생하고, 그저 생략했을 시 NULL값 형성

```

## 테이블 생성시 제약 넣기

| AUTO_INCREMENT | 새 행 생성시마다 자동으로 1씩 증가 |
| PRIMARY KEY | 중복 입력 불가, NULL(빈 값) 불가 |
| UNIQUE | 중복 입력 불가 |
| NOT NULL | NULL(빈 값) 입력 불가 |
| UNSIGNED | (숫자일시) 양수만 가능 |
| DEFAULT | 값 입력이 없을 시 기본값 |

**PRIMARY KEY**

- 테이블마다 하나만 가능

```sql
create table people (
    person_id int auto_increment primary key, -- 그래서 행마다 값을 넣을 필요가 없다.
    person_name varchar(10) not null,
    nickname varchar(10) unique not null,
    age tinyint unsigned,
    is_married tinyint default 0
)
```

# 자료형

## 숫자 자료형

| 정수형 | 바이트 | SIGNED |
| TINYINT | 1 | -128 ~ 127 |
| SMALLINT | 2 | -32,768 ~ 32,767 |
| MEDIUMINT | 3 |  |
| INT | 4 |  |
| BIGINT | 8 |  |

### 고정 소수점

| DECIMAL( s, d ) | 실수 부분 자릿수( s ) , 소수 부분 자릿수 ( d ) |

### 부동 소수점

| FLOAT |
| DOUBLE |

## 문자 자료형

### 문자열

| CHAR | 고정 사이즈 | s | 최대 바이트 255 |
| VARCHAR | 가변 사이즈 | 실제 글자 수 + 1  | 최대 바이트 65535 |
- 검색시 CHAR가 더 빠름
- VARCHAR 컬럼 길이값이 4글자보다 적을 경우 CHAR로 자동 변환
- CHAR는 강제로 4바이트를 차지 but, VARCHAR는 글자 수 값 대로 바이트를 차지(실제 길이대로)

### 텍스트

| TINYTEXT | 255 |
| TEXT | 65,535 |
| MEDIUMTEXT |  |
| LONGTEXT |  |

## 시간 자료형

| DATE | YYYY-MM-DD |  |
| TIME | HHH:MI:SS |  |
| DATETIME | YYYY-MM-DD HH:MI:SS | 입력된 시간을 그 값 자체로 저장 |
| TIMESTAMP | YYYY-MM-DD HH:MI:SS | 컴퓨터의 시간대를 기준으로 저장 |
- 보통적인 상황인 경우 DATETIME
- 국제적인 경우 TIMESTAMP 사용

# 데이터 (변경 / 삭제) 하기

| DELETE | 행 삭제 |
| TRUNCATE | 초기화 |
| UPDATE | 행 수정 |

## DELETE

### 주어진 조건의 행 삭제하기

```sql
delete from businesses
where status = 'CLS';
```

### 전체 행 삭제하기

```sql
delete from businesses;
```

- delete로 지우고 새로 데이터를 추가하면  index번호는 그 전 데이터 인덱스에 이어서 자동으로 채워짐 (ex 1, 2, 3 (X) → 18, 19, 20 (O)

## TRUNCATE

### 테이블 초기화

```sql
 truncate businesses;
```

- delete와 다르게 새로운 데이터를 추가하면 index번호는 처음부터 1, 2, 3 ‘’’ 이다.

## UPDATE

### 주어진 조건의 행 수정하기

```sql
update menus
set menu_name = '삼선짜장'
where menu_id = 12;

	--12번 항목의 메뉴이름을 삼선짜장으로 변경
```

### 여러 컬럼 수정하기

```sql
update menus
set 
    menu_name = '열정떡볶이', 
    kilocalories = 492.78,
    price = 5000
where
    fk_business_id = 4
    and menu_id = '국물떡볶이';
```

### 컬럼 데이터 활용하여 수정하기

```sql
update menus
set price = price + 1000
where fk_business_id = 8;
```

### ****조건문 없이는 모든 행 변경****

```sql
update menus
set menu_name = '획일화';
```

WHERE문을 넣어 조건문을 사용하자.
