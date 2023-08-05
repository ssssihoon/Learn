
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
