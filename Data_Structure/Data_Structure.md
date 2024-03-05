# 자료구조

프로그램 = 자료구조 + 알고리즘

# 자료구조와 알고리즘

## 알고리즘의 조건

- 입력 : 0개 이상의 입력이 존재하여야 함.
- 출력 : 1개 이상의 출력이 존재하여야 한다.
- 명백성 : 각 명령어의 의미는 모호하지 않고 명확해야 한다.
- 유한성 : 한정된 수의 단계 후에는 반드시 종료되어야 한다.
- 유효성 : 각 명령어들은 실행 가능한 연산이여야 한다.

## 알고리즘의 기술 방법

- 영어나 한국어와 같은 자연어
- 흐름도
- 의사 코드
- 프로그래밍 언어

## 추상 데이터 타입 ADT

- 데이터 타입을 추상적(수학적)으로 정의한 것
- 데이터나 연산이 무엇인가는 정의되지만 데이터나 연산을 어떻게 컴퓨터 상에서 구현할 것인지는 정의되지 않는다.

## 수행시간 측정

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(void)
{
    clock_t start, stop;
    double duration;
    start = clock();
    for (int i = 0; i < 1000000; i++)
        ;
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("수행시간은 %f초입니다.\n", duration);
    return 0;
}

//수행시간은 0.002339초입니다.
```

# 순환

- 재귀함수

## 팩토리얼 프로그래밍

```c
# include <stdio.h>

int factorial(int n);

int main(void)
{
    int n;
    int result;

    printf("양수를 입력하세요 : ");

    scanf("%d", &n);
    printf("%d의 팩토리얼은 %d입니다.", n, factorial(n));

}

int factorial(int n)
{
    if(n == 0)
    {
        return 1;
    }
    else
    {
        return n * factorial(n-1);
    }
}
```

## 피보나치

- 순환 호출을 하는 경우의 비효율성
    - 같은 항이 중복해서 계산됨.
    - 이러한 현상은 n이 커지면 더 심해짐.

```c
#include <stdio.h>

int fibo(int n);

int main(void)
{
    int n;
    int i;

    printf("피보나치 수열 n 값 입력 : ");

    scanf("%d", &n);

    for(i = 0; i < n; i ++)
    {
        printf("%d ", fibo(i));
    }
}

int fibo(int n)
{
    if (n == 0)
    {
        return 0;
    }
    else if (n == 1)
    {
        return 1;
    }
    else
    {
        return fibo(n-2) + fibo(n-1);
    }
}

/*
피보나치 수열 n 값 입력 : 10
0 1 1 2 3 5 8 13 21 34 
*/
```

## 하노이탑

- A, B, C 막대가 있는 상태이며, A에 원판이 쌓여있음.
- 원형으로 A, B, C가 놓여있다고 가정.
- 시계방향으로 홀수 번째의 원판을 이동(A→ B)
- 반시계방향으로 짝수 번째의 원판을 이동(A→ C)
    - 시계 반시계 방향을 B기준으로 똑같이 반복
    - 시계 반시계 방향을 C기준으로 똑같이 반복

![Untitled](%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%2033103380d5b743969ee167c107065a69/Untitled.png)

```c
# include <stdio.h>

// 막대 A에 쌓여있는 n개의 원판을 막대 B를 사용하여 막대 C로 옮김

void hanoi_tower(int n, char A, char B, char C);

int main(void)
{

    int n;

    scanf("%d", &n);

    hanoi_tower(n, 'A', 'B', 'C');

    return 0;
}

void hanoi_tower(int n, char A, char B, char C)
{
    if (n == 1)
    {
        printf("원판 1을 %c에서 %c으로 옮긴다.\n", A, C);
    }
    else
    {
        hanoi_tower(n-1, A, C, B);
        printf("원판 %d을 %c에서 %c으로 옮긴다.\n", n, A, C, B);
        hanoi_tower(n-1, B, A, C);
    }
}
```
