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
