# C언어_포인터

# 포인터

- 포인터는 주소를 저장하기 위한 메모리 공간이다
- 포인터의 크기는 모두 4바이트이다.
- 포인터 변수로 값에 접근할 때는 *연산자를 사용한다
- 2차원 포인터 변수는 1차원 포인터 변수의 주소를 저장하기 위한 메모리 공간이다
- n차원 포인터는 *연산자를 n개 붙여 값을 접근한다.

## 포인터의 선언

- 포인터 : 주소를 저장하는 변수
    - 4바이트

```python
int char c = 'A';
int char *cp = &c;
#-> cp = &c;

*cp = 'B';

printf("%d%d", c, *cp);

# BB
# 여기서 char *cp -> c의 주소를 가리켜라 라는 선언
# *cp = 'B' -> 포인터가 가리키는 위치에 'B'라는 값을 저장하라
```

## 다차원 포인터

```python
#include <stdio.h>

int main(void)
{
    char c = 'A';
    char *cp;
    char **cpp;

    cp = &c;
    cpp = &cp;

    printf("%c%c%c\n", c, *cp, **cpp);

    return 0;
}

'''
AAA
'''
```

---

## 포인터 가. 감산

```python
#include <stdio.h>

int main(void)
{
    char n = 20;
    char *np;
    char **npp;

    np = &n;
    npp = &np;

    printf("%d %d %d\n", n, np, npp);
    printf("%d %d %d \n", n+1, np+1, npp+1);

    return 0;
}

'''
20 1802925691 1802925680
21 1802925692 1802925688
'''            # 2차원주소만큼 건너 뛰겠다(np -> npp) (+4+4)
```
