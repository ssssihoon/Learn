# NumPy

## 넘파이 배열 생성

```python
list1 = [1, 2, 3, 4]
array1 = np.array(list1) # list를 array 형식으로 변환
print(array1) # 1차원 배열

'''
[1, 2, 3, 4]
'''
```

```python
list2 = np.array([[1, 2, 3], [4, 5, 6]])
print(list2) # 2차원 배열

'''
[[1 2 3]
 [4 5 6]]
'''
```

- `array()` : 리스트를 배열로 변환
- `arange()` : 파이썬의 range함수와 유사
- `ones()` : 1로 채운 n차원 배열 생성
- `zeros()` : 0으로 채운 n차원 배열 생성
- `empty()` : 초기화 하지 않은 빈 n차원 배열을 생성
- `eye() or identify()` : 대각선 요소에 1을 채우고 그 외에는 0으로 채원 2차원 배열 생성
- `linspace()` : 초깃값부터 최종값까지 지정한 간격의 수를 채워 배열 생성
- `full()` : 지정한 모양에 지정한 값으로 채운 배열 생성

```python
array4, step = np.linspace(1, 10, 5, retstep = True)
print(array4)
print(step)

'''
[ 1.    3.25  5.5   7.75 10.  ]
2.25
'''
```

```python
a = np.zeros(2)
print('a\n', a)
'''
a[0. 0.]
'''

b = np.zeros((2, 2))
print('b\n', b)
'''
b[[0. 0.]
 [0. 0.]]
'''

c = np.ones((2, 3))
print('c\n', c)
'''
c[[1. 1. 1.]
 [1. 1. 1.]]
'''

d = np.full((2, 3), 5)
print('d\n', d)
'''
d[[5 5 5]
 [5 5 5]]
'''

e = np.eye(3)
print('e\n', e)
'''
e[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''
```

### 배열 자료형 생성, 변환

```python
a = np.array([1, 2], dtype = np.float64) #float형식으로 생성
print(a.dtype)

b = a.astype(np.int8) #int형식으로 변환
print(b)
```

속성

- `ndim` : 배열의 차원을 나타냄
- `itemsize` : 요소의 바이트 수
- `size` : 요소의 개수
- `nbytes` : 배열 전체의 바이트 수
- `shape` : 배열의 모양

배열의 모양을 변경하는 함수

- `flatten()` : 1차원 배열로 변경
- `resize(i, j)` : 배열의 모양을 i x j로 변경
- `transpose() or T` : 열과 행을 교차
- `resize()` : 튜플을 입력하면 배열의 모양을 변경

```python
array1 = np.array([1, 2, 3, 4])
array1.shape = (2, 2)

'''
[[1 2]
 [3 4]]
'''

```

랜덤 함수

- np,random.randn()
    
    ```python
    print(np.random.randn(2, 3))
    '''
    [[-0.11325588  0.01299004 -1.34583374]
     [ 1.07067319  0.90566715 -0.70917983]]
    '''
    
    print(np.random.randint(-10, 10, size = [2, 3]))
    '''
    [[  0   9   1]
     [ -6 -10   9]]
    '''
    
    print(np.random.randint(0, 10, size = [2, 3]))
    #음의 값 제외
    '''
    [[1 3 3]
     [4 0 6]]
    '''
    ```
    

## 넘파이 배열 다루기

### 마스킹

- 논리값 인덱싱이라고도 부름 , 조건에 맞는 값을 출력

```python
import numpy as np

array1 = np.random.randint(0, 10, size = [4, 2])
print(array1, '\n')

mask = np.array([0, 1, 1, 0], dtype = bool)
print(mask, '\n')

#array1에 mask 적용
print(array1[mask])

'''
[[6 4]
 [3 4]
 [5 5]
 [2 1]] 

[False  True  True False] 

[[3 4]
 [5 5]]
'''
```

### 넘파이 연산 함수 (유니버셜 함수)

`np.함수명(배열명)`

- abs() : 원소의 절댓값 반환
- sqrt() : 원소의 제곱근 반환
- square() : 원소의 제곱 반환
- exp() : 원소의 지수 반환
- log() : 원소의 밑이 e인 로그를 취해 반환
- add() : 두 배열 원소의 합을 반환
- substract() : 두 배열 원소의 차를 반환
- multiply() : 두 배열 원소의 곱을 반환
- divide() : 두 배열 원소의 나눗셈 결과를 반환
- floor_divide() : 나눗셈의 정수 몫을 반환
- mod() : 두 배열 원소 나눗셈의 정수 나머지를 반환

```python
import numpy as np

arr = np.random.rand(1, 10)
print(arr, '\n')

print(np.sqrt(arr))

'''
[[0.90328401 0.1797699  0.06258423 0.81057841 0.90748395 0.42268738
  0.24984986 0.92772145 0.27891094 0.84246104]] 

[[0.95041255 0.42399281 0.25016841 0.90032128 0.95261952 0.65014412
  0.49984984 0.96318298 0.5281202  0.91785676]]
'''
```

### 배열 복사

- 얕은 복사
    - b = a
- 깊은 복사
    - `copy()`
    - b = a.copy()

### 배열 정렬

- np.sort(배열) : 배열을 정렬, 원본 유지
- 배열.sort() : 배열 정럴, 원본 정렬
- np.argsort(배열) : 정렬된 배열의 원래 인덱스 반환
