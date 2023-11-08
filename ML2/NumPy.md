# NumPy

## Numpy

`import numpy as pd`

- **ndarray**
    - 배열 구조를 뜻한다.

```python
array1 = np.array([1, 2, 3])
print('array1 type:', type(array1))
print('array1 array 형태 :', array1.shape)

array2 = np.array([[1, 2, 3],
                  [2, 3, 4]])
print('array2 type:', type(array1))
print('array2 array 형태 :', array2.shape)

array3 = np.array([[1, 2, 3]])
print('array3 type:', type(array1))
print('array3 array 형태 :', array3.shape)

'''
array1 type: <class 'numpy.ndarray'>
array1 array 형태 : (3,)
array2 type: <class 'numpy.ndarray'>
array2 array 형태 : (2, 3)
array3 type: <class 'numpy.ndarray'>
array3 array 형태 : (1, 3)
'''
```

- array의 차원 확인하기
    - `.ndim`

```python
print(array1.ndim, array2.ndim, array3.ndim)

'''
1 2 2
'''
```

- ndarray의 데이터 타입 확인

```python
list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)

'''
<class 'list'>
<class 'numpy.ndarray'>
[1 2 3] int64
'''
```

- array 자료형 변환
    - `.astype(’자료형’)`

```python
array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

'''
[1. 2. 3.] float64
```

- ndarray를 생성
    - `arange()`
    - `zeros()`
    - `ones()`

```python
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

'''
[0 1 2 3 4 5 6 7 8 9]
int64 (10,)
'''
```

```python
zero_array = np.zeros((3, 2), dtype ='int64')
print(zero_array)
print(zero_array, zero_array.shape)

print()

one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)

'''
[[0 0]
 [0 0]
 [0 0]]
[[0 0]
 [0 0]
 [0 0]] (3, 2)

[[1. 1.]
 [1. 1.]
 [1. 1.]]
float64 (3, 2)
'''
```

- ndarray의 차원과 크기를 변경
    - `reshape()`

```python
array1 = np.arange(10)
print('array1 : \n', array1)

array2 = array1.reshape(2, 5)
print('array2 : \n', array2)

array3 = array1.reshape(5, 2)
print('array3 : \n', array3)

'''
array1 : 
 [0 1 2 3 4 5 6 7 8 9]
array2 : 
 [[0 1 2 3 4]
 [5 6 7 8 9]]
array3 : 
 [[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
'''
```

지정하는 사이즈로 변경이 불가능하면 오류를 발생한다.

- -1을 인자로 사용하면 자동으로 ndarray에 호환되는 shape으로 변환해준다.

```python
array1 = np.arange(10)
print(array1)
print()
array2 = array1.reshape(-1, 5)
print(array2)

'''
[0 1 2 3 4 5 6 7 8 9]

[[0 1 2 3 4]
 [5 6 7 8 9]]
**********'''**********
```

- 리스트 자료형으로 변환
    - `tolist()`

```python
array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array3d:\n', array3d.tolist())

'''
array3d:
 [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
'''
```

- 데이터 추출
    - `[row, col]`

```python
array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)
print(array2d)

print(array2d[1, 0])

'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
4
'''
```

- 팬시인덱싱(Fancy indexing)
    
    지정한 위치부터 인덱싱해준다.
    

```python
array = array2d[[0, 1], 2]
print('array2d[[0, 1], 2] =>', array3.tolist())

'''
array2d[[0, 1], 2] => [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
'''
```

- 불린 인덱싱
    
    조건 필터링과 검색을 동시에 할 수 있다.
    

```python
array1d = np.arange(1, 10)
array3 = array1d[array1d > 5]
print(array3)
array_bool = array1d > 5
print(array_bool)

'''
[6 7 8 9]
[False False False False False  True  True  True  True]
'''
```

- 행렬 정렬
    - sort()
    - argsort()

```python
org_array = np.array([3, 1, 9, 5])
print('원본 행렬:', org_array)
sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬', sort_array1)
print('np.sort() 호출 후 원본 행렬 :', org_array)
sort_array2 = org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬:', sort_array2)
print('org_array.sort() 호출 후 원본 행렬:', org_array)

'''
원본 행렬: [3 1 9 5]
np.sort() 호출 후 반환된 정렬 행렬 [1 3 5 9]
np.sort() 호출 후 원본 행렬 : [3 1 9 5]
org_array.sort() 호출 후 반환된 행렬: None
org_array.sort() 호출 후 원본 행렬: [1 3 5 9]
'''
```

- 내림차순 정렬
    - np.sort(array)[::-1]

---

- 2차원 행렬 정렬

```python
array2d = np.array([[8, 12], 
                    [7, 1]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)
sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)
# axis=0 : 로우
# axis=1 : 컬럼

'''
로우 방향으로 정렬:
 [[ 7  1]
 [ 8 12]]
컬럼 방향으로 정렬:
 [[ 8 12]
 [ 1  7]]
'''
```

- 정렬된 행렬의 인덱스 반환
    - argsort()

```python
org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)

'''
<class 'numpy.ndarray'>
행렬 정렬 시 원본 행렬의 인덱스: [1 0 3 2]
'''
```

- ex

```python
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스 :', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력 :', name_array[sort_indices_asc])

'''
성적 오름차순 정렬 시 score_array의 인덱스 : [0 2 4 1 3]
성적 오름차순으로 name_array의 이름 출력 : ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']
'''
```

### 선형대수 연산

- 행렬내적
    - np.dot()

```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
B = np.array([[7, 8], 
              [9, 10], 
              [11, 12]])
dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)

'''
행렬 내적 결과:
 [[ 58  64]
 [139 154]]
'''
```

- 전치행렬
    - np.transpose()

```python
A = np.array([[1, 2], 
              [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)

'''
A의 전치 행렬:
 [[1 3]
 [2 4]]
'''
```
