# ML2

# 머신러닝의 이해

## Numpy

`import numpy as pd`

- **ndarray**

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
