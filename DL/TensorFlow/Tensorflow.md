# Tensorflow

## 상수

- tf.constant() : 상수 선언
    - 값을 변경할 수 없다.

```python
tensor_20 = tf.constant([[23, 4], [32, 51]])
tensor_20
'''
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[23,  4],
       [32, 51]], dtype=int32)>
'''
```

## 변수

- tf.Variable() : 변수 선언
    - 값을 변경할 수 있다. → assign(값)

```python
tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
tf2_variable
'''
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
'''

tf2_variable[0, 2].assign(100)
tf2_variable
'''
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[  1.,   2., 100.],
       [  4.,   5.,   6.]], dtype=float32)>
'''
```
