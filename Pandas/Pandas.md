# Pandas

# 데이터 집합 불러오기

```python
import pandas as pd

df = pd.read_csv('/Users/sihoon/Desktop/Pandas/gapminder.tsv', sep='\t')
print(df.head())

'''
       country continent  year  lifeExp       pop   gdpPercap
0  Afghanistan      Asia  1952   28.801   8425333  779.445314
1  Afghanistan      Asia  1957   30.332   9240934  820.853030
2  Afghanistan      Asia  1962   31.997  10267083  853.100710
3  Afghanistan      Asia  1967   34.020  11537966  836.197138
4  Afghanistan      Asia  1972   36.088  13079460  739.981106
'''
```

- sep 의 속성값으로 \t를 지정했다. : 열이 탭으로 구분되어 있다라는 것!
- df.head() : 상위 5개 행
- df.tail() : 하위 5개 행

## 시리즈와 데이터프레임

- 시리즈 : 시트의 열 1개
- 데이터 프레임 : 딕셔너리와 같다

```python
import pandas as pd

df = pd.read_csv('/Users/sihoon/Desktop/Pandas/gapminder.tsv', sep='\t')
print(type(df))
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())

'''
<class 'pandas.core.frame.DataFrame'>

(1704, 6)

Index(['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap'], dtype='object')

country       object
continent     object
year           int64
lifeExp      float64
pop            int64
gdpPercap    float64
dtype: object

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1704 entries, 0 to 1703
Data columns (total 6 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   country    1704 non-null   object 
 1   continent  1704 non-null   object 
 2   year       1704 non-null   int64  
 3   lifeExp    1704 non-null   float64
 4   pop        1704 non-null   int64  
 5   gdpPercap  1704 non-null   float64
dtypes: float64(2), int64(2), object(2)
memory usage: 80.0+ KB
None
'''
```

- type() : 자료형을 출력해준다.
- df.shape : 행과 열의 크기를 알려 준다.
- df.columns : 데이터프레임의 열 이름을 알려 준다.
- df.dtypes : 데이터프레임의 dtypes 속성을 알려 준다.
- df.info() : 데이터프레임의 info 메서드를 알려 준다.

## 판다스 자료형

| 판다스 자료형 | 파이썬 자료형 | 설명 |
| --- | --- | --- |
| object | string | 문자열 |
| int64 | int | 정수 |
| float64 | float | 소수점을 가진 숫자 |
| datetime64 | datetime | 파이썬 표준 datetime |

# 데이터 추출하기

## 열 단위 추출

```python
country_df = df['country']
print(country_df.head())

'''
0    Afghanistan
1    Afghanistan
2    Afghanistan
3    Afghanistan
4    Afghanistan
Name: country, dtype: object
'''
```

country라는 컬럼을 추출할 수 있다.

```python
import pandas as pd

df = pd.read_csv('/Users/sihoon/Desktop/Pandas/gapminder.tsv', sep='\t')
subset = df[['country', 'continent', 'year']]
print(type(subset))
print(subset.head())

'''
<class 'pandas.core.frame.DataFrame'>

       country continent  year
0  Afghanistan      Asia  1952
1  Afghanistan      Asia  1957
2  Afghanistan      Asia  1962
3  Afghanistan      Asia  1967
4  Afghanistan      Asia  1972
'''
```

다중 컬럼을 선택할 때는 [[]]의 구조로 선택해야하고, 2개 이상의 열을 추출했기 때문에 시리즈가 아니라 데이터프레임을 얻을 수 있다.

## 행 단위 추출

인덱스는 보통 0 부터 시작하지만, 문자열로 지정할 수도 있다.

-1같은 인덱스에 없는 값을 사용하면 오류가 발생한다.

| 속성 | 설명 |
| --- | --- |
| df.loc[[행], [열]] | 인덱스를 기준으로 행 데이터 추출 |
| df.iloc[[행], [열]] | 행 번호를 기준으로 행 데이터 추출 |

### loc 속성으로 행 데이터 추출하기

```python
print(df.loc[0])

'''
country      Afghanistan
continent           Asia
year                1952
lifeExp           28.801
pop              8425333
gdpPercap     779.445314
Name: 0, dtype: object
'''
```

- 마지막 행 데이터 추출하기

1) shape

```python
number_of_rows = df.shape[0] # 0번째 열의 행 전체의 크기를 알려준다.
last_row_index = number_of_rows - 1 # 그 행 크기 전체(마지막) - 1 => 인덱스 마지막
print(df.loc[last_row_index])

'''
country        Zimbabwe
continent        Africa
year               2007
lifeExp          43.487
pop            12311143
gdpPercap    469.709298
Name: 1703, dtype: object
'''
```

2) tail

```python
print(df.tail(n=1))
'''
       country continent  year  lifeExp       pop   gdpPercap
1703  Zimbabwe    Africa  2007   43.487  12311143  469.709298
'''
```

---

- 다중 선택 추출

```python
print(df.loc[[0, 99, 999]])

'''
         country continent  year  lifeExp       pop    gdpPercap
0    Afghanistan      Asia  1952   28.801   8425333   779.445314
99    Bangladesh      Asia  1967   43.453  62821884   721.186086
999     Mongolia      Asia  1967   51.253   1149500  1226.041130
'''
```

********************************************************************************************************************************************************tail 메서드와 loc 속성은 서로 반환하는 자료형이 다르다.********************************************************************************************************************************************************

| loc | 시리즈 |
| --- | --- |
| tail | 데이터프레임 |

### iloc 속성으로 행 데이터 추출하기

loc 속성은 데이터프레임의 인덱스를 사용해 데이터를 추출

iloc 속성은 데이터 순서를 의미하는 행 번호를 사용하여 데이터를 추출      iloc의 i 를 (int)라고 생각해서 *번호*라고 생각하자생각하자생각

기본적인 인덱스는 loc, iloc 둘 다 상관 없지만, 인덱스의 이름을 바꾼다면 iloc는 고유의 인덱스 번호를 알 때 사용하면 된다.

```python
print(df.iloc[0])

'''
country      Afghanistan
continent           Asia
year                1952
lifeExp           28.801
pop              8425333
gdpPercap     779.445314
Name: 0, dtype: object
'''
# loc , iloc 동일
```

- iloc 속성은 [-1]값을 사용하면 마지막 행 데이터를 추출한다. 그러나 인덱스 자체를 벗어나면 ([10000]) 오류 발생

```python
print(df.iloc[-1])

'''
country        Zimbabwe
continent        Africa
year               2007
lifeExp          43.487
pop            12311143
gdpPercap    469.709298
Name: 1703, dtype: object
'''
```

### 슬라이싱 구문으로 데이터 추출하기

모든 행 (:)의 데이터에 대해 year과 pop 열을 추출하는 방법

```python
subset = df.loc[:, ['year', 'pop']]
print(subset.head())

'''
   year       pop
0  1952   8425333
1  1957   9240934
2  1962  10267083
3  1967  11537966
4  1972  13079460
'''
```

country continent year lifeExp pop gdpPercap 컬럼명이 있다.

```python
subset = df.iloc[:, [2, 4, -1]]
print(subset.head())

'''
   year       pop   gdpPercap
0  1952   8425333  779.445314
1  1957   9240934  820.853030
2  1962  10267083  853.100710
3  1967  11537966  836.197138
4  1972  13079460  739.981106
'''
```

3번 째와 5번 째, 마지막 출력

### range 메서드로 데이터 추출하기

range 메서드는 제너레이터를 반환한다. (반복 가능)

제너레이터는 간단하게 리스트로 변환할 수 있다.

list(range(인덱스1, 인덱스2, 증가값))을 이용하자!

```python
subset = df.iloc[:, list(range(0, 5))]
print(subset.head())

'''
       country continent  year  lifeExp       pop
0  Afghanistan      Asia  1952   28.801   8425333
1  Afghanistan      Asia  1957   30.332   9240934
2  Afghanistan      Asia  1962   31.997  10267083
3  Afghanistan      Asia  1967   34.020  11537966
4  Afghanistan      Asia  1972   36.088  13079460
'''
```

---

[:3] 과 list(range(0, 3))의 결과값은 동일하다

하지만 파이썬 슬라이싱 구문을 더 선호한다. → [:3]

## 기초적인 통계 계산하기

lifeExp 열을 연도별로 그룹화하여 평균 계산

그 중 lifeExp만 추출하고

mean()을 이용해 평균값 계산

```python
grouped_year_df = df.groupby('year')
grouped_year_df_lifeEXP = grouped_year_df['lifeExp']
print(grouped_year_df_lifeEXP.mean())

'''
year
1952    49.057620
1957    51.507401
1962    53.609249
1967    55.678290
1972    57.647386
1977    59.570157
1982    61.533197
1987    63.212613
1992    64.160338
1997    65.014676
2002    65.694923
2007    67.007423
Name: lifeExp, dtype: float64
'''

======

print(df.groupby('year')['lifeExp'].mean())
```

### 그룹화한 데이터 개수 세기

통계에서는 **************************************빈도수************************************** 라고 한다.

**nunique** 메서드를 사용하면 된다.

```python
print(df.groupby('continent')['country'].nunique())
'''
continent
Africa      52
Americas    25
Asia        33
Europe      30
Oceania      2
Name: country, dtype: int64
'''
```

continent를 기준으로 그룹핑하고, country열의 빈도수 추출

## 데이터 그리기

```python
global_yearly_life_expectancy = df.groupby('year')['lifeExp'].mean()
global_yearly_life_expectancy.plot()
print(plt.show())

'''
년도를 기준으로 그룹핑 후 lifeExp의 평균값을 추출
이후 데이터 시각화
'''
```

# 데이터프레임과 시리즈

## 데이터 만들기

### 시리즈 만들기

```python
s = pd.Series(['banana', 42])
print(s)

'''
0    banana
1        42
dtype: object
'''
```

```python
df = pd.read_csv('/Users/sihoon/Desktop/Pandas/gapminder.tsv', sep='\t')

s = pd.Series(['banana', 42], index=['fruit', 'count'])
print(s)

'''
fruit    banana
count        42
dtype: object
'''
```

index=[]를 이용해 index를 문자열로 지정할 수 있다.

### 데이터프레임 만들기

```python
Scientists = pd.DataFrame({
    'Name':['Rosaline Franklin', 'William Gosset'],
    'Occupation':['Chemist', 'Statistician'],
    'Age':[37, 61]})

print(Scientists)
```

딕셔너리는 데이터의 순서를 보장하지 않는다. 그래서 **OederedDict 클래스를 사용하면 순서를 유지할 수 있다.**

```python
from collections import OrderedDict

Scientists = pd.DataFrame(OrderedDict([
    ('Name', ['Rosaline Franklin', 'William Gosset']),
    ('Occupation', ['Chemist', 'Statistician']),
    ('Age', [37, 61])
])
)

print(Scientists)

'''
                Name    Occupation  Age
0  Rosaline Franklin       Chemist   37
1     William Gosset  Statistician   61

'''
```

## 시리즈 다루기
