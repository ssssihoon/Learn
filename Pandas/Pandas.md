# Pandas

# 데이터 집합 불러오기

```sql
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

## 시리즈와 데이터프레임

- 시리즈 : 시트의 열 1개
- 데이터 프레임 : 딕셔너리와 같다

```sql
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
