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

시리즈 : **엑셀의 행열 중에 한줄의 열로만 구성된 테이블**

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

### 시리즈 선택

df

```python
Scientists = pd.DataFrame(
    data = {'Occupation': ['Chemist', 'Statistictian'], 
            'Born' : ['1920-07-25', '1876-06-13'], 
            'Died' : ['1958-04-16', '1937-10-16'], 
            'Age' : [37, 61]}, 
    index = ['Rosaline Franklin', 'William Gosset'], 
    columns = ['Occupation', 'Born', 'Died', 'Age'])
```

데이터프레임에서 시리즈를 선택하려면 loc속성에 인덱스(Scientist)를 전달하면 된다.

```python
firstrow = Scientists.loc['William Gosset']
print(firstrow)

'''
Occupation    Statistictian
Born             1876-06-13
Died             1937-10-16
Age                      61
Name: William Gosset, dtype: object
'''
여기서 Age가 정수형인데 무시되고, object형으로 된다.
```

### index, value, keys

**index, value는 시리즈 속성이고, keys는 메서드이다.**

- index

시리즈의 인덱스 출력

```python
print(firstrow.index)

'''
Index(['Occupation', 'Born', 'Died', 'Age'], dtype='object')
'''

print(firstrow.index[0])
'''
Occupation
'''
```

- value

시리즈의 데이터 출력

```python
print(firstrow.values)

'''
['Statistictian' '1876-06-13' '1937-10-16' 61]
'''
```

- keys

== index

```python
print(firstrow.keys())

'''
Index(['Occupation', 'Born', 'Died', 'Age'], dtype='object')
'''

print(firstrow.keys()[0])
'''
Occupation
'''
```

---

```python
print(Scientists['Age'])

'''
Rosaline Franklin    37
William Gosset       61
Name: Age, dtype: int64
'''
```

| 시리즈 메서드 |  |
| --- | --- |
| append | 시리즈 연결 |
| describe | 요약 통계량 계산 |
| drop_duplicates | 중복 없는 시리즈 반환 |
| equals | 시리즈에 해당 값을 가진 요소가 있는지 확인 |
| get_values | 시리즈 값 구하기 |
| isin | 시리즈에 포함된 값이 있는지 확인 |
| min | 최솟값 |
| max | 최댓값 |
| mean | 평균값 |
| median | 중간값 |
| replace | 시리즈 값 교체 |
| sample | 시리즈에서 임의의 값을 반환 |
| sort_values | 값을 정렬 |
| to_frame | 시리즈를 데이터프레임으로 변환 |

```python
print(ages.mean())
print(ages.min())
---
```

## 시리즈 불린 추출

```python
ages = scientists['Age']
print(ages.max())

'''
90
'''
```

```python
ages = scientists['Age']
print(ages[ages > ages.mean()])

'''
1    61
2    90
3    66
7    77
Name: Age, dtype: int64
'''
```

```python
print(ages > ages.mean())

'''
0    False
1     True
2     True
3     True
4    False
5    False
6    False
7     True
Name: Age, dtype: bool
'''
```

## 시리즈와 브로드캐스팅

브로드캐스트 : 모든 데이터에 대해 한 번에 연산하는 것

벡터 : 시리즈처럼 여러 개의 값을 가진 데이터

스칼라 : 단순 크기를 나타내는 데이터

```python
print(ages)
'''
0    37
1    61
2    90
3    66
4    56
5    45
6    41
7    77
Name: Age, dtype: int64
'''
```

```python
print(ages + ages)
'''
0     74
1    122
2    180
3    132
4    112
5     90
6     82
7    154
Name: Age, dtype: int64
'''
벡터의 연산이다.
```

```python
print(ages + 100)
'''
0    137
1    161
2    190
3    166
4    156
5    145
6    141
7    177
Name: Age, dtype: int64
'''
벡터에 스칼라 연산하면 벡터의 모든 값에 스칼라를 적용한다.
```

벡터끼리의 연산 중 서로 길이가 다르면, 누락값 (NaN)이 생긴다.

ex : ) 인덱스 2개의 데이터 + 인덱스 10개의 데이터 → 2개만 연산 , 나머지 NaN

**인덱스의 길이가 일치하다면 연산 가능하다.**

---

### sort_index 메서드

```python
reverse_ages = ages.sort_index(ascending=False)
print(reverse_ages)

'''
7    77
6    41
5    45
4    56
3    66
2    90
1    61
0    37
Name: Age, dtype: int64
'''
```

## 데이터프레임 다루기

### 불린 추출

```python
print(scientists[scientists['Age'] > scientists['Age'].mean()])
'''
Name        Born        Died  Age     Occupation
1        William Gosset  1876-06-13  1937-10-16   61   Statistician
2  Florence Nightingale  1820-05-12  1910-08-13   90          Nurse
3           Marie Curie  1867-11-07  1934-07-04   66        Chemist
7          Johann Gauss  1777-04-30  1855-02-23   77  Mathematician
'''
```

```python
print(scientists*2)

'''
이 연산을 하게되면 데이터의 양이 2배 늘어나는데, 
정수형은 *2의 연산이 되고, 
문자열은 문자열이 2배늘어난다.
'''
```

## 시리즈, 데이터프레임 데이터 처리

```python
print(scientists['Born'].dtype)

'''
object
'''
```

Born열은 문자열이다. datetime형식 자료형으로 바꿔주는 것이 좋은데 이를 해보면

```python
born_datetime = pd.to_datetime(scientists['Born'], format='%Y-%m-%d')
print(born_datetime)

'''
0   1920-07-25
1   1876-06-13
2   1820-05-12
3   1867-11-07
4   1907-05-27
5   1813-03-15
6   1912-06-23
7   1777-04-30
Name: Born, dtype: datetime64[ns]
'''
타입이 바뀌게 된다.
```

이 컬럼을 추가하려면

```python
scientists['born_dt'] = born_datetime
print(scientists.head())

'''
                   Name        Born        Died  Age    Occupation    born_dt
0     Rosaline Franklin  1920-07-25  1958-04-16   37       Chemist 1920-07-25
1        William Gosset  1876-06-13  1937-10-16   61  Statistician 1876-06-13
2  Florence Nightingale  1820-05-12  1910-08-13   90         Nurse 1820-05-12
3           Marie Curie  1867-11-07  1934-07-04   66       Chemist 1867-11-07
4         Rachel Carson  1907-05-27  1964-04-14   56     Biologist 1907-05-27
'''
```

새로운 컬럼이 생긴 것을 볼 수 있다.

### 데이터 섞기

```python
import random

random.seed(42)
random.shuffle(scientists['Age'])
```

- seed()는 난수의 기준값이다.

### 데이터프레임 열 삭제

- drop(컬럼, axis = 1)

```python
print(scientists.columns)

'''
Index(['Name', 'Born', 'Died', 'Age', 'Occupation'], dtype='object')
'''
```

```python
scientists_del = scientists.drop(['Age'], axis=1)
print(scientists_del)

'''
       Name        Born        Died          Occupation
0     Rosaline Franklin  1920-07-25  1958-04-16             Chemist
1        William Gosset  1876-06-13  1937-10-16        Statistician
2  Florence Nightingale  1820-05-12  1910-08-13               Nurse
3           Marie Curie  1867-11-07  1934-07-04             Chemist
4         Rachel Carson  1907-05-27  1964-04-14           Biologist
5             John Snow  1813-03-15  1858-06-16           Physician
6           Alan Turing  1912-06-23  1954-06-07  Computer Scientist
7          Johann Gauss  1777-04-30  1855-02-23       Mathematician
'''
```

## 데이터 저장, 불러오기

### 피클 저장

- to_pickle

스프레드시트보다 더 작은 용량으로 데이터를 저장

오래보관

```python
scientists.to_pickle('경로.pickle')
```

이것을 그냥 열면 안되고, read_pickle 메서드로 읽어야함

```python
df = pd.read.pickle('경로.pickle')
```

### csv, tsv

- csv : 데이터를 컴마(,)로 구분한 파일
- tsv : 데이터를 탭(Tab)으로 구분한 파일

```python
scientists.to_csv('/Users/sihoon/Desktop/Pandas/scientists.tsv', sep = '\t')
```

tsv는 sep= ‘\t’를 꼭 넣어줘야함 구분을 tab으로 하니까

# 그래프 그리기

## matplotlib 라이브러리로 그래프 그리기

1. 전체 그래프가 위치할 기본 틀 만들기
2. 그래프 격자 만들기
    
    plt.figure()
    
3. 격자에 그래프를 하나씩 추가 (좌 → 우)
4. ‘’ (다음 행 (상 → 하))

## seaborn 데이터 불러오기

```python
import seaborn as sns

데이터명을 저장할 변수 이름 = sns.load_dataset('seaborn 데이터명')
```

### seaborn 데이터의 앤스콤 데이터

```python
import seaborn as sns

anscombe = sns.load_dataset("anscombe")

print(anscombe.head())

'''
  dataset     x     y
0       I  10.0  8.04
1       I   8.0  6.95
2       I  13.0  7.58
3       I   9.0  8.81
4       I  11.0  8.33
'''
```

### matplotlib로 그래프 그리기

- plot 메서드는 기본적으로 선을 그래프로 그려준다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

anscombe = sns.load_dataset("anscombe")

dataset_1 = anscombe[anscombe['dataset'] == 'I']
plt.plot(dataset_1['x'], dataset_1['y'])
print(plt.show())

# anscombe 데이터에서 dataset이 I인 값을 추출하는데
# 이 중 x와 y 값을 축으로 사용해 그래프를 출력
```

사진

- x축, y축, ‘o’ : 세 번째 인자를 ‘o’를 사용해 점으로 나타낼 수 있다.

사진

### 앤스콤 데이터 집합 모두 사용해 그래프 만들기

앤스콤데이터는 4개의 집합으로 구성되어 있는 데 이를 모두 시각화 하려면,

```python
import seaborn as sns
import matplotlib.pyplot as plt

anscombe = sns.load_dataset("anscombe")

dataset_1 = anscombe[anscombe['dataset'] == 'I']
dataset_2 = anscombe[anscombe['dataset'] == 'II']
dataset_3 = anscombe[anscombe['dataset'] == 'III']
dataset_4 = anscombe[anscombe['dataset'] == 'IV']
#data들을 지정해 준다.

fig = plt.figure()
# 그래프 격자가 위치할 기본 틀 만들기

axes1 = fig.add_subplot(2, 2, 1)
axes2 = fig.add_subplot(2, 2, 2)
axes3 = fig.add_subplot(2, 2, 3)
axes4 = fig.add_subplot(2, 2, 4)
# add_subplot을 사용 해 그래프 격자 그리기
# 첫 번째 인자 : 행 크기
# 두 번째 인자 : 열 크기
# 세 번째 인자 : n번 째 그래프

axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
axes2.plot(dataset_2['x'], dataset_2['y'], 'o')
axes3.plot(dataset_3['x'], dataset_3['y'], 'o')
axes4.plot(dataset_4['x'], dataset_4['y'], 'o')
# 점으로 표현한 데이터 그래프

fig
# 그래프를 확인하려면 fig가 필요하다.

print(plt.show())
```

사진

### 제목 추가

- 그래프의 제목 추가하기 : set_title

```python
axes1.set_title("dataset_1")
axes2.set_title("dataset_2")
axes3.set_title("dataset_3")
axes4.set_title("dataset_4")
```

- 기본 틀(fig)에 제목 추가하기 : suptitle

```python
fig.suptitle("Ansombe Data")

fig
```

### 레이아웃 조절

- 여러 그래프 간의 충돌을 방지 할 수 있다.

```python
fig.tight_layout()

fig
```

---

df

tips 식당에서 팁을 지불한 손님의 정보

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
print(tips.head())

'''
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
'''
```

- 기본적인 틀

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig = plt.figure()
axes1 = fig.add_subplot(1, 1, 1)
print(plt.show())
```

사진

### 히스토그램 hist

- 일변량

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig = plt.figure()
axes1 = fig.add_subplot(1, 1, 1)
axes1.hist(tips['total_bill'], bins = 10)
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill')

fig

print(plt.show())
```

사진

- x축의 간격을 bins 인잣값으로 조정 가능
- set_xlabel, set_ylabel 축들의 이름을 정할 수 있다.

### 산점도 그래프 (이변량)

- 이변량
    
    두개의 변수를 이용해 그래프를 그림
    

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig = plt.figure()

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Scatterplot of Total Bill vs Tip')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')

print(plt.show())
```

사진

기본 틀과 그래프 격자를 만들고 scatter 메서드에 total_bill, tips 열을 전달

### 박스 그래프

이산형 변수(구분이 가능한 변수)와 연속형 변수 (명확하게 셀 수 없는 변수)를 함께 사용하는 그래프 → x, y축으로 표현

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig = plt.figure()

boxplot = plt.figure()
axes1 = boxplot.add_subplot(1, 1, 1)

axes1.boxplot([tips[tips['sex'] == 'Female']['tip'],
              tips[tips['sex'] == 'Male']['tip']],
              labels=['Female', 'Male'])

axes1.set_xlabel('Sex')
axes1.set_ylabel('Tip')
axes1.set_title('Boxplot of Tips by Sex')

print(plt.show())
```

사진

tips 데이터프레임에서 성별이 female인 데이터와 male인 데이터에서 tip 열 데이터만 추출해 리스트에 담아 전달 한 것

---

현재까지의 그래프에 대해 복습 철저히 할 것

---

### 산점도 그래프 (다변량)

3개 이상의 변수를 사용함.

recode_sex 함수를 브로드캐스팅 하기 위해 apply 메서드를 사용

```python
import seaborn as sns
import matplotlib.pyplot as plt

def recode_sex(sex):
    if sex == 'Female':
        return 0
    elif sex == 'Male':
        return 1
# 함수를 써서 값을 변수를 추가함

tips = sns.load_dataset('tips')
tips['sex_color'] = tips['sex'].apply(recode_sex)

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(
    x = tips['total_bill'],
    y = tips['tip'],
    s = tips['size'] * 10,
    c = tips['sex_color'],
    alpha = 0.5)
'''
s : 점의 크기 size
c : 색상 color
alpha : 점의 투명도 
'''

axes1.set_title('Total Bill vs Tip Colored by Sex and Sized by Size')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')

print(plt.show())
```

사진

---

seaborn 라이브러리로 히스토그램을 그리려면

subplot, distplot 메서드를 사용하면 된다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

ax = plt.subplots() #  기본 틀 만들기
ax = sns.distplot(tips['total_bill']) #
ax.set_title('Total Bill Histogram with Density Plot')

print(plt.show())
```

사진

- 밀집도 그래프를 제외하고 싶으면 kde =False를 적으면 된다.

```python
ax = sns.distplot(tips['total_bill'], kde=False)
```

사진

- 밀집도 그래프만 나타내려면 hist=False를 적으면 된다.

```python
ax = sns.distplot(tips['total_bill'], hist=False)
```

사진

- 양탄자(데이터 밀집도)를 표현하려면 rug = True를 적으면 된다.

```python
ax = sns.distplot(tips['total_bill'], rug=True)
```

사진

| kde=False | 밀집도 그래프 제외 |
| --- | --- |
| hist=False | 밀집도 그래프만 |
| rug=True | 데이터 밀집도 표현 |

### count 그래프

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

ax = plt.subplots() #  기본 틀 만들기

ax = sns.countplot('day', data=tips) # tips 데이터프레임의 day열 데이터를 넣은 count 그래프
ax.set_title('Count of days')
ax.set_xlabel('Day of week')
ax.set_ylabel('Frequency')
print(plt.show())
```

### 다양한 종류의 이변량 그래프

- regplot 메서드 : 산점도 그래프 + 회귀선

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

ax = plt.subplots() #  기본 틀 만들기

ax = sns.regplot(x= 'total_bill', y = 'tip', data = tips)
ax.set_title('Scatterplot of Total Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
print(plt.show())
```

사진

- 만약 회귀선을 제거 하고 싶다 → fit_reg=False

```python
ax = sns.regplot(x= 'total_bill', y = 'tip', data = tips, fit_reg=False)
```

| 그래프 종류 | 설명 |
| --- | --- |
| regplot | 산점도+회귀선 그래프 |
| barplot | 바그래프 |
| boxplot | 박스그래프 |
| vilolinplot | 바이올린그래프 |

## 데이터프레임과 시리즈로 그래프 그리기

hist메서드로 시리즈의 값을 이용해 히스토그램 그리기

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

ax = plt.subplots() #  기본 틀 만들기

ax = tips['total_bill'].plot.hist()
print(plt.show())
```

사진

### 2개의 시리즈

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots() #  기본 틀 만들기

ax = tips[['total_bill', 'tip']].plot.hist(alpha=0.5, bins=20, ax = ax)
print(plt.show())
```

사진

| hist | 히스토그램 |
| --- | --- |
| kde | 밀집도그래프 |
| scatter | 산점도그래프 |
| hexbin | 육각그래프 |
| box | 박스그래프 |
- 육각형의 크기 조절 : gridsize=N 인자 추가

# 데이터 연결하기

```python
import pandas as pd

df1 = pd.read_csv("/Users/sihoon/Downloads/concat_1.csv")
df2 = pd.read_csv("/Users/sihoon/Downloads/concat_2.csv")
df3 = pd.read_csv("/Users/sihoon/Downloads/concat_3.csv")
```

연결할 데이터들을 불러온다.

```python
row_concat = pd.concat([df1, df2, df3])
print(row_concat)

'''
     A    B    C    D
0   a0   b0   c0   d0
1   a1   b1   c1   d1
2   a2   b2   c2   d2
3   a3   b3   c3   d3
0   a4   b4   c4   d4
1   a5   b5   c5   d5
2   a6   b6   c6   d6
3   a7   b7   c7   d7
0   a8   b8   c8   d8
1   a9   b9   c9   d9
2  a10  b10  c10  d10
3  a11  b11  c11  d11
'''
```

concat 메서드는 위에서 아래의 방향으로 연결해준다. + 기존 인덱스 값도 유지

기존 인덱스 값을 초기화하려면, ignore_index = True

```python
print(row_concat.iloc[3, ])

'''
A    a3
B    b3
C    c3
D    d3
Name: 3, dtype: object
'''
```

iloc를 이용해 네 번째 행을 추출할 수 있다.

## 데이터 프레임에 시리즈 연결하기

```python
row_concat = pd.concat([df1, df2, df3])
new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(pd.concat([df1, new_row_series]))

'''
     A    B    C    D    0
0   a0   b0   c0   d0  NaN
1   a1   b1   c1   d1  NaN
2   a2   b2   c2   d2  NaN
3   a3   b3   c3   d3  NaN
0  NaN  NaN  NaN  NaN   n1
1  NaN  NaN  NaN  NaN   n2
2  NaN  NaN  NaN  NaN   n3
3  NaN  NaN  NaN  NaN   n4
'''
```

시리즈를 연결하면, 새로운 행이 추가되는 것이 아니라 새로운 열로 추가된다.

### 행 1개로 구성된 데이터프레임 생성하여 연결하기

```python
row_concat = pd.concat([df1, df2, df3])
new_row_df = pd.DataFrame([['n1', 'n2', 'n3', 'n4']], columns=['A', 'B', 'C', 'D'])
print(new_row_df)

'''
    A   B   C   D
0  n1  n2  n3  n4
'''
```

컬럼을 A, B, C, D로 재정의 해줌으로 D라는 컬럼명이 생긴 것.

```python
print(pd.concat([df1, new_row_df]))

'''
    A   B   C   D
0  a0  b0  c0  d0
1  a1  b1  c1  d1
2  a2  b2  c2  d2
3  a3  b3  c3  d3
0  n1  n2  n3  n4
'''
```

### 로우 인덱스 초기화

위의 설명과 동일. 하지만, 인덱스번호가 깔끔하지 않음. ignore_index = True를 사용해서 깔끔하게 만들~

```python
print(pd.concat([df1, new_row_df], ignore_index=True))

'''
    A   B   C   D
0  a0  b0  c0  d0
1  a1  b1  c1  d1
2  a2  b2  c2  d2
3  a3  b3  c3  d3
4  n1  n2  n3  n4
'''
```

## 열 방향으로 데이터 연결하기

- axis = 1 인자를 사용.

```python
col_concat = pd.concat([df1, df2, df3], axis = 1)
print(col_concat)

'''
    A   B   C   D   A   B   C   D    A    B    C    D
0  a0  b0  c0  d0  a4  b4  c4  d4   a8   b8   c8   d8
1  a1  b1  c1  d1  a5  b5  c5  d5   a9   b9   c9   d9
2  a2  b2  c2  d2  a6  b6  c6  d6  a10  b10  c10  d10
3  a3  b3  c3  d3  a7  b7  c7  d7  a11  b11  c11  d11
'''
```

### 새로운 열 추가

```python
col_concat = pd.concat([df1, df2, df3], axis = 1)
col_concat['new_col_list'] = ['n1', 'n2', 'n3', 'n4']
print(col_concat)

'''
    A   B   C   D   A   B   C   D    A    B    C    D new_col_list
0  a0  b0  c0  d0  a4  b4  c4  d4   a8   b8   c8   d8           n1
1  a1  b1  c1  d1  a5  b5  c5  d5   a9   b9   c9   d9           n2
2  a2  b2  c2  d2  a6  b6  c6  d6  a10  b10  c10  d10           n3
3  a3  b3  c3  d3  a7  b7  c7  d7  a11  b11  c11  d11           n4
'''
```

### 열 인덱스 초기화

- ignore_index=True

```python
col_concat = pd.concat([df1, df2, df3], axis = 1, ignore_index=True)
```

## 공통 열과 공통 인덱스 연결

### 열 방향 연결

- join=’inner’

```python
print(pd.concat([df1, df2, df3], join='inner'))
```

### 행 방향 연결

```python
print(pd.concat([df1, df2, df3], axis = 1, join='inner'))
```
