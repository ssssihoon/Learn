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

## merge 메서드

데이터프레임을 왼쪽으로 지정하고, 첫 번째 인잣값으로 지정한 데이터프레임을 오른쪽으로 지정한다. 

left_on과 right_on 인자는 값이 일치해야 할 왼쪽과 오른쪽 데이터프레임의 열을 지정한다.

- 왼쪽 데이터프레임의 열과 오른쪽 데이터프레임의 열의 값이 일치하면 왼쪽 데이터프레임을 기준으로 연결한다.

```python
변수명 = 데이터프레임.merge(데이터, left_on = '일치하는 열1', right_on = '일치하는 열2')
```

# 누락값 처리

Numpy

Nan == Nan → False

누락값은 값 자체가 없기 때문에 자기 자신과 비교해도 False이다.→ 비교가 안된다.

### 누락값이 생기는 경우

- **범위를 지정하여 데이터를 추출할 때**
- **데이터를 입력할 때 없는 열과 행 데이터를 입력하는 경우**
- **누락값이 포함되어있는 데이터집합을 연결하면 더 많은 누락값이 생긴다**

---

### 누락값이 아닌 개수 구하기

테이블명.count()

### 누락값의 개수 구하기

```python
import pandas as pd

ebola = pd.read_csv("/Users/sihoon/Downloads/country_timeseries.csv")

numrows = ebola.shape[0]
num_missing = numrows - ebola.count()
print(num_missing)

'''
Date                     0
Day                      0
Cases_Guinea            29
Cases_Liberia           39
Cases_SierraLeone       35
Cases_Nigeria           84
Cases_Senegal           97
Cases_UnitedStates     104
Cases_Spain            106
Cases_Mali             110
Deaths_Guinea           30
Deaths_Liberia          41
Deaths_SierraLeone      35
Deaths_Nigeria          84
Deaths_Senegal         100
Deaths_UnitedStates    104
Deaths_Spain           106
Deaths_Mali            110
dtype: int64
'''
```

전체의 열 크기 - count(누락값)

---

**이 메서드를 사용해서 누락값을 구할 수 있다.**

- np.count_nonzero()
- isnull()

```python
print(np.count_nonzero(ebola.isnull()))
print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))

'''
1214
29
'''
```

---

열의 빈도를 구하는 시리즈에 포함된 메서드를 사용해 누락값 구하기

- value_counts

```python
print(ebola.Cases_Guinea.value_counts(dropna=False).head())

'''
Cases_Guinea
NaN      29
86.0      3
495.0     2
112.0     2
390.0     2
Name: count, dtype: int64
'''
```

## 누락값 처리하기

### 누락값 변경

- fillna(반환 값)

```python
print(ebola.fillna(0).iloc[0:10, 0:5])

'''
Date  Day  Cases_Guinea  Cases_Liberia  Cases_SierraLeone
0    1/5/2015  289        2776.0            0.0            10030.0
1    1/4/2015  288        2775.0            0.0             9780.0
2    1/3/2015  287        2769.0         8166.0             9722.0
3    1/2/2015  286           0.0         8157.0                0.0
4  12/31/2014  284        2730.0         8115.0             9633.0

'''
누락값을 0.0으로 변경하였다.
```

- ffill
- bfill
- interporlate()

```python
print(ebola.fillna(method='ffill').iloc[0:10, 0:5].head())
print(ebola.fillna(method='bfill').iloc[0:10, 0:5].head())
print(ebola.interpolate().iloc[0:10, 0:5].head())

''' ffill의 값
         Date  Day  Cases_Guinea  Cases_Liberia  Cases_SierraLeone
0    1/5/2015  289        2776.0            NaN            10030.0
1    1/4/2015  288        2775.0            NaN             9780.0
2    1/3/2015  287        2769.0         8166.0             9722.0
3    1/2/2015  286        2769.0         8157.0             9722.0
4  12/31/2014  284        2730.0         8115.0             9633.0

NaN값은 처음부터 누락값이기 때문에 그대로
'''

''' bfill의 값
         Date  Day  Cases_Guinea  Cases_Liberia  Cases_SierraLeone
0    1/5/2015  289        2776.0         8166.0            10030.0
1    1/4/2015  288        2775.0         8166.0             9780.0
2    1/3/2015  287        2769.0         8166.0             9722.0
3    1/2/2015  286        2730.0         8157.0             9633.0
4  12/31/2014  284        2730.0         8115.0             9633.0
'''
```

ffill의 경우 누락값이 나타나기 전으로 변경

bfill의 경우 누락값이 나타난 이후의 첫번째 값으로 변경

interpolate의 경우 ffill과 bfill의 중간값으로 처리

### 누락값 삭제하기

- dropna()

```python
ebola_dropna = ebola.dropna()
print(ebola_dropna.head())

'''
          Date  Day  ...  Deaths_Spain  Deaths_Mali
19  11/18/2014  241  ...           0.0          6.0

[1 rows x 18 columns]
'''
```

### 누락값을 무시한 채 계산하기

skipna=True

```python
print(ebola.Cases_Guinea.sum(skipna = True))
```

# 깔끔한 데이터

melt 메서드는 깔끔한 데이터로 정리하는데 유용하다.

| 메서드 | 설명 |
| --- | --- |
| id_vars | 위치를 그대로 유지할 열의 이름을 지정 |
| value_vars | 행으로 위치를 변경할 열의 이름을 지정 |
| var_nam | value_vars로 위치를 변경할 열의 이름을 지정 |
| value_name | var_name으로 위치를 변경할 열의 데이터를 저장한 열의 이름을 지정 |

## 열과 피벗

df

```python
import numpy as np
import pandas as pd

pew = pd.read_csv("/Users/sihoon/Downloads/pew.csv")
print(pew.head())

'''
             religion  <$10k  $10-20k  ...  $100-150k  >150k  Don't know/refused
0            Agnostic     27       34  ...        109     84                  96
1             Atheist     12       27  ...         59     74                  76
2            Buddhist     27       21  ...         39     53                  54
3            Catholic    418      617  ...        792    633                1489
4  Don’t know/refused     15       14  ...         17     18                 116

[5 rows x 11 columns]
'''
```

### melt 메서드

### 1개의 열만 고정하고 나머지 열을 행으로 바꾸기

- 우선 6개의 열만 출력해보기

```python
print(pew.iloc[:, 0:6])

'''
                   religion  <$10k  $10-20k  $20-30k  $30-40k  $40-50k
0                  Agnostic     27       34       60       81       76
1                   Atheist     12       27       37       52       35
2                  Buddhist     27       21       30       34       33
3                  Catholic    418      617      732      670      638
4        Don’t know/refused     15       14       15       11       10
5          Evangelical Prot    575      869     1064      982      881
'''
```

- 이제 소득정보 열을 행 데이터로 옮겨보기

```python
pew_long = pd.melt(pew, id_vars='religion')
print(pew_long.head())

             religion variable  value
0            Agnostic    <$10k     27
1             Atheist    <$10k     12
2            Buddhist    <$10k     27
3            Catholic    <$10k    418
4  Don’t know/refused    <$10k     15

```

열을 제외한 나머지 소득 정보 열이 variable 열로 정리되고, 소득 정보 열의 데이터도 value 열로 정리되었다. 이를 ********피벗******** 이라고 한다.

이제

- 이를 variable, value 열 이름 바꾸기

```python
pew_long = pd.melt(pew, id_vars='religion', var_name='income', value_name='count')
print(pew_long.head())

'''
             religion income  count
0            Agnostic  <$10k     27
1             Atheist  <$10k     12
2            Buddhist  <$10k     27
3            Catholic  <$10k    418
4  Don’t know/refused  <$10k     15
'''
```

### 2개 이상의 열을 고정하고 나머지 열을 행으로 바꾸기

df

```python
billboard = pd.read_csv("/Users/sihoon/Downloads/billboard.csv")
print(billboard.head())

'''
   year        artist                    track  time  ... wk73  wk74  wk75  wk76
0  2000         2 Pac  Baby Don't Cry (Keep...  4:22  ...  NaN   NaN   NaN   NaN
1  2000       2Ge+her  The Hardest Part Of ...  3:15  ...  NaN   NaN   NaN   NaN
2  2000  3 Doors Down               Kryptonite  3:53  ...  NaN   NaN   NaN   NaN
3  2000  3 Doors Down                    Loser  4:24  ...  NaN   NaN   NaN   NaN
4  2000      504 Boyz            Wobble Wobble  3:35  ...  NaN   NaN   NaN   NaN

[5 rows x 81 columns]
'''
```

- year, artist, track, time, date.entered 열을 모두 고정하고 나머지 열을 피벗하기

```python
billboard = pd.read_csv("/Users/sihoon/Downloads/billboard.csv")
billboard_long = pd.melt(billboard, id_vars=['year', 'artist', 'track', 'time', 'date.entered'], var_name='week', value_name='rating')
print(billboard_long.head())

'''
   year        artist                    track  time date.entered week  rating
0  2000         2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk1    87.0
1  2000       2Ge+her  The Hardest Part Of ...  3:15   2000-09-02  wk1    91.0
2  2000  3 Doors Down               Kryptonite  3:53   2000-04-08  wk1    81.0
3  2000  3 Doors Down                    Loser  4:24   2000-10-21  wk1    76.0
4  2000      504 Boyz            Wobble Wobble  3:35   2000-04-15  wk1    57.0
'''
```

## 열 이름 관리하기

df

```python
evola = pd.read_csv("/Users/sihoon/Downloads/country_timeseries.csv")
print(evola.columns)

'''
Index(['Date', 'Day', 'Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone',
       'Cases_Nigeria', 'Cases_Senegal', 'Cases_UnitedStates', 'Cases_Spain',
       'Cases_Mali', 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone',
       'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates',
       'Deaths_Spain', 'Deaths_Mali'],
      dtype='object')
'''
```

- 5개의 데이터만 확인

```python
print(evola.iloc[:5, [0, 1, 2, 3, 10, 11]])

'''
         Date  Day  Cases_Guinea  Cases_Liberia  Deaths_Guinea  Deaths_Liberia
0    1/5/2015  289        2776.0            NaN         1786.0             NaN
1    1/4/2015  288        2775.0            NaN         1781.0             NaN
2    1/3/2015  287        2769.0         8166.0         1767.0          3496.0
3    1/2/2015  286           NaN         8157.0            NaN          3496.0
4  12/31/2014  284        2730.0         8115.0         1739.0          3471.0
'''
```

- Date, Day를 고정 후 나머지를 행으로 피벗

```python
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
print(ebola_long)

'''
Date  Day      variable   value
0    1/5/2015  289  Cases_Guinea  2776.0
1    1/4/2015  288  Cases_Guinea  2775.0
2    1/3/2015  287  Cases_Guinea  2769.0
3    1/2/2015  286  Cases_Guinea     NaN
4  12/31/2014  284  Cases_Guinea  2730.0
'''
```

- Cases_Guinea 분리하기

```python
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
variable_split = ebola_long.variable.str.split('_')
print(variable_split[:5])

'''
0    [Cases, Guinea]
1    [Cases, Guinea]
2    [Cases, Guinea]
3    [Cases, Guinea]
4    [Cases, Guinea]
Name: variable, dtype: object
'''
이 때 variable_split에 추가된 것은 리스트형식이다.
```

- Cases_Guinea 분리된 것을 데이터프레임에 추가하기

```python
status_values = variable_split.str.get(0)
country_values = variable_split.str.get(1)
ebola_long['status'] = status_values
ebola_long['country'] = country_values

print(ebola_long.head())

'''
         Date  Day      variable   value status country
0    1/5/2015  289  Cases_Guinea  2776.0  Cases  Guinea
1    1/4/2015  288  Cases_Guinea  2775.0  Cases  Guinea
2    1/3/2015  287  Cases_Guinea  2769.0  Cases  Guinea
3    1/2/2015  286  Cases_Guinea     NaN  Cases  Guinea
4  12/31/2014  284  Cases_Guinea  2730.0  Cases  Guinea
'''
```

## 여러 열을 하나로 정리하기

df

```python
weather = pd.read_csv("/Users/sihoon/Downloads/weather.csv")
print(weather.iloc[:5, :])

'''
        id  year  month element  d1    d2  ...  d26  d27  d28  d29   d30  d31
0  MX17004  2010      1    tmax NaN   NaN  ...  NaN  NaN  NaN  NaN  27.8  NaN
1  MX17004  2010      1    tmin NaN   NaN  ...  NaN  NaN  NaN  NaN  14.5  NaN
2  MX17004  2010      2    tmax NaN  27.3  ...  NaN  NaN  NaN  NaN   NaN  NaN
3  MX17004  2010      2    tmin NaN  14.4  ...  NaN  NaN  NaN  NaN   NaN  NaN
4  MX17004  2010      3    tmax NaN   NaN  ...  NaN  NaN  NaN  NaN   NaN  NaN

[5 rows x 35 columns]
'''
```

- melt로 피벗

```python
weather_melt = pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp')
print(weather_melt.head())

'''
        id  year  month element day  temp
0  MX17004  2010      1    tmax  d1   NaN
1  MX17004  2010      1    tmin  d1   NaN
2  MX17004  2010      2    tmax  d1   NaN
3  MX17004  2010      2    tmin  d1   NaN
4  MX17004  2010      3    tmax  d1   NaN
'''
```

- pivot_table : 행과 열의 위치를 바꿔준다.

```python
weather = pd.read_csv("/Users/sihoon/Downloads/weather.csv")
weather_melt = pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp')
weather_tidy = weather_melt.pivot_table(
    index=['id', 'year', 'month', 'day'],
    columns='element',
    values='temp'
)
print(weather_tidy)

'''
element                 tmax  tmin
id      year month day            
MX17004 2010 1     d30  27.8  14.5
             2     d11  29.7  13.4
                   d2   27.3  14.4
                   d23  29.9  10.7
                   d3   24.1  14.4
             3     d10  34.5  16.8
                   d16  31.1  17.6
                   d5   32.1  14.2
             4     d27  36.3  16.7
             5     d27  33.2  18.2
             6     d17  28.0  17.5
                   d29  30.1  18.0
             7     d3   28.6  17.5
                   d14  29.9  16.5
             8     d23  26.4  15.0
                   d5   29.6  15.8
                   d29  28.0  15.3
                   d13  29.8  16.5
                   d25  29.7  15.6
                   d31  25.4  15.4
                   d8   29.0  17.3
             10    d5   27.0  14.0
                   d14  29.5  13.0
                   d15  28.7  10.5
                   d28  31.2  15.0
                   d7   28.1  12.9
             11    d2   31.3  16.3
                   d5   26.3   7.9
                   d27  27.7  14.2
                   d26  28.1  12.1
                   d4   27.2  12.0
             12    d1   29.9  13.8
                   d6   27.8  10.5
'''
```

reset_index()를 사용해 새로 인데스를 지정한다.

```python
weather_tidy_flat = weather_tidy.reset_index()
print(weather_tidy_flat.head())

'''
element       id  year  month  day  tmax  tmin
0        MX17004  2010      1  d30  27.8  14.5
1        MX17004  2010      2  d11  29.7  13.4
2        MX17004  2010      2   d2  27.3  14.4
3        MX17004  2010      2  d23  29.9  10.7
4        MX17004  2010      2   d3  24.1  14.4
'''
```

## 중복 데이터 처리하기

df

```python
billboard = pd.read_csv("/Users/sihoon/Downloads/billboard.csv")
billboard_long = pd.melt(billboard, id_vars=['year', 'artist', 'track', 'time', 'date.entered'],
                         var_name='week', value_name='rating')
print(billboard_long.head())

'''
   year        artist                    track  time date.entered week  rating
0  2000         2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk1    87.0
1  2000       2Ge+her  The Hardest Part Of ...  3:15   2000-09-02  wk1    91.0
2  2000  3 Doors Down               Kryptonite  3:53   2000-04-08  wk1    81.0
3  2000  3 Doors Down                    Loser  4:24   2000-10-21  wk1    76.0
4  2000      504 Boyz            Wobble Wobble  3:35   2000-04-15  wk1    57.0
'''
```

노래(track)에 관해 중복성을 띄는 것이 몇몇 있는데, 이를 처리해줘야한다.

현재 중복 데이터의 열은 year, artist, track, time, date이다.

- 이 열들을 다른 데이터프레임에 따로 저장시켜줘야한다. → billboard_song에 저장시키기
- drop_duplicates()를 사용해 중복 제거

```python
billboard_song = billboard_long[['year', 'artist', 'track', 'time']]
billboard_song = billboard_song.drop_duplicates()
print(billboard_song.head())

'''
   year        artist                    track  time
0  2000         2 Pac  Baby Don't Cry (Keep...  4:22
1  2000       2Ge+her  The Hardest Part Of ...  3:15
2  2000  3 Doors Down               Kryptonite  3:53
3  2000  3 Doors Down                    Loser  4:24
4  2000      504 Boyz            Wobble Wobble  3:35
'''
```

- 중복을 제거한 데이터프레임에 id도 추가

```python
billboard_song['id'] = range(len(billboard_song))
print(billboard_song.head())

'''
   year        artist                    track  time  id
0  2000         2 Pac  Baby Don't Cry (Keep...  4:22   0
1  2000       2Ge+her  The Hardest Part Of ...  3:15   1
2  2000  3 Doors Down               Kryptonite  3:53   2
3  2000  3 Doors Down                    Loser  4:24   3
4  2000      504 Boyz            Wobble Wobble  3:35   4
'''
```

- merge메서드를 사용해 노래 정보와 주간 순위 데이터를 합치기

```python
billboard_ratings = billboard_long.merge(billboard_song, on=['year', 'artist'
                                                             , 'track', 'time'])
print(billboard_ratings.head())

'''
   year artist                    track  time date.entered week  rating  id
0  2000  2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk1    87.0   0
1  2000  2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk2    82.0   0
2  2000  2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk3    72.0   0
3  2000  2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk4    77.0   0
4  2000  2 Pac  Baby Don't Cry (Keep...  4:22   2000-02-26  wk5    87.0   0
'''
```

# 판다스 자료형

df

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")

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

## 자료형을 문자열로 변환하기

- astype()

```python
tips['sex_str'] = tips['sex'].astype(str)

-> astype 메서드를 사용해 sex열의 데이터를 문자열로 변환하여 sex_str로 저장
```

```python
print(tips.dtypes)

'''
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
sex_str         object
dtype: object
'''
-> object는 파다스에서 문자열이다.
```

---

tatal_bill의 자료형(float)를 문자열로 만들기

```python
tips['total_bill'] = tips['total_bill'].astype(str)
```

다시 실수형으로 만들기

```python
tips['total_bill'] = tips['total_bill'].astype(float)
```

### 잘못 입력한 문자열 처리하기

- to_numeric

total_bill 열의 1, 3, 5, 7 행의 데이터를 missing으로 바꾼 후 변수 tips_sub_miss로 저장한 것

```python
tips_sub_miss = tips.head(10)
tips_sub_miss.loc[[1, 3, 5, 7], 'total_bill'] = 'missing'
print(tips_sub_miss)

'''
  total_bill   tip     sex smoker  day    time  size sex_str
0      16.99  1.01  Female     No  Sun  Dinner     2  Female
1    missing  1.66    Male     No  Sun  Dinner     3    Male
2      21.01  3.50    Male     No  Sun  Dinner     3    Male
3    missing  3.31    Male     No  Sun  Dinner     2    Male
4      24.59  3.61  Female     No  Sun  Dinner     4  Female
5    missing  4.71    Male     No  Sun  Dinner     4    Male
'''
```

```python
print(tips_sub_miss.dtypes)
'''
total_bill      object
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
sex_str         object
dtype: object
'''
total_bill의 타입이 문자열로 바껴있다.
이는 missing때문이다.
판다스는 missing이 문자열을 실수로 변환하는 방법을 모른다
이를 to_numeric 메서드를 사용해 해결하려 해도 에러가 발생한다.
에러를 무시하는 errors=ignore을 써도 문자열이다.

-> 이를 해결하기 위해 errors=coerce를 쓴다.
coerce : 억압하다, 강요하다.
```

```python
tips_sub_miss['total_bill'] = pd.to_numeric(tips_sub_miss['total_bill'], errors='coerce')
print(tips_sub_miss.dtypes)

'''
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
sex_str         object
dtype: object
'''
```

- 다운캐스트 : 자료형을 더 작은 형태로 만든다

예상 범위가 크지 않게 하려면 다운캐스트를 하는 것이 좋다.

→ 메모리 공간 차지를 덜 한다.

```python
tips_sub_miss['total_bill'] = pd.to_numeric(tips_sub_miss['total_bill'], errors='coerce', downcast = 'float')
print(tips_sub_miss.dtypes)

'''
total_bill     **float32**
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
sex_str         object
dtype: object
'''
```

## 카테고리 자료형

### 문자열을 카테고리로 변환하기

- .astype(’category’)

# apply 메서드 활용

df

```python
import pandas as pd

df = pd.DataFrame({'a' : [10, 20, 30], 'b' : [20, 30, 40]})
print(df)

'''
    a   b
0  10  20
1  20  30
2  30  40
'''
```

제곱 값 만들기

```python
print(df['a'] ** 2)

'''
0    100
1    400
2    900
Name: a, dtype: int64
'''
```

- apply함수를 사용해 평균 값 구하기

```python
df = pd.DataFrame({'a' : [10, 20, 30], 'b' : [20, 30, 40]})
df = df.apply(np.mean, axis=1)
print(df)

'''
0    15.0
1    25.0
2    35.0
dtype: float64

'''

axis = 1 인자를 사용해 행 별로 평균값을 구했다.
axis = 0 -> 열 방향으로 함수 적용
```

함수를 사용할 때도 쓰인다.

- df.apply(함수)

## 데이터프레임의 누락값을 처리한 후 apply 메서드 사용하기

df

```python
titanic = sns.load_dataset("titanic")
print(titanic.info)

'''
<bound method DataFrame.info of      survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0           0       3    male  22.0  ...   NaN  Southampton     no  False
1           1       1  female  38.0  ...     C    Cherbourg    yes  False
2           1       3  female  26.0  ...   NaN  Southampton    yes   True
3           1       1  female  35.0  ...     C  Southampton    yes  False
4           0       3    male  35.0  ...   NaN  Southampton     no   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
886         0       2    male  27.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True
888         0       3  female   NaN  ...   NaN  Southampton     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True

[891 rows x 15 columns]>
'''
```

### 누락값의 개수를 반환하는 count_missing 함수만들기

```python
def count_missing(vec):
    null_vec = pd.isnull(vec)
    null_count = np.sum(null_vec)
    return null_count
```

개수 구하기

```python
cmis_col = titanic.apply(count_missing)
print(cmis_col)

'''
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
'''
```

# 데이터 집계

## GROUPBY

df

```python
import pandas as pd
df = pd.read_csv("/Users/sihoon/Desktop/Pandas/gapminder.tsv", sep='\t')
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

## year 열을 기준으로 데이터를 그룹화한 다음 lifeExp 열의 평균을 구하기

```python
avg_life_exp_by_year = df.groupby('year').lifeExp.mean()
print(avg_life_exp_by_year.head())

'''
year
1952    49.057620
1957    51.507401
1962    53.609249
1967    55.678290
1972    57.647386
Name: lifeExp, dtype: float64
'''
```

### 1. 분할

데이터를 중복없이 추출하기

```python
years = df.year.unique()
print(years)

'''
[1952 1957 1962 1967 1972 1977 1982 1987 1992 1997 2002 2007]
'''
```

### 2. 반영

연도별 평균값 구하기

예시

```python
y1952 = df.loc[df.year == 1952, :]
print(y1952.head())

'''
        country continent  year  lifeExp       pop    gdpPercap
0   Afghanistan      Asia  1952   28.801   8425333   779.445314
12      Albania    Europe  1952   55.230   1282697  1601.056136
24      Algeria    Africa  1952   43.077   9279525  2449.008185
36       Angola    Africa  1952   30.015   4232095  3520.610273
48    Argentina  Americas  1952   62.485  17876956  5911.315053

'''
```

lifeExp의 평균값 구하기

```python
y1952_mean = y1952.lifeExp.mean()
print(y1952_mean)
'''
49.057619718309866
'''
```

이 작업을 반복해서 모든 연도의 평균값을 구하면 그것이 ‘반영’ 작업이 끝난 것이다.

### 3. 결합

연도별로 계산한 lifeExp의 평균값을 합치면 그것이 결합 작업이다.

```python
df2 = pd.DataFrame({'year' : [1952, ''''], "" :[y1952_mean, '''']})
print(df2)
```

### agg 메서드

직접 함수를 만들어서 사용할 때, groupby 와 함수를 조합하려면 agg 메서드를 이용한다.

- 평균값을 구하는 함수

```python
def my_mean(values):
    n = len(values)
    sum = 0
    for value in values:
        sum += value
        
    return sum / n
```

이 함수와 groupby 를 조합하려면

agg

```python
agg_my_mean = df.groupby('year').lifeExp.agg(my_mean)
print(agg_my_mean)

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
```

## 데이터 필터링

df

```python
tips = sns.load_dataset('tips')
print(tips['size'].value_counts())

'''
size
2    156
3     38
4     37
5      5
1      4
6      4
Name: count, dtype: int64
'''
```

현재 상황은 1, 5, 6테이블의 주문이 매우 적다.

이 데이터를 제외하려 하는데, 

30번 이상의 주문이 있는 테이블 만 필터링하려면

```python
tips_filtered = tips.\
    groupby('size').\
    filter(lambda x: x['size'].count() >= 30)
print(tips_filtered['size'].value_counts())

'''
size
2    156
3     38
4     37
Name: count, dtype: int64
'''
```

## 그룹 오브젝트

df

```python
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

파이썬에서 모든 열을 계산하려 하면, object 형으로 되어있는 데이터 때문에 계산이 안 될 것 같지만 실제로는 문자열을 제외하고 실수형들을 계산해준다.

```python
sex_grouped = tips.groupby('sex')
print(sex_grouped)

'''
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x126a51c50>
'''
```

### 특정 데이터만 추출하기

- 여성만 추출하기

```python
female = sex_grouped.get_group('Female')
print(female)

'''
     total_bill   tip     sex smoker   day    time  size
0         16.99  1.01  Female     No   Sun  Dinner     2
4         24.59  3.61  Female     No   Sun  Dinner     4
11        35.26  5.00  Female     No   Sun  Dinner     4
14        14.83  3.02  Female     No   Sun  Dinner     2
16        10.33  1.67  Female     No   Sun  Dinner     3
..          ...   ...     ...    ...   ...     ...   ...
226       10.09  2.00  Female    Yes   Fri   Lunch     2
229       22.12  2.88  Female    Yes   Sat  Dinner     2
238       35.83  4.67  Female     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2

[87 rows x 7 columns]

'''
```

- 모든 성별 반복문으로 추출하기

```python
for sex_group in sex_grouped:
    print(sex_group)

'''
('Male',      total_bill   tip   sex smoker  day    time  size
1         10.34  1.66  Male     No  Sun  Dinner     3
2         21.01  3.50  Male     No  Sun  Dinner     3
3         23.68  3.31  Male     No  Sun  Dinner     2
5         25.29  4.71  Male     No  Sun  Dinner     4
6          8.77  2.00  Male     No  Sun  Dinner     2
..          ...   ...   ...    ...  ...     ...   ...
236       12.60  1.00  Male    Yes  Sat  Dinner     2
237       32.83  1.17  Male    Yes  Sat  Dinner     2
239       29.03  5.92  Male     No  Sat  Dinner     3
241       22.67  2.00  Male    Yes  Sat  Dinner     2
242       17.82  1.75  Male     No  Sat  Dinner     2

[157 rows x 7 columns])
('Female',      total_bill   tip     sex smoker   day    time  size
0         16.99  1.01  Female     No   Sun  Dinner     2
4         24.59  3.61  Female     No   Sun  Dinner     4
11        35.26  5.00  Female     No   Sun  Dinner     4
14        14.83  3.02  Female     No   Sun  Dinner     2
16        10.33  1.67  Female     No   Sun  Dinner     3
..          ...   ...     ...    ...   ...     ...   ...
226       10.09  2.00  Female    Yes   Fri   Lunch     2
229       22.12  2.88  Female    Yes   Sat  Dinner     2
238       35.83  4.67  Female     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2

[87 rows x 7 columns])
'''
```

# 시계열 데이터
