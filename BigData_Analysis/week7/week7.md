# week7

# 판다스

- 데이터를 조작하고 분석하는 데 사용 (파이썬 라이브러리)
- 시리즈(열), 데이터프레임(행, 열) 데이터형 사용

## 시리즈 sr

- 1차원 데이터
- 리스트와 비슷하지만 인덱스 부여 가능

## 데이터프레임 df

- 2차원 데이터
- 행과 열로 이루어짐

## 시리즈 & 데이터프레임

- 행에 접근 : `df.loc[idx_name]` or `df.iloc[row_idx]`
- 열에 접근 : `df[col_name]` or `df.col_name`
- 셀 하나에 접근 : `df.loc[idx_name, col_name]` , `df.loc[idx_name][col_name]` or `df.iloc[row_num, col_num]` , `df.iloc[row_num][col_num]`

# 문법

- 데이터프레임의 열 확인
    - df.columns
- 데이터프레임의 열 이름 변경
    - df.rename(columns = {col_name : ‘change_col_name’})
- 값을 확인
    - df.values
- 데이터 구조 확인
    - x.shape
- 데이터프레임의 행의 인덱스 시작과 끝, 증가값 확인
    - df.index
- 행 / 열 삭제
    - df.drop([idx_name1, ```])
        - axis = 1 : 열 삭제
        - axis = 0 : 행 삭제
- 인덱스 새로 지정
    - set_index()
        - append = True : 기존 인덱스 유지
        - append = False : 기존 인덱스 삭제
- 기존 인덱스 삭제하고 초기화
    - reset_index
- 날짜로 인덱스 만들기
    - pd.date_range(시작날짜, 간격, 단위)
