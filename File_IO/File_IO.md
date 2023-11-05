# csv

csv comma separated value : ,로 구분하는 텍스트 데이터

인코딩 encoding : 문자 데이터를 컴퓨터에게 전달하는 방식

메모장에 작성 후

파일 이름을 파일명.csv로 하고, 모든파일 선택

인코딩을 ansi로 변경 후 저장

```python
#csv 파일 출력

from google.colab import files

# 파일 업로드 함수 호출
f = files.upload()

f = open('characters.csv', 'r', encoding = 'utf-8')

import csv

rdr = csv.reader(f)

# 반복문으로 한 행씩 출력하기
for line in rdr:
	print(line)
```

```python
from google.colab import files

# 파일 업로드 함수 호출
f = files.upload()

# csv 라이브러리 가져오기
import csv

# csv 파일 불러오기
f = open('write.csv', 'w', encoding = 'utf-8', newline = '')

# csv 파일 작성할 준비하기
wr = csv.writer(f)

# 데이터를 한 행씩 작성하기
wr.writerow(['ID', '이름', '상징색', '취미'])

# f 변수의 자원 반환
f.close()
```

# 엑셀파일

```python
# 엑셀파일 업로드
from google.colab import files
f = files.upload()

# openpyxl 라이브러리 설치하기
! pip install openpyxl
# openpyxl 라이브러리 가져오기
import openpyxl

wb = openpyxl_workbook('characters.xlsx')

# 특정 열 출력
print(wb.sheetnames)

# 특정 위치의 데이터 읽기
sheet1 = wb['Sheet1']
print(shee1['A1'].value)

# 새 워크북과 워크시트 생성
wb = openpyxl.Workbook() # 워크시트는 자동 생성 1개

# 새 워크시트 생성
wb.create_sheet('Sheet2')
# 시트 이름 출력하기
print(wb.sheetnames)

# 워크시트 이름 변경하기
sheet1 = wb['Sheet']
sheet2 = wb['Sheet2']

sheet1.title = '캐릭터 명단'
sheet2.title = '인기도 조사'

print(wb.sheetnames)

# 워크시트에 데이터 입력하기
sheet['B1'] = '인기도 조사결과'
print(sheet2['B2'].value)

# 워크시트 복제
copysheet = wb.copy_worksheet(sheet2)

# 워크시트 삭제
del wb['인기도 조사']

# 엑셀파일 저장
wb.save('complete.xlsx')
```
