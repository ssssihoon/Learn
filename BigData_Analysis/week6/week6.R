# 내장 데이터 cars 
str(cars) # cars 데이터의 구조를 보는 함수 str -> structure

# x축에 speed, y축에 distance로 산점도 그리기
plot(cars$speed, cars$dist, xlab = "속도", ylab = '거리', pch=1, col="red") 
# pch : 점 모양, col : color, $ : 데이터 내부의 변수 접근가능

# 2012인구데이터 불러오기
load("data.rda")

# 데이터 구조 확인
str(data) # 변수 5개

# 결측치 0으로 변환 후 초기화 (v2. 연령이 담긴 변수)
#data$V2 = (data$V2[is.na(data$V2)] <- 0)

# 문자형을 숫자형으로 변환
data$V2 <- as.numeric(data$V2)

data$V2 <- na.omit(data$V2)

str(data$V2)

# 데이터 빈도수 확인 table 함수
table_v4 = table(data$V4)
table_v4

# 막대그래프 barplot
barplot(table_v4)

# 막대그래프 옵션
barplot(table_v4, xlab = "교육정도", ylab = "빈도수", col = "blue")

# V2데이터(연령) 히스토그램
hist(data$V2, xlab = "연령", ylab = "빈도수")

# 파이차트
pie(table_v4)
# 옵션
pie(table_v4, cex = 0.5)
# cex : 폰트 크기 조정 

str(cars)

# 박스그래프
boxplot(cars$dist)

# cafe 데이터 
tinocafe <- read.csv("cafedata.csv", header = T, na.strings = "na", stringsAsFactors = FALSE) 
# 첫 줄을 헤더로 인식, na를 결측값으로 인식, 문자열을 캐릭터로 유지

str(tinocafe)
tc <- tinocafe$Coffees
# 평균값 구하기. 결측치는 빼고.
mean(tc, na.rm = TRUE)
