# 간단하게 통계하기

import openpyxl

import matplotlib.pyplot as plt
import numpy as np

# 파일선택
loadEx = openpyxl.load_workbook('./excel.xlsx')
listEx = loadEx.active

# 원하는 부분 선택 
select = listEx['N4':'T963']

# 기본값 0 인 45개의 리스트 만들기
checklist = [0 for i in range(45)]

# 추천된 번호 인댁스 번호에 1개씩 추가하기(누적계산)
for cells in select:
    for cell in cells:
        checklist[int(cell.value)-1] += 1

# 보여주기 위해 1~45개의 리스트 만들기
number = [i for i in range(1,46)]
    
# 입력개수와 맞춰줘야된다.
x = np.arange(45)

# 정보입력 후 출력
plt.bar(x,checklist) #좌측계측 리스트
plt.xticks(x, number) #하단표시 리스트
plt.show()

"""

파이썬3.9 버전부턴 pandas 엑셀 오픈기능이 불안정해서 사용을 안한다.
import pandas as pd

rtList =  pd.read_excel('excel.xls', sheet_name=1)

print(trList)

"""
