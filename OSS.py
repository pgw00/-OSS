from sklearn import datasets

# 유방암 데이터셋 불러오기
cancer = datasets.load_breast_cancer()

# 데이터 정보 출력
print(cancer)