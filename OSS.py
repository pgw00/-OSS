from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 1. 데이터셋 유방암 불러오기(데이터셋 로드)
cancer = datasets.load_breast_cancer()

# 데이터 정보 출력
#print(cancer)

# 입력 데이터(X)와 정답 데이터(y)
X = cancer.data
y = cancer.target

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)


# 3. 머신러닝 모델 학습
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# 4. 예측 및 정확도
y_pred = model.predict(X_test)

scores = metrics.accuracy_score(y_test, y_pred)
print(scores)