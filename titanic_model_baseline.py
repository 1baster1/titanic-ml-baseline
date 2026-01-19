import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("START baseline")

# 1) Загружаем данные
df = pd.read_csv("Titanic-Dataset.csv", encoding="UTF-8")

# 2) Target (что предсказываем)
y = df["Survived"]

# 3) Features (по каким данным предсказываем)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]

print("Data loaded:", X.shape, "Target:", y.shape)

# 4) Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Какие столбцы числовые / категориальные
num_cols = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
cat_cols = ["Sex", "Embarked"]

# 6) Препроцессинг чисел: заполняем пропуски медианой
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

# 7) Препроцессинг категорий: заполняем пропуски самым частым + one-hot
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 8) Собираем всё вместе
preprocess = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

# 9) Модель (baseline)
model = LogisticRegression(max_iter=1000)

# 10) Pipeline: сначала preprocess, потом model
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# 11) Обучаем
clf.fit(X_train, y_train)

# 12) Предсказываем
y_pred = clf.predict(X_test)

# 13) Оценка качества
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nReport:\n", classification_report(y_test, y_pred))

print("END baseline")

