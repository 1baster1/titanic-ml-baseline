import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

print("START v2")

# 1) Загружаем данные
df = pd.read_csv("Titanic-Dataset.csv", encoding="utf-8")

# 2) Target (что предсказываем)
y = df["Survived"]

# 3) Features (по каким данным предсказываем)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]

print("Data loaded:", X.shape, "Target:", y.shape)

# 4) Делим на train/test (stratify важно, чтобы пропорции классов сохранились)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5) Какие столбцы числовые / категориальные
num_cols = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
cat_cols = ["Sex", "Embarked"]

# 6) Препроцессинг чисел: пропуски -> медиана
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

# 7) Препроцессинг категорий: пропуски -> самый частый + one-hot
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 8) Собираем preprocess
preprocess = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])

# ===== A) Dummy baseline =====
dummy = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DummyClassifier(strategy="most_frequent"))
])

dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_acc = accuracy_score(y_test, dummy_pred)

print("\n=== DUMMY BASELINE ===")
print("Accuracy:", dummy_acc)

# ===== B) Logistic Regression =====
model = LogisticRegression(max_iter=2000, class_weight="balanced")


clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# вероятности для ROC-AUC
y_proba = clf.predict_proba(X_test)[:, 1]
for thr in [0.3, 0.5, 0.7]:
    y_pred_thr = (y_proba >= thr).astype(int)

    acc_thr = accuracy_score(y_test, y_pred_thr)
    cm_thr = confusion_matrix(y_test, y_pred_thr)

    print(f"\n=== THRESHOLD = {thr} ===")
    print("Accuracy:", acc_thr)
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm_thr)


print("\nПример 5 предсказаний (класс):", y_pred[:5])
print("Пример 5 вероятностей (выжил):", y_proba[:5])
print("Настоящие ответы:", y_test.values[:5])

print("\nПроверка порогов на этих 5 вероятностях:")

probs5 = y_proba[:5]

for thr in [0.3, 0.5, 0,7]:
    y_pred_thr = (y_proba >=thr).astype(int)
    cm_thr = confusion_matrix(y_test, y_pred_thr)
tn, fp, fn, tp, = cm_thr.ravel()
print(f"\nthr={thr}")
print("TN FP FN TP", tn,fp,fn,tp,)

for thr in [0.3, 0.5, 0.7]:
    preds5 = (probs5 >= thr).astype(int)
    print("thr =", thr, "->", preds5)


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n=== LOGREG MODEL ===")
print("Accuracy:", acc)
print("ROC-AUC:", auc)

print("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
print(cm)

print("\nReport:\n", classification_report(y_test, y_pred))

print("END v2")
