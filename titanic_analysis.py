import pandas as pd

# Загружаем датасет
df = pd.read_csv("Titanic-Dataset.csv", encoding="UTF-8")


# Показываем первые 5 строк таблицы
print("Первые строки данных:")
print(df.head())

# Показываем структуру таблицы
print("\nИнформация о таблице:")
print(df.info())

print("\nПропуски в данных:")
print(df.isnull().sum())

print("\nСтатистика чисел:")
print(df.describe())

print("\nСколько людей выжило и не выжило:")
print(df["Survived"].value_counts())

print("\nВыживаемость по полу:")
print(df.groupby("Sex")["Survived"].mean())

print("\nсредний возвравст пассажиров;")
print(df["Age"].mean())

df["Age"] = df["Age"].fillna(df["Age"].mean())

print("\nПропуски возраста после заполнения")
print(df["Age"].isnull().sum())




