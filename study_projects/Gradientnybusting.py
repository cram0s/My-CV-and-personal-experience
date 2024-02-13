import pandas as pd
import pymysql.cursors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

con = pymysql.connect(host='localhost',
                      user='root',
                      password='root',
                      charset='utf8mb4',
                      db='forecast1',
                      cursorclass=pymysql.cursors.DictCursor)
print('Подключение к базе данных прошло успешно!')

with con.cursor() as cursor:
    cursor.execute("select * from complete_no_null_water5")
    lst = cursor.fetchall()
    df = pd.DataFrame(lst)
    df = df[
        ['Код поста', 'Дата - время', 'Уровень воды', 'Температура воздуха', 'Атмосферное давление', 'Скорость ветра',
         'Количество осадков']]
    df = df.fillna(0)
    df = df.iloc[:220000]
    con.close()

post = input("Введите код поста: ")
start_date = input("Введите дату начала прогнозирования (ГГГГ-ММ-ДД): ")
end_date = input("Введите дату конца прогнозирования (ГГГГ-ММ-ДД): ")

# Фильтрация данных по датам
df['Дата - время'] = pd.to_datetime(df['Дата - время'])
df_filtered = df[(df['Код поста'] == post) & (df['Дата - время'] >= start_date) & (df['Дата - время'] <= end_date)]

if not df_filtered.empty:
    df['Уровень воды(t-1)'] = df.groupby('Код поста')['Уровень воды'].shift(1)
    df['Уровень воды(t-2)'] = df.groupby('Код поста')['Уровень воды'].shift(2)
    df['Уровень воды(t-3)'] = df.groupby('Код поста')['Уровень воды'].shift(3)

    # Заполнение пропущенных значений
    df[['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']] = df.groupby('Код поста')[
        ['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']].fillna(0)

    X = df[['Код поста', 'Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Атмосферное давление',
            'Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']]
    y = df['Уровень воды']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Запрос параметров от пользователя
    n_estimators = int(input("Введите количество деревьев (n_estimators): "))
    learning_rate = float(input("Введите скорость обучения (learning_rate): "))
    max_depth = int(input("Введите максимальную глубину (max_depth): "))

    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                      random_state=42)
    model.fit(X_train, y_train)

    check_train = model.score(X_train, y_train)
    check_test = model.score(X_test, y_test)
    print(f"MSE train = {check_train}")
    print(f"MSE test = {check_test}")

    # Прогнозы для дат между началом и концом прогнозирования
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    filtered_df = df[(df['Код поста'] == post) & (df['Дата - время'].isin(date_range))]
    X_forecast = filtered_df[
        ['Код поста', 'Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Атмосферное давление',
         'Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']]
    y_forecast = model.predict(X_forecast)

    forecast_df = pd.DataFrame({'Дата - время': filtered_df['Дата - время'], 'Прогнозируемые уровни воды': y_forecast})
    real_values = df_filtered[['Дата - время', 'Уровень воды']]
    merged_df = forecast_df.merge(real_values, on='Дата - время', how='left')
    print(merged_df)
else:
    print("Нет данных для выбранной комбинации кода поста и дат прогнозирования!")
