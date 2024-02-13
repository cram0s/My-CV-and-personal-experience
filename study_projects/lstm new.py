import pandas as pd
import pymysql.cursors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
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
df['Уровень воды(t-1)'] = df.groupby('Код поста')['Уровень воды'].shift(1)
df['Уровень воды(t-2)'] = df.groupby('Код поста')['Уровень воды'].shift(2)
df['Уровень воды(t-3)'] = df.groupby('Код поста')['Уровень воды'].shift(3)

# Заполнение пропущенных значений
df[['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']] = df.groupby('Код поста')[
    ['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']].fillna(0)

con.close()


def train_model(df, post):
    # Фильтрация данных по коду гидропоста
    df_filtered = df[df['Код поста'] == post]

    if not df_filtered.empty:

        # Заполнение пропущенных значений
        df_filtered[['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']] = df_filtered[
            ['Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']].fillna(0)

        X = df_filtered[['Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Атмосферное давление',
                         'Уровень воды(t-1)', 'Уровень воды(t-2)', 'Уровень воды(t-3)']].astype(float)
        y = df_filtered['Уровень воды'].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(1, X_train_scaled.shape[2]), return_sequences=True),
            layers.LSTM(32, activation='relu', return_sequences=True),
            layers.LSTM(16, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

        return model, scaler


def predict_water_level(model, scaler, input_data):
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    predictions = model.predict(input_data)
    return predictions


# Ввод данных от пользователя
post = input("Введите код поста: ")
start_date = input("Введите дату начала прогнозирования (ГГГГ-ММ-ДД): ")
end_date = input("Введите дату конца прогнозирования (ГГГГ-ММ-ДД): ")

# Фильтрация данных по датам
df['Дата - время'] = pd.to_datetime(df['Дата - время'])
df_filtered = df[(df['Код поста'] == post) & (df['Дата - время'] >= start_date) & (df['Дата - время'] <= end_date)]

if not df_filtered.empty:
    model, scaler = train_model(df, post)

    # Фактические значения уровня воды на выбранном периоде
    actual_values = df_filtered['Уровень воды'].values

    # Ввод данных для прогнозирования
    input_data = df_filtered[['Температура воздуха', 'Скорость ветра', 'Количество осадков',
                             'Атмосферное давление', 'Уровень воды(t-1)', 'Уровень воды(t-2)',
                             'Уровень воды(t-3)']].astype(float)

    predictions = predict_water_level(model, scaler, input_data)

    # Вывод прогнозов и сравнение с фактическими значениями
    result = pd.DataFrame({'Фактические значения': actual_values, 'Прогнозные значения': predictions.ravel()})
    print(result)
else:
    print("Данные для выбранного гидропоста и диапазона дат отсутствуют.")