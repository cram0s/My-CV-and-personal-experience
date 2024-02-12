import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
from warnings import filterwarnings

filterwarnings('ignore')

# Загрузка датасетов
data = pd.read_parquet('df_competitors.parquet')
weather = pd.read_parquet('weather_df.parquet')

# Группировка данных по продавцу, продукту и дате
data_grouped = data.groupby(['competitor', 'product', 'date'])['price'].mean().reset_index()

# Создание временного ряда для каждого продукта и продавца
time_series = data_grouped.pivot_table(index='date', columns=['competitor', 'product'], values='price', fill_value=0)

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series)

# Добавление данных о погоде
data_with_weather = pd.merge(data_grouped, weather, on='date', how='left')

# Добавление стоимости за 5 предыдущих дат
look_back = 5
for i in range(1, look_back + 1):
    data_with_weather[f'price_lag_{i}'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(i)

# Удаление строк с отсутствующими данными
data_with_weather = data_with_weather.dropna()

# Убери строки, где цена меньше или равна нулю
data_with_weather = data_with_weather[data_with_weather['price'] > 0]

# Создание временных рядов для каждого места и продавца
time_series_with_weather = data_with_weather.pivot_table(index='date', columns=['place', 'competitor', 'product'],
                                                         values='price', fill_value=0)

print(time_series_with_weather)


# Подготовка данных для GRU
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)


look_back = 30
X, y = create_sequences(scaled_data, look_back)

# Разбивка данных на обучающую и тестовую выборки
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(GRU(120, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(GRU(60, return_sequences=True))
model.add(GRU(30))
model.add(Dense(X_train.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=15, batch_size=32)

# Прогнозирование цен на тестовой выборке
y_pred = model.predict(X_test)

# Обратное масштабирование данных
y_pred = y_pred.reshape(-1, X_train.shape[2])
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Оценка производительности модели
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Вывод по 10 предсказанных значений для каждого продавца и города
for competitor in time_series.columns.get_level_values(0).unique():
    for place in time_series.columns.get_level_values(1).unique():
        y_pred_comp_place = y_pred[:, (time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        y_test_comp_place = y_test[:, (time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        dates_comp_place = data['date'].iloc[-5:].values
        print(f"Продавец: {competitor}, Город: {place}")
        for i in range(5):
            print(
                f"Дата: {dates_comp_place[i]}, Предсказанное значение: {y_pred_comp_place[i][0]}, Реальное значение: {y_test_comp_place[i][0]}")
