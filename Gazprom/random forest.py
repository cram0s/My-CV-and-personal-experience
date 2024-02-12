import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Загрузка датасетов
data = pd.read_parquet('df_competitors.parquet')
print(data.shape)
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

# Подготовка данных для случайного леса
def create_sequences_rf(data, look_back=5):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y)

look_back_rf = 30
X_rf, y_rf = create_sequences_rf(scaled_data, look_back_rf)

# Разбивка данных на обучающую и тестовую выборки
train_size_rf = int(len(X_rf) * 0.8)
X_train_rf, X_test_rf = X_rf[:train_size_rf], X_rf[train_size_rf:]
y_train_rf, y_test_rf = y_rf[:train_size_rf], y_rf[train_size_rf:]

# Инициализация и обучение модели случайного леса
rf_model = RandomForestRegressor(n_estimators=15, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Прогнозирование цен на тестовой выборке
y_pred_rf = rf_model.predict(X_test_rf)

# Обратное масштабирование данных
y_pred_rf = scaler.inverse_transform(y_pred_rf.reshape(-1, time_series.shape[1]))

# Оценка производительности модели
rmse_rf = sqrt(mean_squared_error(scaler.inverse_transform(y_test_rf), y_pred_rf))
print(f'RMSE for Random Forest: {rmse_rf}')

# Вывод по 10 предсказанных значений для каждого продавца и города
for competitor in time_series.columns.get_level_values(0).unique():
    for place in time_series.columns.get_level_values(1).unique():
        y_pred_comp_place_rf = y_pred_rf[:, (time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        y_test_comp_place_rf = scaler.inverse_transform(y_test_rf)[:, (time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        dates_comp_place_rf = data['date'].iloc[-5:].values
        print(f"Продавец: {competitor}, Город: {place}")
        for i in range(5):
            print(
                f"Дата: {dates_comp_place_rf[i]}, Предсказанное значение: {y_pred_comp_place_rf[i][0]}, Реальное значение: {y_test_comp_place_rf[i][0]}")
