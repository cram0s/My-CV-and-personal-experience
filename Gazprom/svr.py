import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
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

# Подготовка данных для SVR
X, y = scaled_data[:, :-1], scaled_data[:, -1]

# Разбивка данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели SVR
svr_model = make_pipeline(MinMaxScaler(), SVR(C=1.0, epsilon=0.2))
svr_model.fit(X_train, y_train)

# Прогнозирование цен на тестовой выборке
y_pred_svr = svr_model.predict(X_test)

# Обратное масштабирование данных
y_pred_svr = y_pred_svr.reshape(-1, 1)
y_pred_svr = scaler.inverse_transform(y_pred_svr)[:, -1]  # Исправление формы

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))[:, -1]  # Исправление формы

# Оценка производительности модели
rmse_svr = sqrt(mean_squared_error(y_test, y_pred_svr))
print(f'RMSE for SVR: {rmse_svr}')

# Вывод по 10 предсказанных значений для каждого продавца и города
for competitor in time_series.columns.get_level_values(0).unique():
    for place in time_series.columns.get_level_values(1).unique():
        y_pred_comp_place_svr = y_pred_svr[(time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        y_test_comp_place_svr = y_test[(time_series.columns.get_level_values(0) == competitor) & (
                time_series.columns.get_level_values(1) == place)]
        dates_comp_place_svr = data['date'].iloc[-5:].values
        print(f"Продавец: {competitor}, Город: {place}")
        for i in range(5):
            print(
                f"Дата: {dates_comp_place_svr[i]}, Предсказанное значение: {y_pred_comp_place_svr[i]}, Реальное значение: {y_test_comp_place_svr[i]}")
