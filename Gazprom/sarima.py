import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Загрузка датасетов
data = pd.read_parquet('df_competitors.parquet')
weather = pd.read_parquet('weather_df.parquet')

# Merge based on 'date' and 'place'
data_with_weather = pd.merge(data, weather, on=['date', 'place'], how='left')

# Добавление стоимости за 5 предыдущих дат
look_back = 10
for i in range(1, look_back + 1):
    data_with_weather[f'price_lag_{i}'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(i)
data_with_weather['price_trend'] = np.where(data_with_weather.groupby(['competitor', 'product'])['price'].diff() > 0, 1, 0)
data_with_weather['target'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(-1)

# Удаление строк с отсутствующими данными
data_with_weather = data_with_weather.dropna()

# Выбор данных для конкретного конкурента, города и товара
competitor = 'Арториас&Co'
place = 'Анор Лондо'
product = 'Целебные травы'
data_filter = data_with_weather[(data_with_weather['competitor'] == competitor) &
                                (data_with_weather['place'] == place) &
                                (data_with_weather['product'] == product)]

# Создание временного ряда
time_series = data_filter.set_index('date')['price']

# Визуализация временного ряда
plt.figure(figsize=(12, 6))
time_series.plot()
plt.title('Time Series - Price')
plt.show()

# Разделение данных на обучающую и тестовую выборки
split_index = int(len(time_series) * 0.8)
train_data, test_data = time_series[:split_index], time_series[split_index:]

# Построение модели SARIMA
order = (1, 1, 1)  # порядок ARIMA
seasonal_order = (1, 1, 1, 12)  # порядок сезонной составляющей
model = sm.tsa.SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                      enforce_invertibility=False)

# Обучение модели
results = model.fit()

# Прогноз
start = len(train_data)
end = start + len(test_data) - 1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')

# Визуализация результатов
plt.figure(figsize=(12, 6))
train_data.plot(label='Train')
test_data.plot(label='Test')
predictions.plot(label='SARIMA Predictions')
plt.title('SARIMA Model Forecast')
plt.legend()
plt.show()
