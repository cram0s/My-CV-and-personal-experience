import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from warnings import filterwarnings

filterwarnings('ignore')

# Загрузка датасетов
data = pd.read_parquet('df_competitors.parquet')
weather = pd.read_parquet('weather_df.parquet')

# Merge based on 'date' and 'place'
data_with_weather = pd.merge(data, weather, on=['date', 'place'], how='left')

# Добавление стоимости за 5 предыдущих дат
look_back = 20
for i in range(1, look_back + 1):
    data_with_weather[f'price_lag_{i}'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(i)

# Добавление признака тренда (скользящее среднее)
data_with_weather['trend'] = data_with_weather.groupby(['competitor', 'product'])['price'].transform(lambda x: x.rolling(window=10).mean())

# Добавление признака общей динамики изменения цен
data_with_weather['price_change'] = data_with_weather.groupby(['competitor', 'product'])['price'].pct_change()

# Удаление строк с отсутствующими данными
data_with_weather = data_with_weather.dropna()

# Create an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Competitor', 'Place', 'Product', 'Date', 'Predicted_Value'])

# Список уникальных значений для каждого параметра
competitors = data_with_weather['competitor'].unique()
places = data_with_weather['place'].unique()
products = data_with_weather['product'].unique()

# Итерация по уникальным датам
for competitor in competitors:
    for place in places:
        for product in products:
            print(f"Predicting for: Competitor={competitor}, Place={place}, Product={product}")
            # Фильтрация данных для конкретного конкурента, города и товара
            data_filter = data_with_weather[(data_with_weather['competitor'] == competitor) &
                                            (data_with_weather['place'] == place) &
                                            (data_with_weather['product'] == product)]
            # Проверка, достаточно ли данных для предсказания
            if len(data_filter) >= look_back:
                # Обновление данных для предсказания
                X = data_filter[
                    ['price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
                     'price_lag_5', 'price_lag_6', 'price_lag_7', 'price_lag_8',
                     'price_lag_9', 'price_lag_10', 'trend', 'price_change', 'hot', 'rain', 'snow']]

                y = data_filter['price']

                # Разбивка данных на обучающую и тестовую выборки
                split_index = int(len(data_filter) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                # MinMax Scaling
                scaler_X = MinMaxScaler(feature_range=(0, 1))
                scaler_y = MinMaxScaler(feature_range=(0, 1))

                X_train_scaled = scaler_X.fit_transform(X_train.values)
                X_test_scaled = scaler_X.transform(X_test.values)

                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
                y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

                # Обучение модели градиентного бустинга
                model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.001, max_depth=12, random_state=42)
                model.fit(X_train_scaled, y_train_scaled.ravel())

                # Прогноз на тестовой выборке
                y_pred = model.predict(X_test_scaled)

                # Обратное масштабирование данных
                y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                y_test_original = scaler_y.inverse_transform(y_test_scaled)

                # Вычисление RMSE
                rmse = sqrt(mean_squared_error(y_test_original, y_pred_original))
                print(f"RMSE: {rmse}")

                # Прогноз на следующие 90 дней
                # Прогноз на следующие 90 дней
                X_pred = X_test_scaled[-1].reshape(1, X_test_scaled.shape[1])
                for date in pd.date_range(start=data_with_weather['date'].max() + pd.Timedelta(days=1), periods=90):
                    y_pred = model.predict(X_pred)
                    X_pred = np.append(X_pred[:, 1:], y_pred.reshape((1, 1)), axis=1)

                    # Обратное масштабирование данных
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

                    # Запись предсказаний в датафрейм
                    predictions_df = pd.concat([
                        predictions_df,
                        pd.DataFrame({
                            'Competitor': [competitor],
                            'Place': [place],
                            'Product': [product],
                            'Date': date,
                            'Predicted_Value': [y_pred[0][0]],
                        })
                    ], ignore_index=True)
                predictions_df.to_excel('predicted_prices_gb.xlsx', index=False)
