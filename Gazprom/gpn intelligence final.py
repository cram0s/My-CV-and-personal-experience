import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from warnings import filterwarnings
from keras.layers import Dropout

filterwarnings('ignore')

# Загрузка датасетов
data = pd.read_parquet('df_competitors.parquet')
weather = pd.read_parquet('weather_df.parquet')

data_with_weather = pd.merge(data, weather, on=['date', 'place'], how='left')

# Добавление стоимости за 5 предыдущих дат
look_back = 10
for i in range(1, look_back + 1):
    data_with_weather[f'price_lag_{i}'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(i)

# Добавление скользящих средних за последние 25, 50, 100 дней
data_with_weather['rolling_mean_25'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(1).rolling(
    window=25).mean()
data_with_weather['rolling_mean_50'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(1).rolling(
    window=50).mean()
data_with_weather['rolling_mean_100'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(1).rolling(
    window=100).mean()
data_with_weather['price_trend'] = np.where(data_with_weather.groupby(['competitor', 'product'])['price'].diff() > 0, 1, 0)

# Удаление строк с отсутствующими данными
data_with_weather = data_with_weather.dropna()


predictions_df = pd.DataFrame(columns=['Competitor', 'Place', 'Product', 'Date', 'Predicted_Value'])

competitors = data_with_weather['competitor'].unique()
places = data_with_weather['place'].unique()
products = data_with_weather['product'].unique()

for place in places:
    for product in products:
        print(f"Predicting for:  Place={place}, Product={product}")

        data_filter = data_with_weather[
            (data_with_weather['place'] == place) &
            (data_with_weather['product'] == product)]

        # Обновление данных для предсказания
        X = data_filter[
            ['price_trend', 'rolling_mean_25', 'rolling_mean_50', 'hot', 'rain', 'snow',
             'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lag_5', 'price_lag_6',
             'price_lag_7', 'price_lag_8', 'price_lag_9', 'price_lag_10',
             ]]
        y = data_filter['price']

        # Разбивка данных на обучающую и тестовую выборки
        split_index = int(len(data_filter) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_train_scaled = scaler_X.fit_transform(X_train.values)
        X_test_scaled = scaler_X.transform(X_test.values)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        # Обучение модели
        model = Sequential()

        if place == 'Нокрон' :
            # Customize the model architecture for Нокрон
            model.add(LSTM(240, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))
            model.add(LSTM(120, return_sequences=True))
            model.add(LSTM(60))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled),
                      epochs=40, batch_size=32, verbose=1)
        else:
            # Default model architecture for other cases
            model.add(LSTM(120, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))
            model.add(LSTM(60, return_sequences=True))
            model.add(LSTM(30))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled),
                      epochs=30, batch_size=32, verbose=1)



        # Прогноз на тестовой выборке
        y_pred = model.predict(X_test_scaled)

        # Обратное масштабирование данных
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_test_original = scaler_y.inverse_transform(y_test_scaled)

        # Вычисление RMSE
        rmse = sqrt(mean_squared_error(y_test_original, y_pred_original))
        print(f"RMSE: {rmse}")

        # Прогноз на следующие 90 дней
        X_pred = X_test_scaled[-1].reshape(1, X_test_scaled.shape[1], 1)
        for competitor in competitors:
            # Найти последние доступные цены для текущего конкурента
            last_prices = data_filter[data_filter['competitor'] == competitor]['price'].values[-look_back:]

            # Использовать последние цены для начала прогнозирования
            X_pred[:, :, 0] = scaler_X.transform(np.array([[
                data_filter['price_trend'].values[-1],
                data_filter['rolling_mean_25'].values[-1],
                data_filter['rolling_mean_50'].values[-1],
                data_filter['hot'].values[-1],
                data_filter['rain'].values[-1],
                data_filter['snow'].values[-1],
                *last_prices
            ]]))

            for date in pd.date_range(start=data_with_weather['date'].max() + pd.Timedelta(days=1), periods=90):
                y_pred = model.predict(X_pred)
                X_pred = np.append(X_pred[:, 1:, :], y_pred.reshape((1, 1, 1)), axis=1)

                y_pred = scaler_y.inverse_transform(y_pred)

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


        predictions_df.to_excel('predicted_prices1.xlsx', index=False)

predictions_df = predictions_df.sort_values(by=['Competitor', 'Place', 'Product', 'Date']).reset_index(drop=True)

pivot_df = predictions_df.pivot(index=['Place', 'Product', 'Date'], columns='Competitor',
                                values='Predicted_Value').reset_index()

# Сохранение повернутого DataFrame в Excel
pivot_df.to_excel('predicted_prices_pivoted1.xlsx', index=False)

# Загрузка данных из Excel файла
df = pd.read_excel('predicted_prices_pivoted1.xlsx')

# Преобразование 'Date' в datetime
df['Date'] = pd.to_datetime(df['Date'])

# Разбивка данных на блоки по три строки
blocks = [df.iloc[i:i + 3].copy() for i in range(0, len(df), 3)]

# Создание списка для хранения результатов
min_values = []

# Обработка  блока
for block in blocks:
    # Нахождение максимального значения в каждой строке (кроме 'Date')
    max_values = block.drop('Date', axis=1).apply(pd.to_numeric, errors='coerce').max(axis=1)

    # Создание списков с тремя значениями максимума
    for i in range(0, len(max_values), 3):
        max_values_set = max_values[i:i + 3].tolist()

        # Нахождение минимума в каждом списке
        min_value = min(max_values_set)

        # Добавление минимального значения в список
        min_values.append(min_value)

# Преобразование списка в DataFrame
result_df = pd.DataFrame({'min_value': min_values})
# Добавление 15% к столбцу min_value
result_df['my_price'] = result_df['min_value'] * 1.18
df['my_price'] = result_df['my_price'].repeat(3).reset_index(drop=True)

# Цикл для сравнения и изменения значений столбца 'my_price'
for i in range(1, len(result_df)):
    # Проверка условия: разница больше чем 1 и номер строки кратен 30
    if abs(result_df['my_price'].iloc[i] - result_df['my_price'].iloc[i-1]) > 1 and (i) % 30 == 0:
        # Ничего не делаем, так как условие выполняется
        pass
    else:
        # Проверка разницы больше чем 1 в блоке else
        if abs(result_df['my_price'].iloc[i] - result_df['my_price'].iloc[i-1]) > 1:
            # Изменяем значение на my_price[i] = my_price[i-1] + 1 или -1
            result_df['my_price'].iloc[i] = result_df['my_price'].iloc[i-1] + 1 if result_df['my_price'].iloc[i] > result_df['my_price'].iloc[i-1] else result_df['my_price'].iloc[i-1] - 1


print(result_df)


# Добавление столбца my_price в основной датафрейм
df['my_price'] = result_df['my_price'].repeat(3).reset_index(drop=True)
diff_mask = np.abs(df['my_price'].diff()) > 1

# Выводим строки, где значения отличаются больше чем на 1
print(df[diff_mask])
df[diff_mask].to_excel('anomaly.xlsx')
# Вывод окончательного результата
df = df.iloc[:, [0, 1, 2, -1]]
df.to_excel('final1.xlsx')
df.to_parquet('final1.parquet')
