import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from warnings import filterwarnings

filterwarnings('ignore')

# Load datasets
data = pd.read_parquet('df_competitors.parquet')
weather = pd.read_parquet('weather_df.parquet')

# Merge based on 'date' and 'place'
data_with_weather = pd.merge(data, weather, on=['date', 'place'], how='left')

# Add the cost for the last 20 days as features
look_back = 20
for i in range(1, look_back + 1):
    data_with_weather[f'price_lag_{i}'] = data_with_weather.groupby(['competitor', 'product'])['price'].shift(i)

# Add trend and price change features
data_with_weather['trend'] = data_with_weather.groupby(['competitor', 'product'])['price'].transform(lambda x: x.rolling(window=10).mean())
data_with_weather['price_change'] = data_with_weather.groupby(['competitor', 'product'])['price'].pct_change()

# Drop rows with missing data
data_with_weather = data_with_weather.dropna()

# Create an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Competitor', 'Place', 'Product', 'Date', 'Predicted_Value'])

# Get unique values for each parameter
competitors = data_with_weather['competitor'].unique()
places = data_with_weather['place'].unique()
products = data_with_weather['product'].unique()

# Iterate over unique dates
for competitor in competitors:
    for place in places:
        for product in products:
            print(f"Predicting for: Competitor={competitor}, Place={place}, Product={product}")
            # Filter data for a specific competitor, place, and product
            data_filter = data_with_weather[(data_with_weather['competitor'] == competitor) &
                                            (data_with_weather['place'] == place) &
                                            (data_with_weather['product'] == product)]
            # Check if there is enough data for prediction
            if len(data_filter) >= look_back:
                # Update data for prediction
                X = data_filter[
                    ['price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
                     'price_lag_5', 'price_lag_6', 'price_lag_7', 'price_lag_8',
                     'price_lag_9', 'price_lag_10', 'trend', 'price_change', 'hot', 'rain', 'snow']]
                y = data_filter['price']

                # Split data into training and testing sets
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

                # Create and train the model
                model = LinearRegression()
                model.fit(X_train_scaled, y_train_scaled)

                # Predict on the test set
                y_pred = model.predict(X_test_scaled)

                # Inverse transform for original scale
                y_pred_original = scaler_y.inverse_transform(y_pred)
                y_test_original = scaler_y.inverse_transform(y_test_scaled)

                # Calculate RMSE
                rmse = sqrt(mean_squared_error(y_test_original, y_pred_original))
                print(f"RMSE: {rmse}")

                # Predict for the next 90 days
                X_pred = X_test_scaled[-1].reshape(1, X_test_scaled.shape[1])
                for date in pd.date_range(start=data_with_weather['date'].max() + pd.Timedelta(days=1), periods=90):
                    y_pred = model.predict(X_pred)
                    X_pred = np.append(X_pred[:, 1:], y_pred.reshape((1, 1)), axis=1)

                    # Inverse transform for original scale
                    y_pred = scaler_y.inverse_transform(y_pred)

                    # Record predictions in the dataframe
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
                predictions_df.to_excel('predicted_prices.xlsx', index=False)
