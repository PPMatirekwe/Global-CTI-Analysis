#@title Total regional frequency ARIMA


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Load the datasets
#df_cissm = pd.read_csv('/content/cleaned_cissm_with_region.csv')
#df_alienvault = pd.read_csv('/content/updated_alienvault_with_regions.csv')

# Merge the datasets (l want to make this optional , otherwise process them separately)
df_combined = pd.read_csv('/content/updated_alienvault_with_regions.csv')

# Convert 'evtDate' to datetime format with automatic format inference
df_combined['evtDate'] = pd.to_datetime(df_combined['evtDate'], infer_datetime_format=True, errors='coerce')

# Check for any rows where the date conversion failed (NaT values) and handle them
#df_combined = df_combined.dropna(subset=['evtDate'])

# Now proceed with the rest of your analysis


# Fill missing dates with zero attacks
df_combined = df_combined.set_index('evtDate').groupby('region').resample('D').size().reset_index(name='total_attacks')

# List of unique regions
regions = df_combined['region'].unique()

# Function to check stationarity using the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    return p_value

# Dictionary to store ARIMA MSE and MAPE results
region_performance = {}

# Function to calculate MAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-9
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

# Loop through each region, check for stationarity, and apply ARIMA or auto-ARIMA
for region in regions:
    df_regional = df_combined[df_combined['region'] == region].set_index('evtDate')
    df_regional = df_regional.resample('M').sum()  # Resample to monthly data

    # Skip regions with less than 24 months of data
    if len(df_regional) < 24:
        print(f"Skipping region {region} due to insufficient data.")
        continue

    # Train-test split: 80% training, 20% testing
    train_size = int(len(df_regional) * 0.8)
    train, test = df_regional[:train_size], df_regional[train_size:]

    # Check stationarity
    p_value = adf_test(train['total_attacks'])
    if p_value > 0.05:
        print(f"{region} is non-stationary, differencing required.")

    # Try auto-ARIMA first
    try:
        auto_arima_model = auto_arima(train['total_attacks'], seasonal=True, stepwise=True, suppress_warnings=True, trace=True)
        test['arima_predictions_auto'] = auto_arima_model.predict(n_periods=len(test))
    except:
        print(f"Auto-ARIMA failed for {region}, falling back to grid search.")

    # Manual Grid Search for ARIMA if auto-ARIMA fails or for comparison
    best_score, best_order = float("inf"), None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    order = (p, d, q)
                    model = ARIMA(train['total_attacks'], order=order)
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=len(test))
                    mse = mean_squared_error(test['total_attacks'], predictions)
                    if mse < best_score:
                        best_score, best_order = mse, order
                except:
                    continue

    # Fit the best ARIMA model from the grid search
    model = ARIMA(train['total_attacks'], order=best_order)
    model_fit = model.fit()
    test['arima_predictions_grid'] = model_fit.forecast(steps=len(test))

    # Calculate performance metrics for both models
    mse_auto = mean_squared_error(test['total_attacks'], test['arima_predictions_auto'])
    mse_grid = mean_squared_error(test['total_attacks'], test['arima_predictions_grid'])

    smape_auto = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['arima_predictions_auto'])
    smape_grid = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['arima_predictions_grid'])

    region_performance[region] = {
        'MSE_Auto_ARIMA': mse_auto,
        'SMAPE_Auto_ARIMA': smape_auto,
        'MSE_Grid_ARIMA': mse_grid,
        'SMAPE_Grid_ARIMA': smape_grid
    }

    # Plotting the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(train.index, train['total_attacks'], label='Training Data')
    # plt.plot(test.index, test['total_attacks'], label='Test Data (Observed)', color='blue')
    # plt.plot(test.index, test['arima_predictions_auto'], label='Auto-ARIMA Predictions', color='green', linestyle='--')
    # plt.plot(test.index, test['arima_predictions_grid'], label='Grid-ARIMA Predictions', color='red', linestyle='--')
    # plt.title(f'ARIMA: Observed vs Predicted Cyber Attacks (Monthly) - {region}')
    # plt.xlabel('Date')
    # plt.ylabel('Total Attacks')
    # plt.legend()
    # plt.show()

    # Convert the region performance dictionary to a DataFrame
    performance_df = pd.DataFrame(region_performance).T

    # Display the DataFrame
    print(performance_df)

    # Optionally, save the DataFrame as a CSV file if you want to analyze it later
    performance_df.to_csv('arima_regional_forecasting_results.csv', index=True)

    # To download the CSV file in a Colab environment, use:
    from google.colab import files
    files.download('arima_regional_forecasting_results.csv')
