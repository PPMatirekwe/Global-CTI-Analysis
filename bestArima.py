import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Title of the App
st.title("Regional Frequency ARIMA Model for Cyber Attacks")

# File upload for two datasets (optional)
uploaded_file_1 = st.file_uploader("Upload First Dataset (Optional)", type=["csv"])
uploaded_file_2 = st.file_uploader("Upload Second Dataset (Required)", type=["csv"])

if uploaded_file_2:
    # Load the required AlienVault dataset
    df_alienvault = pd.read_csv(uploaded_file_2)
    df_alienvault['evtDate'] = pd.to_datetime(df_alienvault['evtDate'], infer_datetime_format=True, errors='coerce')
    df_alienvault = df_alienvault.dropna(subset=['evtDate'])  # Drop rows with missing 'evtDate'

    # Check if an optional CISSM dataset is uploaded for merging
    if uploaded_file_1:
        df_cissm = pd.read_csv(uploaded_file_1)
        df_cissm['evtDate'] = pd.to_datetime(df_cissm['evtDate'], infer_datetime_format=True, errors='coerce')
        df_cissm = df_cissm.dropna(subset=['evtDate'])  # Drop rows with missing 'evtDate'
        
        # Merge datasets
        merge_option = st.checkbox("Merge CISSM and AlienVault datasets?")
        if merge_option:
            df_combined = pd.concat([df_cissm, df_alienvault], ignore_index=True)
        else:
            df_combined = df_alienvault
    else:
        df_combined = df_alienvault

    # Fill missing dates with zero attacks and resample by day
    df_combined = df_combined.set_index('evtDate').groupby('region').resample('D').size().reset_index(name='total_attacks')

    # List of unique regions
    regions = df_combined['region'].unique()

    # Dictionary to store ARIMA MSE and MAPE results
    region_performance = {}

    # Function to check stationarity using the Augmented Dickey-Fuller test
    def adf_test(series):
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        return p_value

    # Function to calculate MAPE
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-9  # Avoid division by zero
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    # Loop through each region, check for stationarity, and apply ARIMA or auto-ARIMA
    for region in regions:
        df_regional = df_combined[df_combined['region'] == region].set_index('evtDate')
        df_regional = df_regional.resample('M').sum()  # Resample to monthly data

        # Skip regions with less than 24 months of data
        if len(df_regional) < 24:
            st.write(f"Skipping region {region} due to insufficient data.")
            continue

        # Train-test split: 80% training, 20% testing
        train_size = int(len(df_regional) * 0.8)
        train, test = df_regional[:train_size], df_regional[train_size:]

        # Check stationarity
        p_value = adf_test(train['total_attacks'])
        if p_value > 0.05:
            st.write(f"{region} is non-stationary, differencing required.")

        # Try auto-ARIMA first
        try:
            auto_arima_model = auto_arima(train['total_attacks'], seasonal=True, stepwise=True, suppress_warnings=True)
            test['arima_predictions_auto'] = auto_arima_model.predict(n_periods=len(test))
        except:
            st.write(f"Auto-ARIMA failed for {region}, falling back to grid search.")

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
            'Best_ARIMA_Order': best_order,
            'MSE_Grid_ARIMA': mse_grid,
            'SMAPE_Grid_ARIMA': smape_grid,
            'MSE_Auto_ARIMA': mse_auto,
            'SMAPE_Auto_ARIMA': smape_auto
        }

    # Convert the region performance dictionary to a DataFrame
    performance_df = pd.DataFrame(region_performance).T

    # Display the best ARIMA model parameters and results
    st.write("Best ARIMA model parameters and performance metrics for each region:")
    st.dataframe(performance_df)

    # Option to download the performance DataFrame as CSV
    csv = performance_df.to_csv(index=True).encode('utf-8')
    st.download_button(label="Download ARIMA Performance CSV", data=csv, file_name="arima_regional_performance.csv", mime='text/csv')

else:
    st.write("Please upload the AlienVault dataset to proceed.")
