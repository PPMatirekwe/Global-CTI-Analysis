import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# Title of the App
st.title("Regional Frequency SARIMAX Model for Cyber Attacks")

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

    # Dictionary to store SARIMAX MSE and MAPE results
    region_performance_sarimax = {}

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

    # Loop through each region, check for stationarity, and apply SARIMAX or auto-SARIMA
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

        # Initialize flag to track if auto_arima succeeds
        auto_arima_success = False

        # Try auto-ARIMA first to get SARIMA parameters
        try:
            auto_sarima_model = auto_arima(train['total_attacks'], seasonal=True, m=12, stepwise=True, suppress_warnings=True, trace=True)
            test['sarima_predictions_auto'] = auto_sarima_model.predict(n_periods=len(test))
            auto_arima_success = True
        except:
            st.write(f"Auto-SARIMA failed for {region}, falling back to grid search.")

        # Manual Grid Search for SARIMAX (p,d,q)(P,D,Q,s)
        best_sarima_mse = float("inf")
        best_sarima_order, best_sarima_seasonal_order = None, None
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    for P in range(0, 2):
                        for D in range(0, 2):
                            for Q in range(0, 2):
                                try:
                                    sarima_order = (p, d, q)
                                    seasonal_order = (P, D, Q, 12)  # Seasonal order for monthly data
                                    sarima_model = SARIMAX(train['total_attacks'], order=sarima_order, seasonal_order=seasonal_order)
                                    sarima_fit = sarima_model.fit(disp=False)
                                    predictions = sarima_fit.forecast(steps=len(test))
                                    mse = mean_squared_error(test['total_attacks'], predictions)
                                    if mse < best_sarima_mse:
                                        best_sarima_mse = mse
                                        best_sarima_order = sarima_order
                                        best_sarima_seasonal_order = seasonal_order
                                except:
                                    continue

        # Fit the best SARIMAX model from the grid search
        sarima_model = SARIMAX(train['total_attacks'], order=best_sarima_order, seasonal_order=best_sarima_seasonal_order)
        sarima_fit = sarima_model.fit(disp=False)
        test['sarima_predictions_grid'] = sarima_fit.forecast(steps=len(test))

        # Calculate performance metrics for both models (if auto_arima was successful)
        if auto_arima_success:
            mse_auto_sarima = mean_squared_error(test['total_attacks'], test['sarima_predictions_auto'])
            smape_auto_sarima = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['sarima_predictions_auto'])
        else:
            mse_auto_sarima = np.nan
            smape_auto_sarima = np.nan

        mse_grid_sarima = mean_squared_error(test['total_attacks'], test['sarima_predictions_grid'])
        smape_grid_sarima = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['sarima_predictions_grid'])

        region_performance_sarimax[region] = {
            'MSE_Auto_SARIMA': mse_auto_sarima,
            'SMAPE_Auto_SARIMA': smape_auto_sarima,
            'MSE_Grid_SARIMA': mse_grid_sarima,
            'SMAPE_Grid_SARIMA': smape_grid_sarima
        }

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train['total_attacks'], label='Training Data')
        plt.plot(test.index, test['total_attacks'], label='Test Data (Observed)', color='blue')

        if auto_arima_success:
            plt.plot(test.index, test['sarima_predictions_auto'], label='Auto-SARIMA Predictions', color='green', linestyle='--')

        plt.plot(test.index, test['sarima_predictions_grid'], label='Grid-SARIMA Predictions', color='red', linestyle='--')
        plt.title(f'SARIMA: Observed vs Predicted Cyber Attacks (Monthly) - {region}')
        plt.xlabel('Date')
        plt.ylabel('Total Attacks')
        plt.legend()
        st.pyplot(plt)

    # Convert the SARIMAX region performance dictionary to a DataFrame
    performance_df_sarimax = pd.DataFrame(region_performance_sarimax).T

    # Display the DataFrame
    st.write("Best SARIMAX model parameters and performance metrics for each region:")
    st.dataframe(performance_df_sarimax)

    # Option to download the performance DataFrame as CSV
    csv_sarimax = performance_df_sarimax.to_csv(index=True).encode('utf-8')
    st.download_button(label="Download SARIMAX Performance CSV", data=csv_sarimax, file_name="sarimax_regional_performance.csv", mime='text/csv')

else:
    st.write("Please upload the AlienVault dataset to proceed.")
