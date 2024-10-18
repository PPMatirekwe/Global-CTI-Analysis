import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Title of the App
st.title("Regional Frequency Exponential Smoothing Models for Cyber Attacks")

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

    # Dictionary to store Exponential Smoothing MSE and MAPE results
    region_performance_expo = {}

    # Function to calculate MAPE
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-9  # Avoid division by zero
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    # Loop through each region and apply Exponential Smoothing methods
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

        # Initialize a dictionary to store MSE and SMAPE for each method
        performance = {}

        # Simple Exponential Smoothing (SES)
        try:
            ses_model = SimpleExpSmoothing(train['total_attacks']).fit()
            test['ses_predictions'] = ses_model.forecast(steps=len(test))
            mse_ses = mean_squared_error(test['total_attacks'], test['ses_predictions'])
            smape_ses = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['ses_predictions'])
            performance['MSE_SES'] = mse_ses
            performance['SMAPE_SES'] = smape_ses
        except:
            st.write(f"SES failed for {region}")

        # Holt’s Linear Trend Method
        try:
            holt_model = ExponentialSmoothing(train['total_attacks'], trend='add', seasonal=None).fit()
            test['holt_predictions'] = holt_model.forecast(steps=len(test))
            mse_holt = mean_squared_error(test['total_attacks'], test['holt_predictions'])
            smape_holt = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['holt_predictions'])
            performance['MSE_Holt'] = mse_holt
            performance['SMAPE_Holt'] = smape_holt
        except:
            st.write(f"Holt’s Linear Trend failed for {region}")

        # Holt-Winters Seasonal Method
        try:
            hw_model = ExponentialSmoothing(train['total_attacks'], trend='add', seasonal='add', seasonal_periods=12).fit()
            test['hw_predictions'] = hw_model.forecast(steps=len(test))
            mse_hw = mean_squared_error(test['total_attacks'], test['hw_predictions'])
            smape_hw = symmetric_mean_absolute_percentage_error(test['total_attacks'], test['hw_predictions'])
            performance['MSE_Holt_Winters'] = mse_hw
            performance['SMAPE_Holt_Winters'] = smape_hw
        except:
            st.write(f"Holt-Winters Seasonal Method failed for {region}")

        # Store performance for each region
        region_performance_expo[region] = performance

        # Plotting the results for Holt-Winters (as it's likely the most appropriate for seasonal data)
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train['total_attacks'], label='Training Data')
        plt.plot(test.index, test['total_attacks'], label='Test Data (Observed)', color='blue')
        plt.plot(test.index, test['ses_predictions'], label='SES Predictions', color='green', linestyle='--')
        plt.plot(test.index, test['holt_predictions'], label='Holt Predictions', color='red', linestyle='--')
        plt.plot(test.index, test['hw_predictions'], label='Holt-Winters Predictions', color='orange', linestyle='--')
        plt.title(f'Exponential Smoothing Methods: {region} (Monthly)')
        plt.xlabel('Date')
        plt.ylabel('Total Attacks')
        plt.legend()
        st.pyplot(plt)

    # Convert the Exponential Smoothing region performance dictionary to a DataFrame
    performance_df_expo = pd.DataFrame(region_performance_expo).T

    # Display the DataFrame
    st.write("Exponential Smoothing model performance metrics for each region:")
    st.dataframe(performance_df_expo)

    # Option to download the performance DataFrame as CSV
    csv_expo = performance_df_expo.to_csv(index=True).encode('utf-8')
    st.download_button(label="Download Exponential Smoothing Performance CSV", data=csv_expo, file_name="exponential_smoothing_performance.csv", mime='text/csv')

else:
    st.write("Please upload the AlienVault dataset to proceed.")
