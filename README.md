# Cyber Threat Analysis using CISSM and Honeypot Data

This repository contains a Streamlit application developed as part of a Computer Science honours thesis that merges and analyzes cyber threat data from the **CISSM Cyber Events Dataset** and **Honeypot IOC Dataset**. It provides tools for analyzing cyberattacks, attack types, and geographical distributions using various statistical models.

## Pages Overview

### 1. `merger2.py`
- **Description**: Main page of the app. Upload the CISSM dataset and Honeypot IOC dataset to merge and preprocess the data.
- **Functionality**:
  - Upload datasets.
  - Preprocess the data: merge and clean it.
  - Map attack types and countries to categories.
  - Select industries to focus on for specific analysis.
  - Generates an initial dataset for further analysis.

### 2. `eda.py` (Exploratory Data Analysis)
- **Description**: This page focuses on visualizing the merged dataset to understand the distribution of cyberattacks.
- **Functionality**:
  - Plot various distributions of cyber events (by region, attack type, etc.).
  - Visualize the relationship between variables in the dataset.
  - Analyze industry-based trends.

### 3. `stats.py`
- **Description**: Performs statistical analysis on the merged dataset.
- **Functionality**:
  - Provides correlation analysis.
  - Conducts Chi-Square, ANOVA, and other statistical tests on the cyber event data.
  - Outputs key insights from the statistical tests.

### 4. `bestArima.py`
- **Description**: This page implements the **ARIMA** model for time-series forecasting of cyberattacks.
- **Functionality**:
  - Runs ARIMA on the time-series data (frequency of cyberattacks by region).
  - Outputs forecasted trends and model performance metrics.

### 5. `bestExp.py`
- **Description**: Implements **Exponential Smoothing** for time-series forecasting.
- **Functionality**:
  - Forecasts cyberattack data using Exponential Smoothing.
  - Outputs forecasted trends and model evaluation metrics.

### 6. `bestSarima.py`
- **Description**: Implements the **SARIMA** model for seasonal time-series forecasting.
- **Functionality**:
  - Provides seasonal adjustments to time-series data.
  - Forecasts trends while considering seasonal variations in cyberattack patterns.
  - Displays the performance of the SARIMA model.

### 7. `merger.py`
- **Description**: An alternate version of the data merger script.
- **Functionality**: Similar to `merger2.py` but can be used for variations in merging techniques or data processing.

## How to Run the App

1. Navigate to the project directory and and run "streamlit run {filename}
 so for examople for eda it will be: streamlit run eda.py
