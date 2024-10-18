#Here l am trying to implement the code for stats test for significant relations 

import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
import numpy as np

# Function to compute Cramér's V
def cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

# Function to perform Chi-Square test
def chi_square_test(data, col1, col2):
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value

# Function to perform Mann-Whitney U test using event frequency
def mann_whitney_test(data, group1, group2):
    group1_data = data[data['region'] == group1]['event_type'].value_counts().reindex(data['event_type'].unique(), fill_value=0)
    group2_data = data[data['region'] == group2]['event_type'].value_counts().reindex(data['event_type'].unique(), fill_value=0)
    stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    return stat, p_value

# Streamlit App
st.title("Statistical Tests for Cyber Attack Data")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    # Perform the tests
    df['region_group'] = df['region'].apply(lambda x: 'Africa' if x == 'Africa' else 'Non-Africa')

    # Chi-Square Test between Africa and non-Africa regions
    chi2_stat, chi2_p = chi_square_test(df, 'region_group', 'event_type')

    # Mann-Whitney U Test: Africa vs other regions
    mann_whitney_africa_vs_asia = mann_whitney_test(df, 'Africa', 'Asia')
    mann_whitney_africa_vs_europe = mann_whitney_test(df, 'Africa', 'Europe')
    mann_whitney_africa_vs_na = mann_whitney_test(df, 'Africa', 'North America')

    # Cramér's V Test
    contingency_table = pd.crosstab(df['region'], df['event_type'])
    cramers_v_value = cramers_v(contingency_table)

    # Ensure Cramér's V is handled as a scalar value
    if isinstance(cramers_v_value, pd.Series):
        cramers_v_value = cramers_v_value.iloc[0]

    # Organize the results
    test_results = {
        "Test": ["Chi-Square Test", "Mann-Whitney U Test (Africa vs Asia)", "Mann-Whitney U Test (Africa vs Europe)",
                 "Mann-Whitney U Test (Africa vs North America)", "Cramér's V"],
        "Statistic": [chi2_stat, mann_whitney_africa_vs_asia[0], mann_whitney_africa_vs_europe[0],
                      mann_whitney_africa_vs_na[0], cramers_v_value],
        "P-Value": [chi2_p, mann_whitney_africa_vs_asia[1], mann_whitney_africa_vs_europe[1], mann_whitney_africa_vs_na[1], "-"],
        "Conclusion": [
            "No significant difference between Africa and non-Africa regions" if chi2_p > 0.05 else "Significant difference between Africa and non-Africa regions",
            "No significant difference between Africa and Asia" if mann_whitney_africa_vs_asia[1] > 0.05 else "Significant difference between Africa and Asia",
            "No significant difference between Africa and Europe" if mann_whitney_africa_vs_europe[1] > 0.05 else "Significant difference between Africa and Europe",
            "No significant difference between Africa and North America" if mann_whitney_africa_vs_na[1] > 0.05 else "Significant difference between Africa and North America",
            "Very weak association between event types and regions" if cramers_v_value < 0.1 else "Stronger association detected between event types and regions"
        ]
    }

    # Convert results into DataFrame
    results_df = pd.DataFrame(test_results)

    # Display results in the app
    st.write("Statistical Test Results:")
    st.dataframe(results_df)

    # Option to download the results as a CSV file
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Test Results CSV", data=csv, file_name="stat_test_results.csv", mime='text/csv')

else:
    st.write("Please upload a CSV file.")
