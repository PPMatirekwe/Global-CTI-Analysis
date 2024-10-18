import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kendalltau
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to plot pie charts with values outside
def plot_pie_chart(data, title):
    for region in data.index:
        region_data = data.loc[region]
        region_data = region_data[region_data > 0]  # Filter out zero values
        
        # Plot pie chart
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(region_data, autopct='%1.1f%%', startangle=140)

        # Adjust text to avoid overlap
        for i, a in enumerate(autotexts):
            a.set_text(f'{region_data.index[i]}: {a.get_text()}')
        
        # Set the title
        plt.title(f'{title} for {region}', fontsize=16)

        # Add the legend on the right side
        plt.legend(wedges, region_data.index, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Display the plot
        plt.tight_layout()
        st.pyplot(plt)

# Function to encode categorical variables for Kendall's Tau
def encode_categories(data, columns):
    le = LabelEncoder()
    for col in columns:
        data[col] = le.fit_transform(data[col].astype(str))
    return data

# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = np.sum((confusion_matrix.to_numpy() - confusion_matrix.to_numpy().mean()) ** 2 / confusion_matrix.to_numpy().mean())
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# Main Streamlit App
st.title("Cyber Attack Data Analysis and Visualization")

uploaded_file = '/mnt/data/Merged_Dataset.csv'

if uploaded_file is not None:
    st.write("File loaded successfully!")
    df = pd.read_csv(uploaded_file)

    # Ensure that 'evtDate' is in datetime format
    df['evtDate'] = pd.to_datetime(df['evtDate'], errors='coerce')

    # Event type analysis
    st.header("Event Type Analysis")
    top_events = df['event_type'].value_counts().nlargest(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_events.index, y=top_events.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top 10 Cyber Event Types")
    st.pyplot(fig)

    # Rolling sum of cyber attacks
    st.header("Rolling Sum of Cyber Attacks (365-day window)")
    df_resampled = df.set_index('evtDate').resample('D').size().rolling(365).sum().dropna()
    fig, ax = plt.subplots()
    df_resampled.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Attacks (365-day sum)')
    ax.set_title('365-Day Rolling Sum of Cyber Attacks')
    st.pyplot(fig)

    # Pie charts
    st.header("Pie Chart Analysis by Region")
    region_event_counts = df.groupby(['region', 'event_type']).size().unstack().fillna(0)
    plot_pie_chart(region_event_counts, 'Event Type Distribution')

    # Cramér’s V calculation for categorical data
    st.header("Cramér’s V Correlation Tests")
    categorical_columns = ['actor_type', 'motive', 'event_type', 'industry', 'region']
    cramers_v_results = {}

    for i, col1 in enumerate(categorical_columns):
        for col2 in categorical_columns[i+1:]:
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            cramers_v_value = cramers_v(confusion_matrix)
            cramers_v_results[(col1, col2)] = cramers_v_value

    cramers_v_df = pd.DataFrame(cramers_v_results.items(), columns=["Variable Pair", "Cramér's V"])
    st.dataframe(cramers_v_df)

    # Kendall's Tau correlation
    st.header("Kendall’s Tau Correlation Tests")
    encoded_data = encode_categories(df.copy(), categorical_columns)
    kendalls_tau_results = {}

    for i, col1 in enumerate(categorical_columns):
        for col2 in categorical_columns[i+1:]:
            tau, _ = kendalltau(encoded_data[col1], encoded_data[col2])
            kendalls_tau_results[(col1, col2)] = tau

    kendalls_tau_df = pd.DataFrame(kendalls_tau_results.items(), columns=["Variable Pair", "Kendall's Tau"])
    st.dataframe(kendalls_tau_df)

    # Chi-square test
    st.header("Chi-Square Test: Region vs. Event Type")
    chi_square_test_region_event_type = pd.crosstab(df['region'], df['event_type'])
    chi2, p, dof, expected = chi2_contingency(chi_square_test_region_event_type)
    st.write(f"Chi-Square Statistic: {chi2:.4f}")
    st.write(f"p-value: {p:.4f}")
    st.write(f"Degrees of Freedom: {dof}")

else:
    st.write("Please upload a CSV file.")
