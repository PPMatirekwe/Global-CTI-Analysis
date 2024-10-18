import streamlit as st
import pandas as pd
import requests
import pycountry_convert as pc

# Title of the App
st.title("CISSM and Honeypot Data Merger with IP-to-Country and Country-to-Region Mapping")

# File upload for CISSM dataset
uploaded_cissm = st.file_uploader("Upload CISSM Dataset", type=["csv"])

# File upload for Honeypot IOC dataset
uploaded_ioc = st.file_uploader("Upload Honeypot IOC Dataset", type=["csv"])

# Event Type Mapping (Manually Define Common Honeypot Attack Types)
attack_type_mapping = {
    'SSH Brute Force': 'Exploitive',
    'Telnet Brute Force': 'Exploitive',
    'RDP Brute Force': 'Disruptive',
    'HTTP Flooding': 'Disruptive',
    'SQL Injection': 'Exploitive',
    'ICMP Flooding': 'Disruptive'
}

# Industry Input Options
industries = ['Finance', 'Healthcare', 'Technology', 'Retail', 'Government', 'Education']
selected_industry = st.selectbox("Select the Industry", industries)

# Function to map countries to regions
def country_to_region(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
        continent_name = pc.country_alpha2_to_continent_code(country_code)
        continent_map = {
            'AF': 'Africa',
            'NA': 'North America',
            'SA': 'South America',
            'AS': 'Asia',
            'EU': 'Europe',
            'OC': 'Oceania'
        }
        return continent_map.get(continent_name, 'Other')
    except:
        return 'Unknown'

# Function to map IP to country using an API
def ip_to_country(ip_address):
    try:
        response = requests.get(f"https://ipapi.co/{ip_address}/country_name/")
        if response.status_code == 200:
            return response.text
        else:
            return 'Unknown'
    except:
        return 'Unknown'

# Function to clean and preprocess datasets
def preprocess_datasets(df_cissm, df_ioc):
    # Keep only necessary columns from CISSM dataset
    cissm_columns_to_keep = ['evtDate', 'motive', 'event_type', 'country']
    df_cissm = df_cissm[cissm_columns_to_keep]

    # Add 'industry' and 'event_type' columns to Honeypot dataset
    df_ioc['industry'] = selected_industry

    # Map indicator type to event types based on attack_type_mapping
    df_ioc['event_type'] = df_ioc['Indicator type'].map(attack_type_mapping)

    # Perform IP to country mapping for the IOC dataset
    df_ioc['country'] = df_ioc['Indicator'].apply(ip_to_country)

    # Map countries to regions
    df_ioc['region'] = df_ioc['country'].apply(country_to_region)
    df_cissm['region'] = df_cissm['country'].apply(country_to_region)

    # Drop unnecessary columns from IOC dataset after mapping
    df_ioc = df_ioc.drop(columns=['Indicator', 'Indicator type', 'Description'])

    # Merge the two datasets on relevant columns
    df_combined = pd.concat([df_cissm, df_ioc], ignore_index=True)

    return df_combined

# Process the datasets after upload
if uploaded_cissm and uploaded_ioc:
    df_cissm = pd.read_csv(uploaded_cissm)
    df_ioc = pd.read_csv(uploaded_ioc)

    # Preprocess and merge datasets
    df_combined = preprocess_datasets(df_cissm, df_ioc)

    st.write("Merged Dataset Preview:")
    st.dataframe(df_combined.head())

    # Option to download the merged dataset as CSV
    csv_combined = df_combined.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Merged Dataset CSV",
        data=csv_combined,
        file_name='merged_cissm_honeypot.csv',
        mime='text/csv'
    )
else:
    st.write("Please upload both the CISSM and Honeypot IOC datasets to proceed.")
