import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

# Manually defining latitude and longitude for a set of countries
country_coords = {
    'United States of America': (37.0902, -95.7129),
    'Germany': (51.1657, 10.4515),
    'Russian Federation': (61.5240, 105.3188),
    'United Kingdom of Great Britain and Northern Ireland': (55.3781, -3.4360),
    'France': (46.6034, 1.8883),
    'China': (35.8617, 104.1954),
    'India': (20.5937, 78.9629),
    'Brazil': (-14.2350, -51.9253),
    'Canada': (56.1304, -106.3468),
    'Australia': (-25.2744, 133.7751),
    'South Africa': (-30.5595, 22.9375),
    'Japan': (36.2048, 138.2529),
    'Italy': (41.8719, 12.5674),
    'Spain': (40.4637, -3.7492),
    'Mexico': (23.6345, -102.5528),
    'Netherlands': (52.1326, 5.2913),
    'Argentina': (-38.4161, -63.6167),
    'Switzerland': (46.8182, 8.2275),
    'Sweden': (60.1282, 18.6435),
    'South Korea': (35.9078, 127.7669),
    'Turkey': (38.9637, 35.2433),
    'Belgium': (50.8503, 4.3517),
    'Norway': (60.4720, 8.4689),
    'Denmark': (56.2639, 9.5018),
    'Finland': (61.9241, 25.7482),
    'Austria': (47.5162, 14.5501),
    'Israel': (31.0461, 34.8516),
    'Singapore': (1.3521, 103.8198),
    'New Zealand': (-40.9006, 174.8860),
    'Ireland': (53.1424, -7.6921),
    'Malaysia': (4.2105, 101.9758),
    'Philippines': (12.8797, 121.7740),
    'Saudi Arabia': (23.8859, 45.0792),
    'United Arab Emirates': (23.4241, 53.8478),
    'Nigeria': (9.0820, 8.6753),
    'Egypt': (26.8206, 30.8025),
    'Pakistan': (30.3753, 69.3451),
    'Indonesia': (-0.7893, 113.9213),
    'Vietnam': (14.0583, 108.2772),
    'Thailand': (15.8700, 100.9925),
    'Ukraine': (48.3794, 31.1656),
    'Poland': (51.9194, 19.1451),
    'Czech Republic': (49.8175, 15.4730),
    'Greece': (39.0742, 21.8243),
    'Portugal': (39.3999, -8.2245),
    'Hungary': (47.1625, 19.5033),
    'Chile': (-35.6751, -71.5430),
    'Romania': (45.9432, 24.9668),
    'Bangladesh': (23.6850, 90.3563),
    'Peru': (-9.1899, -75.0152),
    'Colombia': (4.5709, -74.2973),
    'Venezuela': (6.4238, -66.5897),
    'Kazakhstan': (48.0196, 66.9237),
    'Morocco': (31.7917, -7.0926),
    'Algeria': (28.0339, 1.6596),
    'Tunisia': (33.8869, 9.5375),
    'Kenya': (-1.2921, 36.8219),
    'Ethiopia': (9.1450, 40.4897),
    'Sudan': (12.8628, 30.2176),
    'Ghana': (7.9465, -1.0232),
    'Uganda': (1.3733, 32.2903),
    'Tanzania': (-6.3690, 34.8888),
    'Angola': (-11.2027, 17.8739),
    'Mozambique': (-18.6657, 35.5296),
    'Zimbabwe': (-19.0154, 29.1549),
    'Ivory Coast': (7.5399, -5.5471),
    'Cameroon': (3.8480, 11.5021),
    'Zambia': (-13.1339, 27.8493),
    'Senegal': (14.4974, -14.4524),
    'Malawi': (-13.2543, 34.3015),
    'Rwanda': (-1.9403, 29.8739),
    'Botswana': (-22.3285, 24.6849),
    'Namibia': (-22.9576, 18.4904),
    'Chad': (15.4542, 18.7322),
    'Mali': (17.5707, -3.9962),
    'Niger': (17.6078, 8.0817),
    'Guinea': (9.9456, -9.6966),
    'Benin': (9.3077, 2.3158),
    'Burkina Faso': (12.2383, -1.5616),
    'Togo': (8.6195, 0.8248),
    'Sierra Leone': (8.4606, -11.7799),
    'Liberia': (6.4281, -9.4295),
    'Central African Republic': (6.6111, 20.9394),
    'Congo': (-4.2634, 15.2429),
    'DR Congo': (-4.0383, 21.7587),
    'Gabon': (-0.8037, 11.6094),
    'Equatorial Guinea': (1.6508, 10.2679),
    'Madagascar': (-18.7669, 46.8691),
    'Mauritius': (-20.3484, 57.5522),
    'Seychelles': (-4.6796, 55.4920),
    'Somalia': (5.1521, 46.1996),
    'Libya': (26.3351, 17.2283),
    'Iraq': (33.2232, 43.6793),
    'Syria': (34.8021, 38.9968),
    'Jordan': (30.5852, 36.2384),
    'Lebanon': (33.8547, 35.8623),
    'Kuwait': (29.3117, 47.4818),
    'Qatar': (25.3548, 51.1839),
    'Bahrain': (26.0667, 50.5577),
    'Oman': (21.5126, 55.9233),
    'Yemen': (15.5527, 48.5164),
    'Afghanistan': (33.9391, 67.7100),
    'Uzbekistan': (41.3775, 64.5853),
    'Turkmenistan': (38.9697, 59.5563),
    'Tajikistan': (38.8610, 71.2761),
    'Kyrgyzstan': (41.2044, 74.7661),
    'Armenia': (40.0691, 45.0382),
    'Azerbaijan': (40.1431, 47.5769),
    'Georgia': (42.3154, 43.3569),
    'Belarus': (53.7098, 27.9534),
    'Lithuania': (55.1694, 23.8813),
    'Latvia': (56.8796, 24.6032),
    'Estonia': (58.5953, 25.0136),
    'Slovakia': (48.6690, 19.6990),
    'Slovenia': (46.1512, 14.9955),
    'Croatia': (45.1000, 15.2000),
}


# Function to plot pie charts with values outside
def plot_pie_chart(data, title):
    region_data = data[data > 0]  # Filter out zero values
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(region_data, autopct='%1.1f%%', startangle=140)

    # Adjust text to avoid overlap
    for i, a in enumerate(autotexts):
        a.set_text(f'{region_data.index[i]}: {a.get_text()}')

    plt.title(f'{title}', fontsize=16)
    plt.legend(wedges, region_data.index, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(plt)

# Function to plot rolling windows for various intervals
def plot_rolling_windows(df, date_col):
    intervals = ['D', 'W', 'M', 'Y']  # Day, Week, Month, Year
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Ensure date is in correct format
    df.set_index(date_col, inplace=True)

    fig, axs = plt.subplots(len(intervals), 1, figsize=(10, 8))
    fig.tight_layout(pad=3.0)

    for i, interval in enumerate(intervals):
        window = {'D': 365, 'W': 52, 'M': 12, 'Y': 1}[interval]
        resampled_data = df.resample(interval).size().rolling(window=window).sum().dropna()
        axs[i].plot(resampled_data)
        axs[i].set_title(f'{interval}-Based Rolling Sum of Cyber Attacks')
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('Attack Count')

    st.pyplot(fig)

# Function to plot the overall dataset analysis
def plot_event_distributions(df):
    df['evtDate'] = pd.to_datetime(df['evtDate'], errors='coerce')
    df = df.dropna(subset=['evtDate'])  # Drop rows with missing 'evtDate'
    df.set_index('evtDate', inplace=True)

    df_hours = df.groupby(df.index.hour).size()
    df_week = df.groupby(df.index.day_name()).size()
    df_month = df.groupby(df.index.month).size()
    df_year = df.groupby(df.index.year).size()
    df_region = df.groupby('region').size()

    df_month = df_month.reset_index()
    df_month['evtDate'] = df_month['evtDate'].apply(lambda x: calendar.month_abbr[int(x)] if not pd.isnull(x) else '')

    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    sns.barplot(x=df_hours.index, y=df_hours.values, palette="OrRd_r", ax=axs[0, 0])
    axs[0, 0].set_title('Events by Hour')

    sns.barplot(x=df_week.index, y=df_week.values, palette="OrRd_r", ax=axs[0, 1])
    axs[0, 1].set_title('Events by Day of the Week')

    sns.barplot(data=df_month, x='evtDate', y=0, palette="OrRd_r", ax=axs[1, 0])
    axs[1, 0].set_title('Events by Month')

    sns.barplot(x=df_year.index, y=df_year.values, palette="OrRd_r", ax=axs[1, 1])
    axs[1, 1].set_title('Events by Year')

    sns.barplot(x=df_region.index, y=df_region.values, palette="OrRd_r", ax=axs[2, 0])
    axs[2, 0].set_title('Events by Region')

    axs[2, 1].axis('off')
    fig.suptitle('Cyber Events Distribution Analysis')
    st.pyplot(fig)

# Function to map geographical distribution of cyber events
def plot_geographical_distribution(df):
    if 'country' in df.columns:
        df['latitude'] = df['country'].map(lambda x: country_coords.get(x, (None, None))[0])
        df['longitude'] = df['country'].map(lambda x: country_coords.get(x, (None, None))[1])
        df_clean = df.dropna(subset=['latitude', 'longitude'])

        if not df_clean.empty:
            gdf = gpd.GeoDataFrame(df_clean, geometry=gpd.points_from_xy(df_clean.longitude, df_clean.latitude))
            gdf.set_crs(epsg=4326, inplace=True)

            fig = px.scatter_geo(gdf,
                                 lat=gdf.geometry.y,
                                 lon=gdf.geometry.x,
                                 hover_name="country",
                                 hover_data={"event_type": True, "industry": True},
                                 color="event_type",
                                 title="Geographical Distribution of Cyber Events")

            fig.update_layout(geo=dict(showland=True, landcolor="lightgray"))
            st.plotly_chart(fig)
        else:
            st.write("No valid geographical data available for mapping.")
    else:
        st.write("'country' column not found. Skipping geographical mapping.")

# Main Streamlit App
st.title("Cyber Attack Data Analysis and Visualization")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    if 'evtDate' in df.columns:
        df['evtDate'] = pd.to_datetime(df['evtDate'], errors='coerce')
        st.write("Date column exists and has been processed.")
    else:
        st.write("No 'evtDate' column found.")

    st.header("Global Event Distribution Analysis ")
    plot_event_distributions(df)

    if 'region' in df.columns:
        unique_regions = df['region'].unique()
        selected_region = st.selectbox("Select a region to explore", unique_regions)

        region_data = df[df['region'] == selected_region]

        st.header(f"Event Type Analysis for {selected_region}")
        if 'event_type' in region_data.columns:
            top_events = region_data['event_type'].value_counts().nlargest(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_events.index, y=top_events.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(f"Top 10 Cyber Event Types in {selected_region}")
            st.pyplot(fig)

            st.header(f"Rolling Window Analysis in {selected_region}")
            plot_rolling_windows(region_data, 'evtDate')

            st.header(f"Pie Chart for {selected_region}")
            region_event_counts = region_data['event_type'].value_counts()
            plot_pie_chart(region_event_counts, f'Event Type Distribution in {selected_region}')
        else:
            st.write(f"'event_type' column missing in the dataset for {selected_region}.")
    else:
        st.write("No 'region' column found for filtering.")

    st.header("Geographical Distribution of Cyber Events")
    plot_geographical_distribution(df)

else:
    st.write("Please upload a CSV file.")
