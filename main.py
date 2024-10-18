import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("simple dashboard ")

uploaded_file = st.file_uploader("choose a fiel",type="csv")

if uploaded_file is not None:
    st.write("file uploaded...")

    df = pd.read_csv(uploaded_file)

    st.subheader("prev")
    st.write(df.head())

    st.subheader("descr")
    st.write(df.describe())

    st.subheader("filter")

    columns = df.columns.tolist()

    selected_column = st.selectbox("slect column to filter by", columns)
    filtered_df = df[selected_column]
    unique_values = df[selected_column].unique()

    x_column = st.selectbox("slect x", columns)
    y_column = st.selectbox("slect y", columns)

    if st.button("Generate plot"):
        st.line_chart(filtered_df.set_index(x_column)(y_column))
    else:
        st.write("no upload")