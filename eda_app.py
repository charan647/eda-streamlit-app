import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="EDA App", layout="wide")

st.title("📊 EDA & Data Preprocessing App")

# ===============================
# Upload
# ===============================
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    # ===============================
    # OVERVIEW
    # ===============================
    st.header("📌 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # ===============================
    # MISSING VALUES
    # ===============================
    st.header("❗ Missing Values")
    st.dataframe(df.isnull().sum())

    # ===============================
    # PREPROCESSING
    # ===============================
    st.header("⚙️ Preprocessing")

    # Drop Columns
    drop_cols = st.multiselect("Drop Columns", df.columns)
    if st.button("Apply Drop"):
        st.session_state.df = df.drop(columns=drop_cols)
        st.success("Columns dropped")
        st.rerun()

    # Fill Missing
    if st.button("Fill Missing Values"):
        df = st.session_state.df

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        st.session_state.df = df
        st.success("Missing values handled")
        st.rerun()

    # Encoding
    if st.button("Encode Categorical Columns"):
        df = st.session_state.df
        le = LabelEncoder()

        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col].astype(str))

        st.session_state.df = df
        st.success("Encoding done")
        st.rerun()

    # ===============================
    # ANALYSIS
    # ===============================
    st.header("📊 Analysis")

    analysis_type = st.radio("Choose Analysis Type", ["Univariate", "Bivariate"])

    # -------- UNIVARIATE --------
    if analysis_type == "Univariate":
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        col = st.selectbox("Select Column", numeric_cols)

        chart = st.selectbox("Chart Type", ["Histogram", "Boxplot"])

        fig, ax = plt.subplots()

        if chart == "Histogram":
            sns.histplot(df[col], kde=True, ax=ax)
        else:
            sns.boxplot(y=df[col], ax=ax)

        st.pyplot(fig)

    # -------- BIVARIATE --------
    elif analysis_type == "Bivariate":

        if "Survived" in df.columns:
            option = st.selectbox(
                "Select Relationship",
                ["Pclass vs Survived", "Sex vs Survived", "Age vs Survived", "Fare vs Survived"]
            )

            fig, ax = plt.subplots()

            if option == "Pclass vs Survived":
                sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)

            elif option == "Sex vs Survived":
                sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)

            elif option == "Age vs Survived":
                sns.boxplot(x="Survived", y="Age", data=df, ax=ax)

            elif option == "Fare vs Survived":
                sns.boxplot(x="Survived", y="Fare", data=df, ax=ax)

            st.pyplot(fig)
        else:
            st.warning("Target column 'Survived' not found")

    # ===============================
    # HEATMAP
    # ===============================
    st.header("🔥 Correlation Heatmap")

    num_df = df.select_dtypes(include=["int64", "float64"])

    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ===============================
    # REPORT
    # ===============================
    st.header("📄 Auto Report")

    if st.button("Generate Report"):
        profile = ProfileReport(df)
        html(profile.to_html(), height=800)

    # ===============================
    # DOWNLOAD
    # ===============================
    st.header("⬇️ Download Cleaned Data")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")