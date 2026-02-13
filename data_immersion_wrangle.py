# ==========================================================
# TASK 1 - Data Immersion & Wrangling
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# 1. DATA ACCESS & FAMILIARIZATION
# ==========================================================

def load_data(file_path):
    print("\nLoading Dataset...")
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
    return df


def data_overview(df):
    print("========= DATA OVERVIEW =========")
    print("\nFirst 5 Rows:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())
    print("\nStatistical Summary:\n", df.describe(include='all'))
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Records:", df.duplicated().sum())


# ==========================================================
# 2. DATA QUALITY ASSESSMENT
# ==========================================================

def data_quality_check(df):
    report = []

    report.append("========= DATA QUALITY REPORT =========\n")

    report.append("\nMissing Values:\n")
    report.append(str(df.isnull().sum()))

    report.append("\n\nDuplicate Records:\n")
    report.append(str(df.duplicated().sum()))

    report.append("\n\nData Types:\n")
    report.append(str(df.dtypes))

    report.append("\n\nOutlier Check (Numerical Columns):\n")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        report.append(f"{col}: {len(outliers)} outliers")

    with open("profiling_report.txt", "w") as f:
        for line in report:
            f.write(line + "\n")

    print("\nData Quality Report Generated: profiling_report.txt")


# ==========================================================
# 3. DATA CLEANING & TRANSFORMATION
# ==========================================================

def clean_data(df):

    print("\nCleaning Data...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert dates
    if 'transactiondate' in df.columns:
        df['transactiondate'] = pd.to_datetime(df['transactiondate'], errors='coerce')

    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

    # Create Age Feature
    if 'date_of_birth' in df.columns:
        today = pd.Timestamp.today()
        df['customer_age'] = (today - df['date_of_birth']).dt.days // 365

    # Handle Missing Values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Remove negative values in Quantity or Price
    if 'quantity' in df.columns:
        df = df[df['quantity'] >= 0]

    if 'price' in df.columns:
        df = df[df['price'] >= 0]

    # Create Revenue Column
    if 'quantity' in df.columns and 'price' in df.columns:
        df['revenue'] = df['quantity'] * df['price']

    print("Data Cleaning Completed!\n")

    return df


# ==========================================================
# 4. DATA DICTIONARY CREATION
# ==========================================================

def create_data_dictionary(df):
    dictionary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.count(),
        "Description": "Add business description here"
    })

    dictionary.to_excel("data_dictionary.xlsx", index=False)
    print("Data Dictionary Generated: data_dictionary.xlsx")


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    file_path = "raw_data.csv"

    if not os.path.exists(file_path):
        print("ERROR: raw_data.csv not found in project folder.")
        return

    df = load_data(file_path)

    data_overview(df)

    data_quality_check(df)

    cleaned_df = clean_data(df)

    cleaned_df.to_csv("cleaned_data.csv", index=False)

    create_data_dictionary(cleaned_df)

    print("\nFinal Cleaned Dataset Saved as: cleaned_data.csv")
    print("Task Completed Successfully!")


if __name__ == "__main__":
    main()
