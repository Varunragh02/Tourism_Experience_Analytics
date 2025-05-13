import pandas as pd
import numpy as np

# ========== Load Excel Files ==========

user_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/User.xlsx")
transaction_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Transaction.xlsx")
mode_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Mode.xlsx")
item_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Item.xlsx")
updated_item_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Updated_Item.xlsx")
type_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Type.xlsx")
country_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Country.xlsx")
region_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Region.xlsx")
continent_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/Continent.xlsx")
city_df = pd.read_excel("D:/Guvi/Tourism_Experience_Analytics/Excel/City.xlsx")

# ========== Combine Item Data ==========
total_item_df = pd.concat([item_df, updated_item_df], ignore_index=True)

# ========== Helper Cleaning Functions ==========

def clean_missing(df, important_cols=None):
    df.dropna(how='all', inplace=True)
    if important_cols:
        df.dropna(subset=important_cols, inplace=True)
    df.fillna("Unknown", inplace=True)
    return df

def standardize_categorical(df, col_name, mapping_dict=None):
    df[col_name] = df[col_name].astype(str).str.strip().str.title()
    if mapping_dict:
        df[col_name] = df[col_name].replace(mapping_dict)
    return df

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def remove_duplicates(df, subset_cols=None):
    df.drop_duplicates(subset=subset_cols, inplace=True)
    return df

# ========== Clean Datasets ==========

transaction_df = clean_missing(transaction_df, important_cols=['UserId', 'AttractionId'])
transaction_df = standardize_categorical(transaction_df, 'VisitMode')
transaction_df = remove_outliers(transaction_df, 'Rating')
transaction_df = remove_duplicates(transaction_df)

user_df = clean_missing(user_df, important_cols=['UserId', 'CityId'])
user_df = remove_duplicates(user_df, subset_cols=['UserId'])

city_df = clean_missing(city_df, important_cols=['CityId', 'CityName'])
city_df = standardize_categorical(city_df, 'CityName')
city_df = remove_duplicates(city_df, subset_cols=['CityId'])

# ========== Merge All Cleaned DataFrames ==========


# Merge with User to get location IDs
merged_df = transaction_df.merge(user_df, on='UserId', how='left')

# Merge with city to get CityName
merged_df = merged_df.merge(city_df[['CityId', 'CityName']], on='CityId', how='left')

# Merge with item and type
merged_df = merged_df.merge(total_item_df, on='AttractionId', how='left')
merged_df = merged_df.merge(type_df[['AttractionTypeId', 'AttractionType']], on='AttractionTypeId', how='left')

# Merge to get location names from IDs
merged_df = merged_df.merge(country_df[['CountryId', 'Country']], on='CountryId', how='left')
merged_df = merged_df.merge(region_df[['RegionId', 'Region']], on='RegionId', how='left')
merged_df = merged_df.merge(continent_df[['ContinentId', 'Continent']], on='ContinentId', how='left')

# Merge visit mode
mode_df = standardize_categorical(mode_df, 'VisitModeId')
merged_df = merged_df.merge(mode_df, left_on='VisitMode', right_on='VisitModeId', how='left')

# Remove rows with placeholder region
merged_df = merged_df[merged_df['Region'] != '-']

# ========== Final Touches ==========

# Drop redundant ID columns if needed
columns_to_drop = ['CountryId', 'RegionId', 'ContinentId']
merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], inplace=True)
# Remove duplicates
merged_df=merged_df.rename(columns={'VisitMode_y': 'VisitMode'})

# Export
merged_df.to_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv", index=False)
print("✅ Merged cleaned data exported successfully.")

# Quick EDA
print("Merged DataFrame Shape:", merged_df.shape)
print(merged_df.dtypes)
print(merged_df.columns)

