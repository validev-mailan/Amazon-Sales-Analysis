import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("amazon_sales_data 2025.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 2. Data Cleaning

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y")

# Fix text inconsistencies (strip spaces)
df["Product"] = df["Product"].str.strip()
df["Category"] = df["Category"].str.strip()
df["Customer Name"] = df["Customer Name"].str.strip()

# 3. Feature Engineering

# Recalculate Total Sales
df["Total Sales"] = df["Price"] * df["Quantity"]

# Create Month feature
df["Month"] = df["Date"].dt.month

# Create Year feature
df["Year"] = df["Date"].dt.year

print("\nCleaned Dataset:")
print(df.head())

# 4. Exploratory Data Analysis (EDA)

# Histogram (Price distribution)
plt.figure()
plt.hist(df["Price"], bins=10)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Bar Chart (Category wise sales)
category_sales = df.groupby("Category")["Total Sales"].sum()

plt.figure()
category_sales.plot(kind="bar")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()

# Boxplot (detect outliers)
plt.figure()
sns.boxplot(x=df["Total Sales"])
plt.title("Outliers in Total Sales")
plt.show()

# Scatter Plot (Price vs Quantity)
plt.figure()
plt.scatter(df["Price"], df["Quantity"])
plt.title("Price vs Quantity")
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()

# Correlation Heatmap
plt.figure()
corr = df[["Price","Quantity","Total Sales"]].corr()
sns.heatmap(corr, annot=True)
plt.title