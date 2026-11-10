import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Handle missing values

# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop(columns=["Cabin"], inplace=True)

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Exploratory Data Analysis (EDA)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Survival count
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# Gender-wise passenger count
print("\nGender-wise Passenger Count:")
print(df["Sex"].value_counts())

# Survival Count Visualization

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.savefig("survival_count.png")
plt.show()

# Gender-wise Survival Analysis

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Sex", hue="Survived")

plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.savefig("gender_survival.png")
plt.show()

# Survival Analysis by Passenger Class
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.savefig("class_survival.png")
plt.show()

# Survival Count
survival_count = df['Survived'].value_counts()

print("\nSurvival Count:")
print(survival_count)

# Age Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Age'].dropna(), bins=20)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Survival Rate by Gender

gender_survival = df.groupby('Sex')['Survived'].mean() * 100

print("\nSurvival Rate by Gender:")
print(gender_survival)

gender_survival.plot(kind='bar', figsize=(7, 5))
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate (%)')
plt.show()

# Survival Rate by Passenger Class

class_survival = df.groupby('Pclass')['Survived'].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(class_survival)

class_survival.plot(kind='bar', figsize=(7, 5))
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.show()

# Survival Rate by Embarked

embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100

print("\nSurvival Rate by Embarked:")
print(embarked_survival)

embarked_survival.plot(kind='bar', figsize=(7, 5))
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked Port')
plt.ylabel('Survival Rate (%)')
plt.show()

# Missing Values Analysis

missing_values = df.isnull().sum()

print("\nMissing Values in Dataset:")
print(missing_values)

# Handling Missing Values

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column because it has many missing values
df = df.drop(columns=['Cabin'], errors='ignore')

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Final Findings

print("\n========== FINAL FINDINGS ==========")

print("1. Total passengers: 891")

print("2. Total survived passengers: 342")

print("3. Total non-survived passengers: 549")

print("4. Female passengers had a higher survival rate.")

print("5. First class passengers had a higher survival rate.")

print("6. Cherbourg (C) had the highest survival rate.")

print("7. No missing values were found in the available dataset columns.")