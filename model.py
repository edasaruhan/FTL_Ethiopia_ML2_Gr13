import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns



os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load and prepare the data (repeating key preprocessing steps)
primary = pd.read_csv("ethiopia_education_data_Primary.csv")
secondary = pd.read_csv("ethiopia_education_data_Secondary.csv")
population = pd.read_csv("eth_admpop_adm1_2022_v2.csv")

def clean_numeric(df, columns):
    for col in columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

# Rename
primary = primary.rename(columns={'region': 'Region_name'})
secondary = secondary.rename(columns={'region': 'Region_name'})
population = population.rename(columns={'admin1Name_en': 'Region_name'})

primary = primary.rename(columns={'2022/23 Total_P': 'Primary_enrollment'})
secondary = secondary.rename(columns={'2022/23 Total_S': 'Secondary_enrollment'})


enrollment_df = pd.merge(primary, secondary, on=['Region_name'])

# Clean values
enrollment_df = clean_numeric(enrollment_df, ['Primary_enrollment', 'Secondary_enrollment', 'Number_of_P_Schools', 'Number_of_S_Schools'])

# Eligible population
population['P_eligible'] = population['T_05_09'] + population['T_10_14']
population['S_eligible'] = population['T_15_19']
population_elig = population[['Region_name', 'P_eligible', 'S_eligible']]

# Merge
full_df = pd.merge(enrollment_df, population_elig, on='Region_name', how='left')
full_df['Total_enrolled'] = full_df['Primary_enrollment'] + full_df['Secondary_enrollment']
full_df['Eligible_population'] = full_df['P_eligible'] + full_df['S_eligible']
full_df['Literacy_percent'] = (full_df['Total_enrolled'] / full_df['Eligible_population']) * 100

full_df = full_df[full_df['Total_enrolled'] > 0]

full_df['High_Literacy'] = (full_df['Literacy_percent'] >= 70).astype(int)
# Encode categorical and temporal features
# full_df['Year_num'] = full_df['Year'].str[:4].astype(int)

region_encoder = LabelEncoder()
full_df['Region_encoded'] = region_encoder.fit_transform(full_df['Region_name'])

full_df['P_ratio'] = full_df['Primary_enrollment'] / full_df['P_eligible']
full_df['S_ratio'] = full_df['Secondary_enrollment'] / full_df['S_eligible']
full_df['School_ratio'] = full_df['Number_of_P_Schools'] / (full_df['Number_of_S_Schools'] + 1e-5)  # Avoid division by 0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(full_df['Primary_enrollment'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Primary Enrollment Distribution")

sns.histplot(full_df['Secondary_enrollment'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Secondary Enrollment Distribution")

sns.histplot(full_df['P_ratio'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Primary Enrollment Rate Distribution")

sns.histplot(full_df['S_ratio'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Secondary Enrollment Rate Distribution")

plt.tight_layout()
plt.show()

#correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(full_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#box plot
plt.figure(figsize=(12, 5))
sns.boxplot(data=full_df[['P_ratio', 'S_ratio']])
plt.title("Boxplot of Enrollment Rates")
plt.show()


print(full_df['Literacy_percent'])
numeric_df = full_df.select_dtypes(include=[np.number])

# Define features and target
features = ['Region_encoded', 'P_ratio', 'S_ratio']
# features = ['Region_encoded', 'Primary_enrollment', 'Secondary_enrollment', 'Number_of_P_Schools', 'Number_of_S_Schools']
X = full_df[features]
y = full_df['Literacy_percent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing data
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(imputer.transform(X_test), columns=features)


# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Compare models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))  # ✅ Corrected
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        'Model': name,
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'R2 Score': round(r2, 2)
    })

# Show results
results_df = pd.DataFrame(results).sort_values(by='RMSE')
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))


# Train model
X = pd.DataFrame(imputer.fit_transform(X), columns=features)
model = LinearRegression()
model.fit(X, y)

# checking overfitting
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Results
print("R² scores on each fold:", scores)
print("Mean R²:", np.mean(scores))
print("Standard Deviation:", np.std(scores))

# feature importance
importances = model.coef_

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()
# plt.close()

# Save model and encoder
joblib.dump(model, "model/literacy_forecast_model.pkl")
joblib.dump(region_encoder, "model/region_encoder.pkl")
joblib.dump(features, "model/forecast_features.pkl")

# Evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"Model RMSE on full data: {rmse:.2f}")


# print(full_df)
