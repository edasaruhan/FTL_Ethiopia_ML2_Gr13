import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
primary = pd.read_csv("ethiopia_education_data_Primary.csv")
secondary = pd.read_csv("ethiopia_education_data_Secondary.csv")
population = pd.read_csv("eth_admpop_adm1_2022_v2.csv")

# 2. Rename region columns for consistency
primary = primary.rename(columns={'region': 'Region_name'})
secondary = secondary.rename(columns={'region': 'Region_name'})
population = population.rename(columns={'admin1Name_en': 'Region_name'})

# 3. Extract relevant columns
primary['Total_primary'] = primary['2022/23 Total']
secondary['Total_secondary'] = secondary['2022/23 Total']

# 4. Estimate eligible population
population['P_eligible'] = population['T_07_09'] + population['T_10_14'] if 'T_07_09' in population.columns else population['T_05_09']
population['S_eligible'] = population['T_15_19']

# 5. Create simplified population dataframe
population_data = population[['Region_name', 'P_eligible', 'S_eligible']]

# 6. Merge datasets
merged = pd.merge(primary[['Region_name', 'Total_primary']], 
                  secondary[['Region_name', 'Total_secondary']], 
                  on='Region_name')
merged = pd.merge(merged, population_data, on='Region_name')

# 7. Feature engineering
merged['Total_enrolled'] = merged['Total_primary'] + merged['Total_secondary']
merged['Eligible_population'] = merged['P_eligible'] + merged['S_eligible']
merged['Literacy_percent'] = (merged['Total_enrolled'] / merged['Eligible_population']) * 100

# 8. Clean data
final_df = merged.dropna()

# 9. Correlation heatmap
plt.figure(figsize=(10, 6))
corr_matrix = final_df[['Total_primary', 'Total_secondary', 'Total_enrolled', 
                        'P_eligible', 'S_eligible', 'Eligible_population', 
                        'Literacy_percent']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

# 10. Model training
X = final_df[['Total_primary', 'Total_secondary', 'Total_enrolled', 
              'P_eligible', 'S_eligible', 'Eligible_population']]
y = final_df['Literacy_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 12. Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 13. Feature importance
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# 14. Display feature importance table
print(feat_imp_df)
