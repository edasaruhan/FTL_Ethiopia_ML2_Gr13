from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model, encoder, and features
model = joblib.load("model/literacy_forecast_model.pkl")
region_encoder = joblib.load("model/region_encoder.pkl")
features = joblib.load("model/forecast_features.pkl")

# Load sample data for computing average ratios (used for prediction)
primary = pd.read_csv("ethiopia_education_data_Primary.csv")
secondary = pd.read_csv("ethiopia_education_data_Secondary.csv")
population = pd.read_csv("eth_admpop_adm1_2022_v2.csv")

# Preprocess data to compute average ratios for fallback
primary = primary.rename(columns={'region': 'Region_name', '2022/23 Total_P': 'Primary_enrollment'})
secondary = secondary.rename(columns={'region': 'Region_name', '2022/23 Total_S': 'Secondary_enrollment'})
population = population.rename(columns={'admin1Name_en': 'Region_name'})

enrollment_df = pd.merge(primary, secondary, on='Region_name')
enrollment_df['Primary_enrollment'] = enrollment_df['Primary_enrollment'].astype(str).str.replace(',', '').astype(float)
enrollment_df['Secondary_enrollment'] = enrollment_df['Secondary_enrollment'].astype(str).str.replace(',', '').astype(float)

population['P_eligible'] = population['T_05_09'] + population['T_10_14']
population['S_eligible'] = population['T_15_19']
population_elig = population[['Region_name', 'P_eligible', 'S_eligible']]

merged_df = pd.merge(enrollment_df, population_elig, on='Region_name', how='left')
merged_df['P_ratio'] = merged_df['Primary_enrollment'] / merged_df['P_eligible']
merged_df['S_ratio'] = merged_df['Secondary_enrollment'] / merged_df['S_eligible']

# Region list
region_list = sorted(merged_df['Region_name'].unique())

@app.route('/')
def index():
    return render_template('index.html', regions=region_list)

@app.route('/predict', methods=['POST'])
def predict():
    region = request.form['region']
    try:
        region_encoded = region_encoder.transform([region])[0]

        row = merged_df[merged_df['Region_name'] == region].iloc[0]
        p_ratio = row['P_ratio']
        s_ratio = row['S_ratio']
    except:
        region_encoded = 0
        p_ratio = merged_df['P_ratio'].mean()
        s_ratio = merged_df['S_ratio'].mean()

    input_data = pd.DataFrame([[region_encoded, p_ratio, s_ratio]], columns=features)
    prediction = model.predict(input_data)[0]

    return render_template('index.html', regions=region_list, prediction=round(prediction, 2), selected_region=region)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

