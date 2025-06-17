
# FTL_Ethiopia_ML2_Gr13
# Literacy Rate Predictor - FTL_g13

This repository contains a machine learning project to predict literacy rates in Ethiopia based on education and population data. The project includes:

- Data preprocessing and feature engineering  
- Model training with Linear Regression, Random Forest, and XGBoost  
- A Flask web app for interactive literacy rate prediction by region  
- Docker setup for containerized deployment  
- Deployment on Render.com for easy web access  

---

## 🚀 Live Demo

Access the deployed app here:  
[https://literacy-predictor.onrender.com/](https://literacy-predictor.onrender.com/)

---

## 📂 Repository Structure

```
FTL_g13/
├── app.py                    # Flask web app
├── model/                    # Saved model, encoder, and features files
├── plots/                    # Visualization images (correlation, feature importance)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container setup
├── templates/                # HTML template for Flask app
├── ethiopia_education_data_Primary.csv
├── ethiopia_education_data_Secondary.csv
├── eth_admpop_adm1_2022_v2.csv
└── README.md                 # This file
```

---

## ⚙️ Setup and Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/FTL_g13.git
cd FTL_g13
```

### 2. Install dependencies

Use a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Flask app

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000) and use the dropdown to select a region and predict literacy rate.

---

## 🐳 Run with Docker

Make sure Docker is installed and running.

### Build Docker image:

```bash
docker build -t literacy_flask .
```

### Run container locally:

```bash
docker run -p 5000:5000 literacy_flask
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## ☁️ Deployment on Render

This app is deployed on [Render.com](https://render.com) with the following settings:

- Environment: Python 3.10  
- Start Command: `python app.py`  
- Port: 5000 (Render automatically maps internal port 5000)  

**Note:** On the free Render plan, the app may sleep after inactivity, causing some delay on the first access.

---

## 🔧 How It Works

- Loads pre-trained Linear Regression model (`model/literacy_forecast_model.pkl`)  
- Uses encoded regions and precomputed average enrollment ratios for predictions  
- User selects a region from a dropdown menu on the web page  
- Predicts literacy rate based on the selected region and shows result  

---

## 👥 Group Members

- Dereje Kaba  
- Yohannes Alelign  
- Workineh Tilahun  
- Sadik Tofiq  

---

## 🏅 Program

This project is part of the **Frontier Tech Leaders Programme** by **UNDP**.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Ethiopian Ministry of Education data (Primary & Secondary enrollment)  
- HDX Population dataset for Ethiopia  
- Frontier Tech Leaders Programme  

---
