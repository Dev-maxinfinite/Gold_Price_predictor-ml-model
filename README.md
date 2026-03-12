# Gold Price Prediction App 🪙💰

## 📖 / Introduction
यह एक **Gold Price Prediction Web Application** है जो **machine learning model** का उपयोग करके सोने (Gold) की कीमत का पूर्वानुमान लगाती है।  
This is a Flask-based web app that predicts gold prices using a pre-trained ML model. Users can input data via forms and view predictions on a dashboard.

**Demo:** Run locally at `http://127.0.0.1:5000`

## ✨ Features
- 🏠 Responsive homepage with about section
- 📊 Input form for prediction parameters
- 📈 Dashboard to view predictions and historical data
- ⚡ Fast predictions using pickled scikit-learn model
- 📱 Mobile-friendly UI with Bootstrap

## 🛠️ Tech Stack
- **Backend:** Python 3, Flask
- **ML:** scikit-learn, pandas, joblib (gld_price_model.pkl)
- **Frontend:** HTML, CSS (Bootstrap), JS (Charts.js?)
- **Data:** JSON files for data/predictions

## 🚀 Quick Start / Installation
1. **Clone/Navigate:** Already in `/home/as/Desktop/gold_prdiction`
2. **Virtual Environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```
3. **Install Dependencies:**
   ```
   pip install flask pandas scikit-learn joblib numpy
   ```
4. **Run the App:**
   ```
   python app.py
   ```
5. **Open:** http://127.0.0.1:5000

**Note:** Model `gld_price_model.pkl` and data ready to use.

## 📁 Project Structure
```
gold_prdiction/
├── app.py                 # Main Flask app
├── data.json              # Historical data
├── gld_price_model.pkl    # Trained ML model
├── predictions.json       # Prediction outputs
├── templates/             # HTML pages
│   ├── base.html
│   ├── index.html
│   ├── input.html
│   ├── dashboard.html
│   └── about.html
├── static/                # CSS/JS/images
├── data/
└── models/
```

## 🤖 Model Details
- **Model:** Pre-trained regressor (likely RandomForest/XGBoost) on gold price features.
- **Input:** Features like SPX, oil price, USD index etc. (check input.html form).
- **Output:** Predicted GLD price.
- Retrain if needed: Use `data.json` for training.

## 📸 Screenshots
(Add your own or use:)
- Home: ![Home](static/screenshots/home.png)
- Prediction: ![Predict](static/screenshots/predict.png)

## 🔮 Future Improvements
- Live API integration for real-time data
- Model retraining endpoint
- User auth for saved predictions
- Deployment to Heroku/Vercel

## 🤝 Contributing
1. Fork the repo
2. Create branch `feature/xyz`
3. PR to `main`

## 📄 License
MIT License - © 2024 Gold Prediction Team

**Happy Predicting! 🚀**
