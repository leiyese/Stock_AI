# 📈 Stock_AI: AI-Powered Stock Analysis and Prediction

## 🔍 Overview
**StockAI** is a Python-based application that leverages machine learning and data analysis to predict stock price movements and provide actionable insights for investors. The project uses historical stock data and ML models to forecast future price trends.

OBS: Currently only LSTM ML models is implemented only using stock data. 
Future implementations: technical indicators and cutting-edge AI models

---

## 🚀 Features
- 📊 **Data Visualization**: Generate charts for stock prices.
- 🤖 **AI Predictions**: Predict stock price trends using machine learning models (e.g., LSTM, ).
- 📈 **Backtesting**: Evaluate trading strategies using historical data.
- 💾 **Data Sources**: Integrates with APIs like Alpha Vantage to fetch live data.
- ⚙️ **Customizable**: Easily adjust parameters like the stock ticker, prediction timeframe, and model settings.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/StockAI.git
cd Stock_AI
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Set up API Keys
•	Create a .env file in the project root and add your API key:
```plain text
ALPHA_VANTAGE_API=your_api_key
```

### 5. Run the application
```bash
python3 run.py
```

## 📄 Usage !! NEED UPDATE

### 1. Configure Settings and create .env file

•	Open config.json to set your preferred stock ticker, prediction model, and other parameters.
•	ALPHA_VANTAGE_API =
    MODEL_DIR =
    MODEL_PATH = 

### 2. Run Predictions

•	Start the application and view predictions in the terminal or a generated report:
```bash
python3 run.py --ticker AAPL --model LSTM
```

### 3. View Charts

•	Navigate to the output/ directory to view visualizations and results.

## 🧰 Technologies Used

•	**Programming Language:** Python
•	**Libraries:**
    •	Machine Learning: TensorFlow, Scikit-learn
    •	Data Handling: Pandas, NumPy
    •   Visualization: Matplotlib, Seaborn
•	**APIs:**
    •	Alpha Vantage

## 📊 Example Results

xoxoxoxo

## 📝 To-Do

    •	Add support for more APIs (e.g., IEX Cloud).
    •	Implement real-time stock prediction.
    •	Improve model accuracy with feature engineering.


## 🛡️ License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

1.	Fork the repository.

2.	Create a feature branch:

```bash
git checkout -b feature-name
```

3.	Commit your changes:
```bash
git commit -m "Add feature-name"
```

4. Push the branch:
```bash
git push origin feature-name
```

5. Open a pull request

## 📧 Contact

For questions, feel free to reach out:

•	Email: ly.leiye@gmail.com
•	GitHub: leiyese
