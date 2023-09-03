## California House Price Predictor

Predict the median house value for districts in California using various features like median income, housing median age, average rooms, etc.

### Overview

This project uses the California housing dataset, which contains data drawn from the 1990 U.S. Census. The target variable is the median house value for California districts. The project employs two regression models - Linear Regression and Random Forest Regressor - to predict the median house value based on input features. The project also includes a Flask-based web application for real-time predictions.

### Features

- Data exploration and visualization using `pandas`, `matplotlib`, and `seaborn`.
- Data preprocessing and standardization.
- Model training and evaluation using `sklearn`.
- Flask web application for real-time predictions.
  
### Directory Structure

```
California_House_Price_Predictor/
│
├── app.py                          # Flask application script
│
├── train_and_save_model.py         # Script to train, scale, and save model/scaler
│
├── model_evaluation.py             # Script for in-depth model evaluation and visualization
│
├── rf_model.pkl                    # Saved Random Forest model
│
├── scaler.pkl                      # Saved scaler
│
└── templates/                      
    └── index.html                  # HTML template for Flask app
```

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/muhalwan/House_Price_Predictor.git
   cd California_House_Price_Predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt  # You'll need to create this file with all your dependencies
   ```

3. Run the model training and saving script:
   ```bash
   python train_and_save_model.py
   ```

4. Start the Flask application:
   ```bash
   python app.py
   ```

5. Visit `http://127.0.0.1:5000/` in a web browser to interact with the application.

### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
