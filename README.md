# 🏠 House Price Prediction

A machine learning web application that predicts house prices using the Ames Housing dataset.

The project covers the complete machine learning workflow:

- Data cleaning
- Feature engineering
- Data preprocessing
- Model training
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Model saving
- Streamlit deployment

---

## 📌 Project Overview

The goal of this project is to build a regression model capable of predicting house prices based on different characteristics of a house.

The target variable is `SalePrice`.

Because house prices are right-skewed, the target was transformed using:

```python
np.log1p(SalePrice)
````

The model therefore predicts `LogSalePrice`, which is converted back to the original price using:

```python
np.expm1(prediction)
```

---

## 📊 Dataset

The project uses the **Ames Housing Dataset**.

The dataset contains information about houses such as:

* Overall quality
* Living area
* Basement area
* Garage capacity
* Garage area
* Neighborhood
* Kitchen quality
* Exterior quality
* Year built
* Bathrooms
* And many other features

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

### Missing Values

Categorical missing values that represent the absence of a feature are replaced with:

```text
None
```

Numerical missing values are handled using appropriate imputation strategies.

`LotFrontage` is filled using the median value of the corresponding neighborhood.

---

## ⚙️ Feature Engineering

Additional features were created to improve the model.

Examples include:

* `HouseAge`
* `TotalSF`
* `TotalPorchSF`
* `YearsSinceRemodel`
* `IsRemodeled`
* `HasGarage`
* `HasFireplace`
* `HasPool`
* `HasBasement`
* `TotalBathrooms`
* `GarageSpacePerCar`

---

## 🤖 Machine Learning Models

Several regression algorithms were trained and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet
* Random Forest
* XGBoost

The models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## 🔄 Cross-Validation

After comparing the models, the best-performing model was evaluated using 5-fold cross-validation.

This helps determine whether the model performs consistently across different subsets of the training data.

---

## 🚀 XGBoost Hyperparameter Tuning

XGBoost was further optimized using `GridSearchCV`.

The following parameters were tested:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`

The combination with the best cross-validation R² score was selected as the final model.

---

## 💾 Model Saving

The final trained model is saved using Python's built-in `pickle` module:

```python
with open("models/house_price_model.pkl", "wb") as file:
    pickle.dump(tuned_xgb, file)
```

The model contains the preprocessing pipeline and trained XGBoost model.

---

## 📈 Model Performance

The application displays:

* Training R²
* Testing R²
* Training MAE
* Testing MAE
* Training RMSE
* Testing RMSE
* Cross-validation R²
* Best XGBoost parameters

---

## 🌐 Streamlit Application

The application contains several pages:

### 🏠 Home

Provides an overview of the project and dataset.

### 🔮 Predict Price

Users can enter house characteristics such as:

* Overall quality
* Living area
* Year built
* Basement area
* Garage capacity
* Garage area
* Bathrooms
* Neighborhood
* Kitchen quality
* Exterior quality

The application then predicts the estimated house price.

### 📊 Model Performance

Displays the model's evaluation metrics and XGBoost parameters.

### 📈 Explore Data

Allows users to explore the dataset, statistics, and missing values.

### ℹ️ About Project

Provides information about the project and machine learning techniques used.

---

## 📁 Project Structure

```text
house-price-prediction/
│
├── app.py
├── preprocessing.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
└── results/
    ├── metrics.json
    └── model_comparison.csv
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Pickle

---

## ▶️ Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/your-username/house-price-prediction.git
```

Move into the project directory:

```bash
cd house-price-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

Make sure the repository contains:

```text
app.py
requirements.txt
models/house_price_model.pkl
results/metrics.json
data/train.csv
```

Then connect the GitHub repository to Streamlit Community Cloud and select:

```text
app.py
```

as the main application file.

---

## ⚠️ Disclaimer

The predicted price is a machine learning estimate based on patterns learned from the training dataset.

It should not be considered a professional property valuation.

---

## 👨‍💻 Author

**Mahmud Umar Yahaya**

AI Engineer | Computer Vision Engineer | Data Scientist

