#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("train.csv")
df.head(10000).to_csv("train_sample.csv", index=False)


# In[2]:


import pandas as pd

# Load the dataset sample
df = pd.read_csv("train_sample.csv", parse_dates=["date"])

# View basic info
print(df.info())
print(df.head())


# In[18]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# In[4]:


#Load Dataset
train = pd.read_csv("train_sample.csv", parse_dates=["date"])
stores = pd.read_csv("stores.csv")
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
oil = pd.read_csv("oil.csv", parse_dates=["date"])


# In[5]:


# === STEP 2: Merge and Preprocess ===
df = train.merge(stores, on="store_nbr", how="left")
df = df.merge(transactions, on=["store_nbr", "date"], how="left")
df = df.merge(oil, on="date", how="left")  # Add oil prices


# In[6]:


# Fill missing values
df["transactions"].fillna(0, inplace=True)
df["dcoilwtico"].interpolate(method='linear', inplace=True)


# In[22]:


# Fill remaining NaNs (safe default for numeric ML input)
X = X.fillna(0)


# In[23]:


# Add calendar features
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year


# In[24]:


print(df.columns.tolist())


# In[27]:


# Add 'city' and 'state' to one-hot encoding
categorical_cols = ["family", "type", "cluster", "city", "state"]
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)


# In[28]:


#STEP 3: Feature & Label Preparation
FEATURES = [col for col in df.columns if col not in ["id", "date", "sales"]]
X = df[FEATURES]
y = df["sales"]


# In[38]:


# Ensure no NaNs are present in features before training
X = X.fillna(0)  # or choose mean/median if more appropriate

# Recreate the train-test split after filling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[39]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# In[40]:


# === STEP 5: Train 3 Models and Compare ===
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}


# In[41]:


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}



# In[42]:


# Print model comparison
print("\n=== Model Comparison ===")
for name, scores in results.items():
    print(f"{name} - MAE: {scores['MAE']:.2f}, RMSE: {scores['RMSE']:.2f}, R2: {scores['R2']:.4f}")


# In[46]:


# === STEP 3: Final Random Forest Model (Untuned) ===
best_rf = RandomForestRegressor(n_estimators=100, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2 = r2_score(y_test, y_pred_rf)

print("\n=== Final Random Forest Model (Untuned) ===")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))


# In[52]:


# === STEP 4: SHAP Explanation for Individual Predictions ===
import shap
explainer = shap.Explainer(best_rf, X_train)
shap_values = explainer(X_test[:100])  # Limit to first 100 for performance

# Plot summary and individual explanation
shap.plots.beeswarm(shap_values, max_display=15)
shap.plots.bar(shap_values)
shap.plots.waterfall(shap_values[0])  # Explain the first prediction


# In[47]:


#Deep Learning Model (LSTM) ===
X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
X_train_lstm, X_test_lstm = X_lstm[:len(X_train)], X_lstm[len(X_train):]

model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

y_pred_lstm = model.predict(X_test_lstm).flatten()
lstm_mae = mean_absolute_error(y_test, y_pred_lstm)
lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
lstm_r2 = r2_score(y_test, y_pred_lstm)

print("\nLSTM Model - MAE: {:.2f}, RMSE: {:.2f}, R2: {:.4f}".format(lstm_mae, lstm_rmse, lstm_r2))


# In[48]:


# === STEP 5: Visual Comparison ===
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_rf, label='Random Forest Forecast')
plt.title("Random Forest Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from trained Random Forest model
importances = best_rf.feature_importances_
feature_names = X.columns

# Sort by importance (descending)
indices = np.argsort(importances)[::-1]
top_n = 20  # Top N most important features

# Plot
plt.figure(figsize=(12, 6))
plt.title("Top 20 Feature Importances (Random Forest)")
plt.bar(range(top_n), importances[indices[:top_n]], align="center")
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

