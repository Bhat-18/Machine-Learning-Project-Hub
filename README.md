# Store Sales Time-Series Forecasting

This project presents an end-to-end machine learning pipeline for forecasting store sales using historical data. It incorporates modern techniques in feature engineering, interpretable ML (SHAP), and interactive deployment (Streamlit) to provide actionable business insights for inventory planning and demand prediction.

## Why This Project?

Retail demand forecasting is critical for minimizing overstocking or understocking, optimizing supply chains, and increasing profit margins. This dataset provides a rich blend of temporal, categorical, and promotional features that mirror real-world complexity, making it an ideal candidate for practical, industry-aligned ML modeling.

## Project Highlights

-  Real-World Dataset: Sales data from over 100 stores across different product families
-  Time Series Processing: Includes date-based decomposition (day, month, year, day-of-week)
-  ML Models Compared: Linear Regression, Gradient Boosting, and Random Forest Regressor
-  Selected Model: Random Forest gave best R² and lowest MAE/RMSE
-  Hyperparameter Tuning: GridSearchCV attempted and compared with default parameters
-  Model Evaluation: MAE, RMSE, and R² scores provided
-  Model Explainability: SHAP used to visualize feature impact
-  Deployment Ready: Interactive Streamlit UI for predictions and SHAP force plots

##  Dataset Files Used

- `train.csv` – Historical sales training data  
- `stores.csv` – Store metadata  
- `oil.csv` – Daily oil prices (external regressor)  
- `transactions.csv` – Number of transactions per store  
- `holidays_events.csv` – National/local holidays metadata  
- `test.csv` – Data for final prediction submission  

##  Tech Stack

| Tool/Library       | Purpose                          |
|--------------------|----------------------------------|
| `Python`           | Core programming language        |
| `Pandas` / `NumPy` | Data preprocessing               |
| `Scikit-learn`     | ML models, evaluation, splitting |
| `SHAP`             | Explainable AI                   |
| `Streamlit`        | Web app deployment               |
| `Matplotlib`       | Visualizations                   |
| `Joblib`           | Model serialization              |

##  Model Performance

| Model               | MAE    | RMSE   | R²     |
|--------------------|--------|--------|--------|
| Linear Regression  | 227.30 | 596.24 | 0.5380 |
| Gradient Boosting  | 97.66  | 221.51 | 0.9362 |
| **Random Forest**  | **46.32** | **155.33** | **0.9686** |

➡ **Random Forest** was selected as the final model due to its strong generalization and low error metrics without tuning.

## SHAP Visuals

- **Feature importance plot**: Understand which features influence predictions
- **Force plots**: Visualize contributions to individual predictions
- **Table with exact SHAP values**: Exportable and inspectable

##  Streamlit App

- Interactive UI with:
  - Feature input sliders / dropdowns
  - On-click prediction
  - SHAP force plot visualization for the prediction

To run:
```bash
streamlit run Untitled-1-fixed.py
```

##  Directory Structure

```
.
├── train_sample.csv
├── Untitled-1-fixed.py
├── X_test.csv
├── random_forest_model.pkl
├── README.md
└── streamlit_app.py  ← (Optional)
```

##  Author

**Bhanuprakash Bhat**  
 Master's in Data Science with AI  
 Part-time 

##  Future Enhancements

- Integrate Prophet/LSTM for time-series modeling
- Add support for retraining with new data
- Implement CI/CD with Docker & GitHub Actions
