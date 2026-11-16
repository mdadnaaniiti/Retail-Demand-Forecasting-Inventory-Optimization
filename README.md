# Multi-Store Retail Demand Forecasting and Inventory Optimization

🚀 **Live Demo:** [Click here to explore the deployed dashboard](https://retail-demand-forecasting-inventory-optimization-da28kf9yldvry.streamlit.app/)



An end-to-end Machine Learning project that forecasts retail product demand across multiple stores and uses demand predictions to generate inventory replenishment recommendations.

The project combines time-series feature engineering, XGBoost forecasting, inventory optimization, policy backtesting and an interactive Streamlit dashboard.

## 1. Project Overview

Retail businesses need to maintain sufficient inventory to meet customer demand while avoiding unnecessary stock accumulation. Poor demand estimation can lead to stockouts, lost sales and inefficient inventory management.

This project develops a data-driven approach to:

- Analyze historical retail sales patterns.
- Identify weekly and monthly demand seasonality.
- Forecast product demand for multiple stores.
- Calculate inventory requirements and reorder recommendations.
- Compare historical-demand-based and forecast-based inventory policies.
- Visualize results through an interactive dashboard.

## 2. Dataset

**Dataset:** Store Item Demand Forecasting Dataset

**Source:** [Kaggle – Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)

| Attribute | Description |
|---|---|
| Historical period | 2013–2017 |
| Number of stores | 10 |
| Number of products | 50 |
| Store-product combinations | 500 |
| Training records | 913,000 |
| Target variable | Daily sales |

**Dataset features:**

- `date`: Sales transaction date
- `store`: Store identifier
- `item`: Product identifier
- `sales`: Number of units sold

The dataset does not contain actual inventory levels, supplier lead times, purchase orders or inventory costs. Therefore, inventory optimization is evaluated through a simulation using explicitly defined assumptions.

## 3. Exploratory Data Analysis

Performed exploratory analysis to understand retail demand patterns.

Key observations:

- Overall sales show an increasing trend across the historical period.
- Weekly seasonality is present, with higher average sales towards weekends.
- July shows the highest average daily sales among the months.
- Store-level and product-level demand varies considerably.
- Some products exhibit higher demand variability than others.

## 4. Feature Engineering

Created time-series and historical demand features to capture demand patterns.

**Calendar features**
- Year
- Month
- Day
- Day of week
- Day of year
- Weekend indicator

**Lag features**
- Lag 1
- Lag 7
- Lag 14
- Lag 28

**Rolling statistics**
- 7-day rolling mean
- 14-day rolling mean
- 30-day rolling mean

Lag and rolling features were constructed using previous observations to avoid target leakage.

## 5. Demand Forecasting

Implemented baseline forecasting methods and machine learning models.

**Baseline models**
- Naive forecasting
- Seasonal naive forecasting

**Machine learning model**
- XGBoost Regressor

A multi-horizon XGBoost model was also developed to predict demand for each of the next seven days using a shared model and a forecast-horizon feature.

### Model Performance

For the rolling one-step-ahead validation experiment on 2017:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 11.2447 | 15.3892 | 19.12% |
| Seasonal Naive | 8.8415 | 11.5967 | 15.03% |
| XGBoost | 6.1180 | 7.9472 | 10.40% |

The multi-horizon XGBoost model achieved:

- Daily MAE: 6.1008
- Daily RMSE: 7.9243
- Daily WAPE: 10.37%
- Aggregate 7-day demand WAPE: 4.48%

The 7-day aggregate WAPE measures total demand prediction error across the evaluated forecast windows. It is not the error for every individual store-product combination.

## 6. Inventory Optimization

Developed a forecast-based inventory planning approach using predicted seven-day demand.

The inventory policy incorporates:

- Forecasted lead-time demand
- Historical demand variability
- Safety stock
- Reorder point
- Target inventory level
- Replenishment quantity
- Days of inventory
- Stockout-risk classification

**Simulation assumptions:**

- Lead time: 7 days
- Review period: 7 days
- Historical demand statistics used to estimate safety buffers
- Simulated initial inventory levels
- Replenishment decisions based on inventory requirements

The application generates recommendations for all 500 store-product combinations.

### Inventory Recommendation Results

| Metric | Result |
|---|---:|
| Store-product combinations | 500 |
| Reorder required | 242 |
| Sufficient stock | 258 |
| Total replenishment quantity | 116,731 units |

These are simulated inventory recommendations, not actual retailer stock records.

## 7. Inventory Policy Backtesting

Compared two inventory policies using a simulated historical backtest against 2017 demand:

1. Historical-demand-based inventory policy
2. Forecast-based inventory policy

### Backtesting Results

| Metric | Historical Policy | Forecast-based Policy |
|---|---:|---:|
| Service level | 86.84% | 94.05% |
| Unmet demand | 1,409,760 | 637,142 |
| Stockout days* | 25,844 | 13,237 |
| Average inventory | 253.78 | 287.44 |
| Total orders | 16,884 | 19,114 |

**Key findings:**

- Service level improved by 7.21 percentage points.
- Unmet demand decreased by approximately 54.8%.
- Stockout days decreased by approximately 48.8%.
- Average inventory increased by approximately 13.3%.

The forecast-based policy improved simulated product availability while maintaining a higher average inventory level.

*Stockout days represent store-product-days with unmet demand, rather than unique calendar days.*

The backtesting results are simulation-based and depend on assumptions including fixed lead time, weekly review, lost sales and simulated initial inventory. They should not be interpreted as actual retailer performance.

## 8. Interactive Streamlit Dashboard

Developed a multi-page dashboard to explore sales, forecasts and inventory recommendations.

**Dashboard modules:**

1. **Overview:** Business KPIs, overall sales trends, store-wise sales and monthly seasonality.
2. **Sales Analysis:** Interactive store and product filters, daily sales trends, top products and weekday demand patterns.
3. **Demand Forecasting:** Seven-day XGBoost predictions with store and product selection.
4. **Inventory Optimization:** Reorder recommendations, replenishment quantities, stockout-risk distribution and inventory filters.
5. **Model Backtesting:** Comparison of historical and forecast-based inventory policies.

**Important:** The displayed seven-day forecast demonstration uses the dataset's final historical date, 31 December 2017. It is not a live forecast for the current date.

## 9. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Altair |
| Machine Learning | Scikit-learn, XGBoost |
| Time-series analysis | Statsmodels |
| Application development | Streamlit |
| Model persistence | Joblib |
| Development environment | VS Code, Jupyter Notebook |