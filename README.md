# 📊 Sales & Churn Intelligence Dashboard  

An **interactive dashboard** built with **Streamlit** and **Plotly** for analyzing **sales performance, customer churn, and customer segmentation**.  
The app combines **data visualization, KPIs, machine learning models, and clustering techniques** to help businesses gain actionable insights into customer behavior and revenue trends.  

## 🔑 Features  

- **Data Upload & Filters**  
  Upload your own CSV or use the default dataset with filters for date range, country, and product category.  
- **KPIs Overview**  
  Displays total revenue, profit, order quantity, and churn rate.  
- **Sales Analysis**  
  - Monthly and quarterly revenue/profit trends  
  - Top products by revenue and order quantity  
  - Revenue vs churn heatmap  
- **Churn Prediction Models**  
  - Logistic Regression & Random Forest with confusion matrices and ROC curves  
  - Feature importance visualization to highlight churn drivers  
- **Customer Segmentation**  
  - KMeans clustering on revenue, engagement score, tenure, etc.  
  - Interactive scatter plots (Engagement vs Revenue)  
  - Cluster summary table with churn insights  

## Install dependencies:
pip install -r requirements.txt

## Run the Streamlit app:
streamlit run churn_analysis.py
