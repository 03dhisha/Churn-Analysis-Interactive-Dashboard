import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.set_page_config(page_title='Sales & Churn Intelligence', layout='wide', page_icon='📊')

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title('Controls')
uploaded = st.sidebar.file_uploader('"C:/Users/dhisha/Downloads/sales_data.csv"', type=['csv'])

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv('sales_data.csv')

    # Fix date parsing - handle different date formats
    if 'Date' in df.columns:
        # Try different date formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    st.sidebar.success('Data Loaded: {} rows'.format(len(df)))

except FileNotFoundError:
    st.error("sales_data.csv file not found. Please upload the file using the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Check if required columns exist
required_columns = ['Date', 'Country', 'Product_Category', 'Customer_ID', 'Revenue', 'Profit', 'Order_Quantity',
                    'Churn']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Sidebar filters
min_date, max_date = df['Date'].min(), df['Date'].max()
date_range = st.sidebar.date_input('Date range', [min_date.date(), max_date.date()])

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

# Safe unique value extraction
try:
    countries = sorted(df['Country'].dropna().unique().tolist())
    country_sel = st.sidebar.multiselect('Country', countries)
except:
    country_sel = []

try:
    products = sorted(df['Product_Category'].dropna().unique().tolist())
    product_sel = st.sidebar.multiselect('Product Category', products)
except:
    product_sel = []

show_quarter = st.sidebar.checkbox('Show quarterly trend', value=True)

# Filter data
mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
if country_sel:
    mask &= df['Country'].isin(country_sel)
if product_sel:
    mask &= df['Product_Category'].isin(product_sel)
df_f = df[mask].copy()

if len(df_f) == 0:
    st.warning("No data available for selected filters.")
    st.stop()

# Feature Engineering
df_f['Month_Year'] = df_f['Date'].dt.to_period('M').astype(str)
df_f['Quarter'] = df_f['Date'].dt.to_period('Q').astype(str)

# Purchase Frequency per customer (orders per month)
try:
    orders_per_cust = df_f.groupby(['Customer_ID', 'Month_Year']).size().groupby(level=0).mean().rename(
        'Purchase_Frequency')
    df_f = df_f.merge(orders_per_cust, on='Customer_ID', how='left')
    df_f['Purchase_Frequency'] = df_f['Purchase_Frequency'].fillna(1)
except:
    df_f['Purchase_Frequency'] = 1

# Engagement Score - handle missing columns gracefully
engagement_components = []
if 'Hour_Spend_Per_Week' in df_f.columns:
    engagement_components.append(df_f['Hour_Spend_Per_Week'].fillna(df_f['Hour_Spend_Per_Week'].median()) * 0.5)
if 'Devices_Registered' in df_f.columns:
    engagement_components.append(df_f['Devices_Registered'].fillna(df_f['Devices_Registered'].median()) * 0.8)
if 'Satisfaction_Score' in df_f.columns:
    engagement_components.append(df_f['Satisfaction_Score'].fillna(df_f['Satisfaction_Score'].median()) * 1.2)

if engagement_components:
    df_f['Engagement_Score'] = sum(engagement_components)
else:
    df_f['Engagement_Score'] = df_f['Revenue'] / df_f['Revenue'].max()  # Fallback

# Subscription Tenure in years
if 'Tenure_Months' in df_f.columns:
    df_f['Subscription_Tenure'] = (df_f['Tenure_Months'].fillna(df_f['Tenure_Months'].median())) / 12.0
else:
    df_f['Subscription_Tenure'] = 1.0  # Default value

# ----------------------
# KPIs
# ----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Revenue', f"${df_f['Revenue'].sum():,.0f}")
col2.metric('Total Profit', f"${df_f['Profit'].sum():,.0f}")
col3.metric('Total Orders', int(df_f['Order_Quantity'].sum()))
col4.metric('Churn Rate', f"{(df_f['Churn'].mean() * 100):.1f}%")

st.markdown('---')

# ----------------------
# Sales Trends
# ----------------------
left, right = st.columns([2, 1])

with left:
    # Monthly trend
    monthly = df_f.groupby('Month_Year', as_index=False).agg(
        {'Revenue': 'sum', 'Profit': 'sum', 'Order_Quantity': 'sum'}).sort_values('Month_Year')
    fig_month = px.line(monthly, x='Month_Year', y='Revenue', markers=True,
                        title='Monthly Revenue($) Trend',
                        hover_data={'Revenue': ':,.0f', 'Month_Year': True})
    fig_month.update_traces(hovertemplate='Month %{x}<br>Revenue %{y:,}<extra></extra>')
    fig_month.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_month, use_container_width=True)

    if show_quarter:
        qtr = df_f.groupby('Quarter', as_index=False).agg({'Revenue': 'sum', 'Profit': 'sum'})
        fig_q = px.bar(qtr, x='Quarter', y='Revenue', title='Quarterly Revenue($)',
                       hover_data={'Revenue': ':,.0f', 'Quarter': True})
        fig_q.update_traces(hovertemplate='Quarter %{x}<br>Revenue %{y:,}<extra></extra>')
        st.plotly_chart(fig_q, use_container_width=True)

with right:
    # Top products - handle missing Product column
    product_column = 'Product' if 'Product' in df_f.columns else 'Product_Category'
    top_prod = df_f.groupby(product_column, as_index=False).agg(
        {'Revenue': 'sum', 'Order_Quantity': 'sum'}).sort_values('Revenue', ascending=False).head(15)
    fig_top = px.bar(top_prod, x='Revenue', y=product_column, orientation='h',
                     title=f'Top 15 {product_column}s by Revenue($)',
                     hover_data={'Revenue': ':,.0f', 'Order_Quantity': ':,.0f'})
    fig_top.update_traces(
        hovertemplate=f'{product_column} %{{y}}<br>Revenue %{{x:,}}<br>Units %{{customdata[0]:,}}<extra></extra>')
    st.plotly_chart(fig_top, use_container_width=True)

# Churn vs Revenue impact heatmap
try:
    heat = df_f.pivot_table(index='Product_Category', columns='Churn', values='Revenue', aggfunc='sum').fillna(0)
    heat = heat.rename(columns={0: 'Active', 1: 'Churned'})
    fig_heat = px.imshow(heat, text_auto=True, title='Revenue($) by Product Category vs Churn Status',
                         color_continuous_scale='Blues')
    st.plotly_chart(fig_heat, use_container_width=True)
except Exception as e:
    st.warning(f"Could not create churn heatmap: {e}")

st.markdown('---')
st.subheader('Customer Segmentation & Model Insights')

# ----------------------
# Modeling
# ----------------------
try:
    # Prepare features - only use columns that exist
    base_feature_cols = ['Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit', 'Cost', 'Revenue']
    feature_cols = [col for col in base_feature_cols if col in df_f.columns]

    # Add engineered features
    engineered_cols = ['Purchase_Frequency', 'Engagement_Score', 'Subscription_Tenure']
    if 'Devices_Registered' in df_f.columns:
        engineered_cols.append('Devices_Registered')
    feature_cols.extend(engineered_cols)

    # Categorical columns
    base_cat_cols = ['Country', 'Product_Category']
    if 'Customer_Gender' in df_f.columns:
        base_cat_cols.append('Customer_Gender')
    cat_cols = [col for col in base_cat_cols if col in df_f.columns]

    # Prepare data
    X = df_f[feature_cols + cat_cols].copy()
    y = df_f['Churn'].astype(int)

    # Fill NaN values
    for col in feature_cols:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')

    numerical = feature_cols
    categorical = cat_cols

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
        ]
    )

    # Logistic Regression
    log_reg = Pipeline(steps=[('prep', preprocess),
                              ('clf', LogisticRegression(max_iter=1000, random_state=42))])

    # Random Forest
    rf = Pipeline(steps=[('prep', preprocess),
                         ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])

    # Check if we have enough data for train/test split
    if len(X) < 10:
        st.warning("Not enough data for machine learning models (minimum 10 samples required).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        log_reg.fit(X_train, y_train)
        rf.fit(X_train, y_train)


        # Predictions & ROC
        def roc_data(model, X_test, y_test, name):
            if hasattr(model.named_steps['clf'], 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.decision_function(X_test)
            fpr, tpr, thr = roc_curve(y_test, proba)
            return {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr), 'name': name}


        roc_lr = roc_data(log_reg, X_test, y_test, 'Logistic Regression')
        roc_rf = roc_data(rf, X_test, y_test, 'Random Forest')


        # Confusion Matrices
        def confmat(model, X, y):
            pred = model.predict(X)
            return confusion_matrix(y, pred)


        cm_lr = confmat(log_reg, X_test, y_test)
        cm_rf = confmat(rf, X_test, y_test)

        cm_tabs = st.tabs(['Logistic Regression', 'Random Forest', 'KMeans Segmentation'])

        with cm_tabs[0]:
            fig_cm_lr = px.imshow(cm_lr, text_auto=True,
                                  labels=dict(x='Predicted', y='Actual', color='Count'),
                                  x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                                  title='Logistic Regression - Confusion Matrix',
                                  color_continuous_scale='Blues')
            st.plotly_chart(fig_cm_lr, use_container_width=True)

            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=roc_lr['fpr'], y=roc_lr['tpr'], mode='lines', name=f"LR AUC={roc_lr['auc']:.3f}"))
            fig_roc.add_trace(
                go.Scatter(x=roc_rf['fpr'], y=roc_rf['tpr'], mode='lines', name=f"RF AUC={roc_rf['auc']:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curves', xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate', legend_title='Model')
            st.plotly_chart(fig_roc, use_container_width=True)

        with cm_tabs[1]:
            fig_cm_rf = px.imshow(cm_rf, text_auto=True,
                                  labels=dict(x='Predicted', y='Actual', color='Count'),
                                  x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                                  title='Random Forest - Confusion Matrix',
                                  color_continuous_scale='Blues')
            st.plotly_chart(fig_cm_rf, use_container_width=True)

            # Feature importances from RF
            rf_clf = rf.named_steps['clf']
            # Get feature names after preprocessing
            ohe = rf.named_steps['prep'].named_transformers_['cat']
            ohe_features = ohe.get_feature_names_out(categorical)
            all_features = numerical + list(ohe_features)
            importances = rf_clf.feature_importances_
            imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance',
                                                                                                    ascending=False).head(
                15)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                             title='Random Forest - Top Feature Importances',
                             color='Importance', color_continuous_scale='viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

        with cm_tabs[2]:
            # KMeans on scaled numeric features for segmentation
            seg_features = [col for col in
                            ['Order_Quantity', 'Revenue', 'Profit', 'Engagement_Score', 'Subscription_Tenure',
                             'Devices_Registered'] if col in df_f.columns]

            if len(seg_features) >= 2:
                X_seg = df_f[seg_features].copy()
                # Fill NaN values
                for col in seg_features:
                    X_seg[col] = X_seg[col].fillna(X_seg[col].median())

                scaler = StandardScaler()
                X_seg_scaled = scaler.fit_transform(X_seg)

                # Determine optimal number of clusters (max 5 for visualization)
                n_clusters = min(4, len(df_f) // 10, len(df_f))
                n_clusters = max(2, n_clusters)

                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(X_seg_scaled)
                df_seg = df_f.copy()
                df_seg['Cluster'] = clusters.astype(str)

                # Scatter plot (Revenue vs Engagement) with cluster color
                fig_seg = px.scatter(df_seg, x='Engagement_Score', y='Revenue', color='Cluster',
                                     hover_data=['Customer_ID', 'Product_Category'] + (
                                         ['Country'] if 'Country' in df_seg.columns else []),
                                     title='KMeans Segmentation: Engagement vs Revenue($)',
                                     labels={'Engagement_Score': 'Engagement Score'})
                st.plotly_chart(fig_seg, use_container_width=True)

                # Cluster summary
                cluster_summary = df_seg.groupby('Cluster')[seg_features + ['Churn']].mean().round(2)
                st.subheader("Cluster Summary")
                st.dataframe(cluster_summary, use_container_width=True)
            else:
                st.warning("Not enough numeric features for clustering analysis.")

except Exception as e:
    st.error(f"Error in modeling section: {e}")
    st.write("Available columns:", df_f.columns.tolist())

st.markdown('---')
st.caption('Hover over points/bars/heatmap cells to see exact counts and values.')