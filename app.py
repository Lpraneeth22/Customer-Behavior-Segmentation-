# =========================
# 📦 Import Dependencies
# =========================
import streamlit as st
import pickle
import pandas as pd

# =========================
# 📂 Load Model & Scaler
# =========================
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scal.pkl", "rb") as f:
    scaler = pickle.load(f)
# =========================
# 🏷️ App Title
# =========================
st.set_page_config(page_title="Customer Behavior Segmentation Predicton", layout="centered")
st.title("📊 Customer Behavior Segmentation Predicton")
st.sidebar.markdown("""
<style>
    .big-link {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: underline;
        margin-bottom: 15px;
        display: block;
        cursor: pointer;
    }
    .big-link:hover {
        color: #d62728;
    }
</style>

<a href="#introduction" class="big-link">Introduction</a>
<a href="#project-overview" class="big-link">Project Overview</a>
<a href="#prediction" class="big-link">Prediction</a>
""", unsafe_allow_html=True)

st.markdown('<h2 id="introduction">📜 Introduction</h2>', unsafe_allow_html=True)
st.write("""
Customer Segmentation is a marketing and analytics strategy used to group customers based on shared characteristics.
It allows businesses to better understand the needs, preferences, and behaviors of their customer base.
Segmentation can be based on demographics, psychographics, geography, or behavioral patterns.
By identifying these groups, companies can create targeted marketing campaigns.
It helps in optimizing resources and improving return on investment (ROI).
Personalized offers can be designed to match specific customer needs.
It also improves customer satisfaction and loyalty.
In our project, we used unsupervised learning techniques to segment customers.
Clustering methods such as K-Means, DBSCAN, and Hierarchical Clustering were applied.
These insights can guide product recommendations, promotions, and strategic decisions.""")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://ezranking.s3.eu-west-2.amazonaws.com/blog/wp-content/uploads/2022/06/10115059/Customer-Segmentation.jpg", width=800)

st.markdown('<h2 id="project-overview">🔍 Project Overview</h2>', unsafe_allow_html=True)

st.header("🎯 Objective")
st.write("""
The objective of this project is to analyze customer behavioral data to identify meaningful segments and predict future purchasing trends. 
By leveraging clustering techniques, the project aims to group customers based on shared characteristics such as spending patterns, product preferences, and demographic attributes. 
These insights will enable businesses to design targeted marketing strategies, improve customer engagement, and optimize resource allocation.""")

st.header("📊 Dataset Overview")


column_data = {
    "Column Name": [
       "ID","Year_Birth","Education","Marital_Status",
"Income","Kidhome","Teenhome","Dt_Customer","Recency",
"MntWines","MntFruits","MntMeatProducts","MntFishProducts",
"MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases",
"NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth","AcceptedCmp1","AcceptedCmp2",
"AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response","Complain","Country","Location"
],
    "Description": [
        "Unique identifier for each customer",
"Year of birth of the customer",
"Education level of the customer",
"Current marital status of the customer",
"Annual household income of the customer (in currency units)",
"Number of small children in the customer's household",
"Number of teenagers in the customer's household",
"Date when the customer enrolled with the company",
"Number of days since the customer's last purchase",
"Amount spent on wine in the last 2 years",
"Amount spent on fruits in the last 2 years",
"Amount spent on meat products in the last 2 years",
"Amount spent on fish products in the last 2 years",
"Amount spent on sweet products in the last 2 years",
"Amount spent on gold products in the last 2 years",
"Number of purchases made with a discount",
"Number of purchases made through the company's website",
"Number of purchases made using a catalog",
"Number of purchases made directly in stores",
"Number of visits to the company's website in the last month",
"1 if customer accepted offer in the 1st campaign, 0 otherwise",
"1 if customer accepted offer in the 2nd campaign, 0 otherwise",
"1 if customer accepted offer in the 3rd campaign, 0 otherwise",
"1 if customer accepted offer in the 4th campaign, 0 otherwise",
"1 if customer accepted offer in the 5th campaign, 0 otherwise",
"1 if customer accepted the offer in the last campaign, 0 otherwise",
"1 if customer has complained in the last 2 years, 0 otherwise",
"Country of residence of the customer",
"Synthetic location field added for analysis (e.g., Telangana, Andhra Pradesh)"

    ]
}
df_columns = pd.DataFrame(column_data)
st.dataframe(df_columns)

st.markdown('<h2 id="prediction"> Prediction</h2>', unsafe_allow_html=True)
st.write("---")


st.markdown("Enter customer details to predict their cluster.")

# =========================
# 📝 User Inputs
# =========================
col1, col2 = st.columns(2)

with col1:
    Income = st.number_input("Income", min_value=1730, max_value=666666, step=100)
    MntWines = st.number_input("MntWines", min_value=0, max_value=1493, step=10)
    MntFruits = st.number_input("MntFruits", min_value=0, max_value=199, step=1)
    MntMeatProducts = st.number_input("MntMeatProducts", min_value=0, max_value=1725, step=10)
    MntFishProducts = st.number_input("MntFishProducts", min_value=0, max_value=259, step=1)
    MntSweetProducts = st.number_input("MntSweetProducts", min_value=0, max_value=262, step=1)

with col2:
    MntGoldProds = st.number_input("MntGoldProds", min_value=0, max_value=321, step=1)
    NumDealsPurchases = st.number_input("NumDealsPurchases", min_value=0, max_value=15, step=1)
    NumWebPurchases = st.number_input("NumWebPurchases", min_value=0, max_value=27, step=1)
    NumCatalogPurchases = st.number_input("NumCatalogPurchases", min_value=0, max_value=28, step=1)
    NumStorePurchases = st.number_input("NumStorePurchases", min_value=0, max_value=13, step=1)
    NumWebVisitsMonth = st.number_input("NumWebVisitsMonth", min_value=0, max_value=20, step=1)


# =========================
# 🚀 Prediction Logic
# =========================
if st.button("🔍 Predict Cluster"):
    # ---- Calculate totals ----
    Total_purchas = (NumDealsPurchases + NumWebPurchases +
                     NumCatalogPurchases + NumStorePurchases +
                     NumWebVisitsMonth)

    Total_spend_product = (MntWines + MntFruits + MntMeatProducts +
                           MntFishProducts + MntSweetProducts + MntGoldProds)

    # ---- Create DataFrame ----
    row = pd.DataFrame([[Income, MntWines, MntFruits, MntMeatProducts,
                         MntFishProducts, MntSweetProducts, MntGoldProds,
                         NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
                         NumStorePurchases, NumWebVisitsMonth,
                         Total_purchas, Total_spend_product]],
                       columns=["Income", "MntWines", "MntFruits", "MntMeatProducts",
                                "MntFishProducts", "MntSweetProducts", "MntGoldProds",
                                "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
                                "NumStorePurchases", "NumWebVisitsMonth",
                                "Total_purchas", "Total_spend_product"])

    # ---- Scale Data ----
    row_scaled = scaler.transform(row)

    # ---- Predict ----
    cluster_num = model.predict(row_scaled)[0]

    # ---- Cluster Labels ----
    cluster_labels = {
        0: "Medium income & medium spending customer",
        1: "Low income & low spending customer",
        2: "High income & High spending customer"
    }

    # ---- Display Output ----
    st.success(f"Predicted Cluster: {cluster_num}")
    st.info(f" {cluster_labels.get(cluster_num, 'Unknown cluster')}")
    st.balloons()








