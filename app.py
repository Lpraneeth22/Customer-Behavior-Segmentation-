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
st.set_page_config(page_title="Customer Segmentation Predictor", layout="centered")
st.title("📊 Customer Segmentation Predictor")
st.markdown("Enter customer details to predict their cluster.")

# =========================
# 📝 User Inputs
# =========================
col1, col2 = st.columns(2)

with col1:
    Income = st.number_input("Income", min_value=0)
    MntWines = st.number_input("MntWines", min_value=0)
    MntFruits = st.number_input("MntFruits", min_value=0)
    MntMeatProducts = st.number_input("MntMeatProducts", min_value=0)
    MntFishProducts = st.number_input("MntFishProducts", min_value=0)
    MntSweetProducts = st.number_input("MntSweetProducts", min_value=0)

with col2:
    MntGoldProds = st.number_input("MntGoldProds", min_value=0)
    NumDealsPurchases = st.number_input("NumDealsPurchases", min_value=0)
    NumWebPurchases = st.number_input("NumWebPurchases", min_value=0)
    NumCatalogPurchases = st.number_input("NumCatalogPurchases", min_value=0)
    NumStorePurchases = st.number_input("NumStorePurchases", min_value=0)
    NumWebVisitsMonth = st.number_input("NumWebVisitsMonth", min_value=0)

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
    st.info(f"📌 Label: {cluster_labels.get(cluster_num, 'Unknown cluster')}")

