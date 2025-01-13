import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    return joblib.load("final_rf_model.pkl")

@st.cache_resource
def load_shap_explainer():
    return joblib.load("shap_explainer.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

def main():
    st.title("Switch Fault Detection")

    st.sidebar.header("Upload Your Data")
    file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

    if file is not None:
        if file.name.endswith("csv"):
            input_data = pd.read_csv(file)
        else:
            input_data = pd.read_excel(file)

        st.write(input_data.head())
        
        sc = load_scaler()
        data = pd.DataFrame(sc.fit_transform(input_data), columns=input_data.columns)

        model = load_model()

        predictions = model.predict(data)
        st.write("### Predictions")
        output_data = input_data.copy()
        output_data["Switch Fault"] = predictions
        st.write(output_data.head())
        
        csv = output_data.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.write("### Feature Importance (SHAP)")
        explainer = load_shap_explainer()
        shap_values = explainer.shap_values(data)
        shap_values = [shap_values[:,:,i] for i in range(shap_values.shape[2])]
        
        shap.summary_plot(shap_values, data, class_names= model.classes_)
        plt.gca().set_xlabel('Feature Importance')
        st.pyplot(plt.gcf())
        plt.clf()

if __name__ == "__main__":
    main()
