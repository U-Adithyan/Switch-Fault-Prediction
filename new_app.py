import streamlit as st
import pandas as pd
import emd
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
    file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if file is not None:
        if file.name.endswith("csv"):
            input_data = pd.read_csv(file)
        else:
            input_data = pd.read_excel(file)
        
        input_data = input_data[20001:40000]

        st.write("### Input Data")
        st.write(input_data.head())
        
        st.write("### Feature Extraction")
        
        config = emd.sift.get_config('sift')
        config['imf_opts/sd_thresh'] = 0.05
        config['extrema_opts/method'] = 'rilling'
        imf_opts = config['imf_opts']
        
        input_data.rename(columns={"Var2_1": "A", "Var2_2": "B", "Var2_3": "C"}, inplace=True)
        input_data = input_data[["A", "B", "C"]]
        
        data = {}
        
        for col in input_data.columns:
            col_data = input_data[col].to_numpy().ravel()
            imf1, _ = emd.sift.get_next_imf(col_data[:, None], **imf_opts)
            imf2, _ = emd.sift.get_next_imf(col_data[:, None]-imf1, **imf_opts)
            imf3, _ = emd.sift.get_next_imf(col_data[:, None]-imf1-imf2, **imf_opts)
            imf4, _ = emd.sift.get_next_imf(col_data[:, None]-imf1-imf2-imf3, **imf_opts)
            imf5, _ = emd.sift.get_next_imf(col_data[:, None]-imf1-imf2-imf3-imf4, **imf_opts)
            
            residual = col_data[:, None]-imf1-imf2-imf3-imf4-imf5
            a1=np.sqrt(np.sum(np.array(imf1) ** 2)) / len(imf1)
            a2=np.sqrt(np.sum(np.array(imf2) ** 2)) / len(imf2)
            a3=np.sqrt(np.sum(np.array(imf3) ** 2)) / len(imf3)
            a4=np.sqrt(np.sum(np.array(imf4) ** 2)) / len(imf4)
            a5=np.sqrt(np.sum(np.array(imf5) ** 2)) / len(imf5)
            ar=np.sqrt(np.sum(np.array(residual) ** 2)) / len(residual)
            T=a1+a2+a3+a4+a5+ar
            
            data[f'P{col}0'] = round((a1/T)*100)
            data[f'P{col}1'] = round((a2/T)*100)
            data[f'P{col}2'] = round((a3/T)*100)
            data[f'P{col}3'] = round((a4/T)*100)
            data[f'P{col}4'] = round((a5/T)*100)
            data[f'P{col}r'] = round((ar/T)*100)
            data[f'P{col}m'] = np.median(residual)
        
        df = pd.DataFrame(list(data.items()), columns=['Feature', 'Value'])
        st.dataframe(df, width=600, hide_index=True)
        
        sc = load_scaler()
        model = load_model()
        
        data = pd.DataFrame(data, index=[0])
        data = pd.DataFrame(sc.transform(data), columns=data.columns)
        predictions = model.predict(data)
        st.write("### Prediction")
        st.write(f"# {predictions[0]}")

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
