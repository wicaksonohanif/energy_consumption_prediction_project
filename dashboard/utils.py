import requests
import joblib
import re
import numpy as np
import streamlit as st

def fetch_asset(endpoint, file_name):
    
    try:
        response = requests.get(endpoint, stream=True)
        response.raise_for_status()
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        st.stop()
    except:
        st.write("Load assets...")


def load_assets(models_ep: list, data_ep: list, scaler_ep: str):
    
    try:
        
        models = [] 
        for x in range(len(models_ep)):
            file_name = f"model{x}.joblib"
            fetch_asset(endpoint=models_ep[x], file_name=file_name)
            models.append(joblib.load(file_name))
            
        data = []
        for x in range(len(data_ep)):
            match = re.search(r"([^/]+)$", data_ep[x])
            file_name = match.group(1)
            fetch_asset(endpoint=data_ep[x], file_name=file_name)
            
            if file_name.split('.')[-1] == "npy":    
                data.append(np.load(file_name))
        
        scaler_name = "scaler.joblib"
        fetch_asset(endpoint=scaler_ep, file_name=scaler_name)
        scaler = joblib.load(scaler_name)
        
        return models, data, scaler 
        
    except Exception as e:
        print(e)
    