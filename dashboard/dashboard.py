import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from utils import (
    load_assets
)
 
from streamlit_option_menu import option_menu 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


MODELS_EP = [
        "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/model_elasticnet.joblib",
        "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/model_lasso.joblib",
        "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/model_ridge.joblib",        
    ]

SCALER_EP = "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/scaler_standardscaler.joblib"

DATA_EP = [
    "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/X_test_scaled.npy",
    "https://github.com/hng011/energy-consumption-forecast/raw/refs/heads/main/models/y_test.npy",
]

models, data, scaler = load_assets(
    models_ep=MODELS_EP, 
    data_ep=DATA_EP,
    scaler_ep=SCALER_EP,
)


# def _test_eval():
#     import joblib
#     print(models[1])
#     print(models[1].predict(data[0])[:5])
#     model_lasso_local = joblib.load("../models/model_lasso.joblib") 
#     print(model_lasso_local.predict(data[0])[:5])

# _test_eval()

def forecast_page():
    st.image("https://raw.githubusercontent.com/hng011/wok/refs/heads/main/assets/banner_linreg.png")
    st.title("Energy Consumption Prediction")
    st.markdown("### Enter the details below to predict the energy consumption.")


    list_bt = ["Residential", "Commercial", "Industrial"]
    list_dw = ["Weekday", "Weekend"]


    building_type = st.selectbox("Building Type", list_bt)
    square_footage = st.number_input("Square Footage", min_value=0, max_value=100000, value=0)
    occupants = st.number_input("Number of Occupants", min_value=0, max_value=1000, value=0)
    appliances = st.number_input("Appliances Used", min_value=0, max_value=1000, value=0)
    temperature = st.number_input("Average Temperature (°C)", min_value=00.0, max_value=100.0, value=0.0)
    day_of_week = st.selectbox("Day of Week", list_dw)


    # Choosing Model
    list_model = [
        "ElasticNet", 
        "Lasso", 
        "Ridge"]
    choosed_model = st.selectbox("Model", list_model)


    # # Choosing Scaler
    # list_scaler = ["StandardScaler", "MinMaxScaler"]
    # choosed_scaler = st.selectbox("Scaler", list_scaler)


    data_cols = [
            "Square Footage", 
            "Number of Occupants", 
            "Appliances Used", 
            "Average Temperature", 
            "Building Type_Commercial",
            "Building Type_Industrial",
            "Building Type_Residential",
            "Day of Week_Weekday",
            "Day of Week_Weekend"
    ]


    input_data = pd.DataFrame(
        data=[ [square_footage, occupants, appliances, temperature, 0.0, 0.0, 0.0, 0.0, 0.0] ], 
        columns=data_cols
    )


    input_data = input_data.astype("float64")

    # Encode
    if building_type == list_bt[0]: input_data[data_cols[6]] = 1.0
    elif building_type == list_bt[1]: input_data[data_cols[4]] = 1.0
    else: input_data[data_cols[5]] = 1.0

    if day_of_week == list_dw[0]: input_data[data_cols[-2]] = 1.0
    else: input_data[data_cols[-1]] = 1.0

    # scaler
    input_data = scaler.transform(input_data)

    # Predict Energy Consumption
    if st.button("Predict Energy Consumption"):
        model = models[0] if choosed_model == list_model[0] else models[1]
        predicted_energy = model.predict(input_data)[0]
        st.subheader("🔮 Prediction Result")
        st.write(f"### Predicted Energy Consumption (kWh): {predicted_energy:.2f}")

def model_eval_page():
     X_test_scaled, y_test = data[0], data[1]
 
     # this list order based on the models order above
     model_names = []
     for x in range(len(models)):
         match = re.search(r"([^/]+)$", MODELS_EP[x])
         model_names.append(match.group(1).split(".")[0])
 
     res = [{"name":name, "y_pred": model.predict(X_test_scaled)} for name, model in zip(model_names, models)]
 
     for x in res:
         x["mse"] = mean_squared_error(y_test, x["y_pred"])
         x["r2"] = r2_score(y_test, x["y_pred"])
         x["mae"] = mean_absolute_error(y_test, x["y_pred"])
 
     metrics = [
         f"mse_{x["name"].lower()}" for x in res
     ] + [
         f"r2_{x["name"].lower()}" for x in res
     ] + [
         f"mae_{x["name"].lower()}" for x in res
     ]
 
     values = [
         x["mse"] for x in res
     ] + [
         x["r2"] for x in res
     ] + [
         x["mae"] for x in res
     ] 
 
     # DF
     df = pd.DataFrame({"Actual": y_test})
     for x in res:
         df[f"pred_{x["name"]}"] = x["y_pred"]
 
     st.dataframe(df.sample(10, ignore_index=True))
 
     # FIG
     fig, ax = plt.subplots()
     ax.bar(metrics, values)
     ax.set_title("Model Eval Metrics")
     plt.xticks(rotation=90)
     for i, v in enumerate(values):
         ax.text(i, v + 0.02, f"{v:.4f}", ha='center')
     st.pyplot(fig)


if __name__ == "__main__":
    menus = [
        "Forecasting Tool", 
        "Model Evaluation"
    ]
    
    with st.sidebar:
        try:
            selected = option_menu(menu_title="Dashboard Menu",
                options=menus,
                default_index=0
            )
        except:
            st.write("streamlit_option_menu module not found")
            st.write("Please install the module using the following command")
            st.write("`pip install streamlit-option-menu`")
            
    if selected == menus[0]:
        forecast_page()
    if selected == menus[1]:
        model_eval_page()