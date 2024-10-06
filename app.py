import streamlit as st

from accident_model import load_accident_model
from PIL import Image


# logo
st.image("C:/Users/rahaf/httpsgithub.com/Tanabbah_capstone/Tanabbah-Logo.jpeg", use_column_width=True)

# load accident model
accident_model = load_accident_model()



import pandas as pd
import matplotlib.pyplot as plt
from Traffic_prediction import predict_traffic_flow
from accident_model import detect_accidents


# the user selection to predict the next hour of traffic
selected_date = st.date_input("Select a date")
selected_time = st.time_input("Select a time")


selected_dt = pd.to_datetime(f"{selected_date} {selected_time}")

if st.button("Predict Traffic"):
    
    traffic_density, traffic_flow = predict_traffic_flow(selected_dt)

    if traffic_density is None:
        st.error(traffic_flow) 
    else:
        st.write(f"Predicted traffic density for {selected_dt + pd.Timedelta(hours=1)}: {traffic_density:.2f}")
        st.write(f"Predicted traffic flow: {traffic_flow}")







# Input for accident detection
accident_model = load_accident_model ()

uploaded_file = st.file_uploader("Upload an image for accident detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Accidents"):
        is_accident = accident_model(accident_model, image)
        if is_accident:
            st.write("Accident Detected!")
        else:
            st.write("No Accident Detected.")
            

