import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, mse
from sklearn.preprocessing import MinMaxScaler

# load the pretrained GRU model
model = load_model('C:/Users/rahaf/httpsgithub.com/Tanabbah_capstone/GRU_model (6).h5', custom_objects={'MeanSquaredError': MeanSquaredError(), 'mse': mse})

# load the dataset
df = pd.read_csv('final_dataset.csv', index_col='Datetime', parse_dates=True)

# scaling the target 
scaler = MinMaxScaler()
scaler.fit(df[['Traffic Density']])

def prepare_input_data(df, selected_datetime):
    past_data = df.loc[selected_datetime - pd.Timedelta(minutes=60): selected_datetime]

    if len(past_data) < 60:
        return None, "Not enough data to make a prediction."
    
    if len(past_data) > 60:
        past_data = past_data.tail(60) 

    input_data = past_data['Traffic Density'].values
    input_data = input_data.reshape(1, 60, 1) 
    input_data = scaler.transform(input_data.reshape(-1, 1)).reshape(1, 60, 1)  

    return input_data, None


def predict_traffic_flow(selected_datetime):
    input_data, error = prepare_input_data(df, selected_datetime)

    if input_data is None:
        return None, error
    
    # make the prediction
    predicted_value = model.predict(input_data)
    predicted_value = scaler.inverse_transform(predicted_value)
    
    # checking the traffic flow based on the prediction value
    traffic_density = predicted_value[0][0]
    if traffic_density <= 0.2:
        traffic_condition = "Light Traffic"

    elif traffic_density <= 0.4:
        traffic_condition = "Medium Traffic"

    else:
        traffic_condition = "Heavy Traffic"

    return traffic_density, traffic_condition