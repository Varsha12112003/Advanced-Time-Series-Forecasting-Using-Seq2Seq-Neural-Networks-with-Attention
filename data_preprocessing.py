import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.interpolate().fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler
