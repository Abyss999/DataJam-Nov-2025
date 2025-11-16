from Imports import *


def loader():
    df = pd.read_csv("Dataset/ai4i2020.csv")
    df = df.drop(columns=["UDI", "Product ID"])

    # separates features and targets 
    X = df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]) # targets 
    y = df["Machine failure"]

    # encodes categorical features
    encoder = LabelEncoder()
    X["Type"] = encoder.fit_transform(X["Type"])

    # scales the numeric features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df.to_csv("data.csv")

    return X_scaled, y, X