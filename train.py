import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("House_Prices.csv")
X = df[["Area","Bedrooms","Bathrooms","Age"]]
y = df["Price"]

model = LinearRegression().fit(X, y)
pickle.dump(model, open("model.pkl","wb"))
