from operator import imod
from pyexpat import model
from statistics import mode
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

dataFrame = pd.read_excel("C:\\python\\TENSORFLOW\\merc.xlsx")

def tensor_1():
    myValues = dataFrame.head()
    print(myValues)


def tensor_2():
    print("")
    print(dataFrame.head())
    print("")
    print(dataFrame.describe())
    # plt.figure(figsize=(7,5))
    # sbn.displot(dataFrame["price"])
    # sbn.countplot(dataFrame["year"])
    # plt.show()
    # sbn.scatterplot(x="mileage",y="price",data=dataFrame)
    # plt.show()
    # myDataList = dataFrame.sort_values("price",ascending=False).head(20)
    myDataList = dataFrame.sort_values("price",ascending=True).head(20)
    print("")
    print(myDataList)
    

def tensor_3():
    myDataFrame = dataFrame.drop("transmission",axis=1)
    y = myDataFrame["price"].values
    x = myDataFrame.drop("price",axis=1).values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = Sequential()
    model.add(Dense(12,activation="relu"))
    model.add(Dense(12,activation="relu"))
    model.add(Dense(12,activation="relu"))
    model.add(Dense(12,activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam",loss="mse")
    model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)
    kayipVerisi = pd.DataFrame(model.history.history)
    print("")
    print(kayipVerisi)
    kayipVerisi.plot()
    plt.show()

tensor_3()



