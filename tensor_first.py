from operator import imod
from statistics import mode
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

dataframe = pd.read_excel("C:\\python\\\TENSORFLOW\\bisiklet_fiyatlari.xlsx")

def tensorflow_1():
    print(dataframe.head())
    print("")
    print(sbn.pairplot(dataframe))
    sbn.pairplot(dataframe)
    plt.show()
    

def tensorflow_2():
    #veri test
    # y-> label
    # x -> feature (özellik)
    y = dataframe["Fiyat"].values
    x = dataframe[["BisikletOzellik1","BisikletOzellik2"]].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=15)
    print("")
    print("x_train: ",x_train.shape)
    print("x_test : ", x_test.shape)
    print("")
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    print("")
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print(x_train)
    model = Sequential()
    model.add(Dense(5,activation="relu"))
    model.add(Dense(5,activation="relu"))
    model.add(Dense(5,activation="relu"))
    model.add(Dense(5,activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="rmsprop",loss="mse")
    model.fit(x_train,y_train,epochs=250)
    loss = model.history.history["loss"]
    sbn.lineplot(x=range(len(loss)),y=loss)
    ##plt.show()
    print(model.evaluate(x_train,y_train))
    print("")
    testTahminleri = model.predict(x_test)
    print(testTahminleri)
    print("")
    tahminDf = pd.DataFrame(y_test,columns=["Gerçek Y"])
    print(tahminDf)
    testTahminleri = pd.Series(testTahminleri.reshape(330,))
    print("")
    print(testTahminleri)
    print("")
    tahminDf = pd.concat([tahminDf,testTahminleri],axis=1)
    tahminDf.columns = ["Gerçek Y","Tahmin Y"]
    print(tahminDf)
    print("")
    sbn.scatterplot(x = "Gerçek Y",y="Tahmin Y",data=tahminDf)
    # plt.show()

 


tensorflow_2()


