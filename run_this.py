#!/usr/bin/env python
# coding: utf-8

# In[2]:
#Importing Packages
# WHole code descripon
# in this code , we are using covid-19 data set, by using this data set we are visualizing confirmed cases, deaths,active case all over the world. So writing code
#we can make analyis on give data to check what is happening all over the world . If we look at data set using our code, we can see US has more corona cases ,which is top country
#So we have a task to predict/forecast future corona cases, so we can make startegy to fight against this disease . Forecasting helps us to understand , that what will be happened
# in future .so we can prepare our self.So bu using keras LSTm deep learning model,we are forecasting future cases .

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px

import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings


# In[3]:


train = pd.DataFrame(pd.read_csv("train.csv"))


# In[4]:


print("The shape of training data is = {}".format(train.shape))


# In[5]:

# fillna used to replace /fill the missing cells in columns
train.Province_State.fillna("", inplace = True)
train.ConfirmedCases.fillna("", inplace = True)
train.Fatalities.fillna("", inplace = True)


# In[6]:


train["Country_Region"].unique()


# In[7]:


confirmed_cases_us = train[train["Country_Region"] == "US"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_us = train[train["Country_Region"] == "US"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_italy = train[train["Country_Region"] == "Italy"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_italy = train[train["Country_Region"] == "Italy"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_india = train[train["Country_Region"] == "India"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_india = train[train["Country_Region"] == "India"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_france = train[train["Country_Region"] == "France"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_france = train[train["Country_Region"] == "France"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_china = train[train["Country_Region"] == "China"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_china = train[train["Country_Region"] == "China"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_taiwan = train[train["Country_Region"] == "Taiwan*"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_taiwan = train[train["Country_Region"] == "Taiwan*"].groupby(["Date"]).Fatalities.sum()

confirmed_cases_uk = train[train["Country_Region"] == "United Kingdom"].groupby(["Date"]).ConfirmedCases.sum()
fatal_cases_uk = train[train["Country_Region"] == "United Kingdom"].groupby(["Date"]).Fatalities.sum()

date = train["Date"].unique()


# In[8]:


plt.figure(figsize = (12,8))
plt.plot(date, confirmed_cases_us, color = "b", label = "US")
plt.plot(date, confirmed_cases_italy, color = "g", label = "Italy")
plt.plot(date, confirmed_cases_india, color = "y", label = "India")
plt.plot(date, confirmed_cases_france, color = "r", label = "France")
plt.plot(date, confirmed_cases_china, color = "c", label = "China")
plt.plot(date, confirmed_cases_taiwan, color = "m", label = "Taiwan")
plt.plot(date, confirmed_cases_uk , color = "k", label = "UK")
plt.grid("both")
plt.title("A comparitive study of confirmed cases across the globe")
plt.legend()

plt.plot()


# In[9]:


plt.figure(figsize = (12, 8))
plt.plot(date, fatal_cases_us, color = "b", label = "US")
plt.plot(date, fatal_cases_italy, color = "g", label = "Italy")
plt.plot(date, fatal_cases_india, color = "y", label = "India")
plt.plot(date, fatal_cases_france, color = "r", label = "France")
plt.plot(date, fatal_cases_china, color = "c", label = "China")
plt.plot(date, fatal_cases_taiwan, color = "m", label = "Taiwan")
plt.plot(date, fatal_cases_uk , color = "k", label = "UK")
plt.grid("both")
plt.title("A comparitive study of fatal cases across the globe")
plt.legend()

plt.plot()


# In[10]:


train["Region"] = train["Country_Region"].astype(str) + train["Province_State"].astype(str)
train.drop(["Country_Region" , "Province_State"], axis = 1, inplace = True)


# In[11]:


train.head()


# In[12]:


choro_map = px.choropleth(train, locations = "Region", locationmode = "country names", color = "ConfirmedCases",
                                        hover_name = "Region", animation_frame = "Date")
choro_map.update_layout(title_text = "Global Confirmed Cases", title_x = 0.5,
                         geo = dict(showframe = False, showcoastlines = True))
choro_map.show()


# In[13]:


choro_map = px.choropleth(train, locations = "Region", locationmode = "country names", color = "Fatalities",
                                        hover_name = "Region", animation_frame = "Date")
choro_map.update_layout(title_text = "Fatalities Across The Globe", title_x = 0.5,
                         geo = dict(showframe = False, showcoastlines = True))
choro_map.show()


# In[14]:


df = train


# In[15]:


df_cc = df.pivot(index = "Region", columns = "Date", values ="ConfirmedCases")
df_cc.head()


# In[16]:


df_fc = df.pivot(index = "Region" , columns = "Date", values = "Fatalities")
df_fc.head()


# In[17]:


df_cc.to_csv("confirmed_cases.csv", encoding = "utf-8-sig")
df_fc.to_csv("Fatal_cases.csv", encoding = "utf-8-sig")


# # Analyzing Confirmed Cases

# In[18]:


# In[19]:


train.head()


# In[20]:


train = train.set_index("Id")
train.head()


# In[21]:


train_india = train[train["Region"] == "India"]
train_india


# In[22]:


train_india["ConfirmedCases"] = train_india["ConfirmedCases"].astype(int)
train_india["Fatalities"] = train_india["Fatalities"].astype(int)


# In[23]:


train_india_size = int(len(train_india) * 0.75)
val_india_size = len(train_india) - train_india_size
print("Training size = {}".format(train_india_size))
print("Validation size = {}".format(val_india_size))


# In[24]:


train_india_confirmed_cases = train_india[["ConfirmedCases"]]
train_india_fatal_cases = train_india[["Fatalities"]]


# In[25]:


print(train_india_confirmed_cases, train_india_fatal_cases)


# In[26]:


plt.figure(figsize = (8, 8))
x = np.arange(1, 116, 1)
y1 = train_india_confirmed_cases
y2 = train_india_fatal_cases
plt.plot(x, y1, color = "m", label = "Confirmed Cases in India from Jan-May")
plt.plot(x, y2, color = "r", label = "Fatal Cases in India from Jan-May")
plt.grid(True)
plt.legend()


# ## CONFIRMED CASES :

# In[27]:


len(train_india_confirmed_cases)


# In[28]:


train_india_confirmed_cases


# In[29]:


train_india_confirmed_cases_data = train_india_confirmed_cases.iloc[0:train_india_size]
val_india_confirmed_cases_data = train_india_confirmed_cases.iloc[train_india_size : len(train_india_confirmed_cases)]


# In[30]:


print(len(train_india_confirmed_cases_data))
print(len(val_india_confirmed_cases_data))


# In[31]:


scaler = MinMaxScaler(feature_range = (0,1))
def createDataset(train) :
    train_scaled = scaler.fit_transform(train)
    x_train = []
    y_train = []
    time_step = 2
    for i in range(time_step, train_scaled.shape[0]):
        x_train.append(train_scaled[i-time_step : i , 0])
        y_train.append(train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1 ))
    y_train = np.reshape(y_train, (y_train.shape[0], 1 ))
    return x_train, y_train


# In[32]:


x_train, y_train = createDataset(train_india_confirmed_cases_data)
x_val, y_val = createDataset(val_india_confirmed_cases_data)


# In[33]:


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# In[34]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 1))


# In[35]:


model.compile(tf.keras.optimizers.Adam(lr = 0.001), loss = "mean_squared_error")


# In[36]:


model.summary()


# In[37]:


EPOCHS = 200
BATCH_SIZE = 1


# In[38]:


with tf.device("/device:GPU:0"):
  history = model.fit(x_train, y_train,epochs = EPOCHS, verbose = 1,
                     batch_size = BATCH_SIZE, validation_data = (x_val, y_val))


# In[39]:


x = np.arange(0, EPOCHS, 1)
plt.figure(1, figsize = (20, 12))
plt.subplot(121)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x, history.history["loss"], label = "Training Loss")
plt.plot(x, history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()


# In[40]:


predicted_cases = model.predict(x_val)
predicted_cases = scaler.inverse_transform(predicted_cases)
real_cases = scaler.inverse_transform(y_val)
plt.figure(figsize= (12, 8))
plt.subplot(1,1,1)
plt.plot(real_cases, color = "red", label = "Real Number Of Cases")
plt.plot(predicted_cases, color = "blue", label = "Predicted Number Of Cases (Validation set)")
plt.title("Corona Cases")
plt.xlabel("Time")
plt.ylabel("Case Count")
plt.legend()
plt.grid("both")
plt.show()


# In[41]:


"""
Saving model's topology
"""
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

"""
Saving model's weights
"""
model.save_weights("model.h5")


# # Fatalities :

# In[42]:


len(train_india_fatal_cases)


# In[43]:


train_india_fatal_cases_data = train_india_fatal_cases.iloc[0:train_india_size]
val_india_fatal_cases_data = train_india_fatal_cases.iloc[train_india_size : len(train_india_fatal_cases)]


# In[44]:


print(len(train_india_fatal_cases_data))
print(len(val_india_fatal_cases_data))


# In[45]:


x_train, y_train = createDataset(train_india_fatal_cases_data)
x_val, y_val = createDataset(val_india_fatal_cases_data)


# In[46]:


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# Using the same model architecture to predict. Re-training the same model on Fatal cases.

# In[47]:


with tf.device("/device:GPU:0"):
  history = model.fit(x_train, y_train,epochs = EPOCHS, verbose = 1, batch_size = BATCH_SIZE,
                     validation_data = (x_val, y_val))


# In[48]:


x = np.arange(0, EPOCHS, 1)
plt.figure(1, figsize = (20, 12))
plt.subplot(121)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x, history.history["loss"], label = "Training Loss")
plt.plot(x, history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()


# In[49]:


predicted_cases = model.predict(x_val)
predicted_cases = scaler.inverse_transform(predicted_cases)
real_cases = scaler.inverse_transform(y_val)
plt.figure(figsize= (12, 8))
plt.subplot(1,1,1)
plt.plot(real_cases, color = "red", label = "Real Number Of Fatalities")
plt.plot(predicted_cases, color = "blue", label = "Predicted Number Of Fatalities (Validation set)")
plt.title("Fatal Cases")
plt.xlabel("Time")
plt.ylabel("Case Count")
plt.legend()
plt.grid("both")
plt.show()

