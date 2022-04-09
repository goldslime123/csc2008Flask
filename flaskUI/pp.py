# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow import keras
from sklearn.metrics import mean_squared_error

# %%
# import temperature data
data = pd.read_csv('temp/temperature.csv').values
data = np.delete(data, 0, 1)
# print(data)
n_steps = 4

# %%
# split a multivariate sequence into samples


def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# %%


# convert into input/output
x, y = split_sequences(data, n_steps)


# print(data[20:23])
# for i in range(len(x)):
#     print(x[i], y[i])
n_features = x.shape[2]
# X.shape
data.shape
# print(x.shape[2])

# %%
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# %%
# fit model
model.fit(x, y, epochs=1000, verbose=0)
# demonstrate prediction

# %%
xinput = np.array([[27.37, 23.29], [28.77, 20.87],
                  [28.73, 22.41], [28.13, 20.35]])
xinput = xinput.reshape((1, n_steps, n_features))
temp = model.predict(xinput, verbose=0)
# print(temp)


# %%
newModel = Sequential()
newModel.add(LSTM(100, return_sequences=True,
             batch_input_shape=(1, 4, 2), stateful=True))
newModel.add(LSTM(100))
newModel.add(Dense(n_features))
newModel.compile(optimizer='adam', loss='mean_squared_error')
newModel.set_weights(model.get_weights())

# %%
newtemp = x[-1]
# print(newtemp)
pp = np.delete(newtemp, obj=[0, 0, 1])
pp = pp.flatten()
pp = np.append(pp, values=temp.flatten())
# print(pp)
pp.shape
pp = pp.reshape((1, n_steps, n_features))
# print(pp)


# %%
predict_the_future = newModel.predict(pp, verbose=0)


# %%
tt = np.copy(predict_the_future)
# print(tt)
for i in range(8):
    pp = np.delete(newtemp, obj=[0, 0, 1])
    pp = pp.flatten()
    pp = np.append(pp, values=predict_the_future.flatten())
    pp = pp.reshape((1, n_steps, n_features))
    # print(pp)
    predict_the_future = newModel.predict(pp, verbose=0)
    tt = np.append(tt, predict_the_future)
    # print(tt)
    newModel.reset_states()


# %%
# tt = tt.flatten()
# print(tt)

temperature = list()
tariff = list()

for i in range(0,len(tt),2):
    temperature.append(tt[i])
    tariff.append(tt[i+1])
print(temperature)
print(tariff)


# %%


# %%
