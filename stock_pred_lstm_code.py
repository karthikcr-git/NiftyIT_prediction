#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('stock_data_10y.csv') #stock_data_thin, stock_data_10y
df = df[['Date','Open','Close']]
df = df[(df.Date!='-') & (df.Open!='-') & (df.Close!='-')]

#number of samples for training - change this to total size*.70 approximately
train_num=int(df.shape[0]*0.9)


#print the head
print("The data frame Head:", df.head())

#setting index as date
#added this part
#df['Date'] = pd.to_datetime(df.Date,format='%d-%b-%Y') #stock_data_thin
df['Date'] = pd.to_datetime(df.Date,format='%d-%b-%y') #stock_data_10y
df.index = df['Date']

#plot the original data
plt.figure(figsize=(16,8))
plt.suptitle('price original', fontsize=16)
plt.plot(df['Close'], label='Close Price history')

#plt.show()

#################################################################################
#LSTM
#################################################################################

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:train_num,:]  #change numbers
valid = dataset[train_num:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=100))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mse','mae', 'mape', 'cosine_proximity'])
history = model.fit(x_train, y_train, epochs=6, batch_size=1, verbose=2)

#predicting 392 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print("The RMS Value: ", rms)

#for plotting
train = new_data[:train_num] #change numbers
valid = new_data[train_num:]
valid['Predictions'] = closing_price
plt.figure()
plt.suptitle('LSTM Prediction', fontsize=16)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

#plt.show()

#metrics for the calculation
from sklearn.metrics import r2_score
score = r2_score(valid[['Predictions']], valid[['Close']])
print('R-squared score for the test set:', round(score,4))


for i in history.history:
    print("\n\n", "-"*10)
    print(i,": ", "\nmax=",max(history.history[i]),"\nmin=", min(history.history[i]))
    print("-="*10)


#print out the graphs for the metrics
fig, axs = plt.subplots(2, 3)
plt.suptitle('Metrics LSTM vs Epoches', fontsize=16)
axs[0, 0].plot(history.history['loss'])
axs[0, 0].set_title('loss')

axs[0, 1].plot(history.history['accuracy'])
axs[0, 1].set_title('accuracy')

axs[0, 2].plot(history.history['mse'])
axs[0, 2].set_title('mse')

axs[1, 0].plot(history.history['mae'])
axs[1, 0].set_title('mae')

axs[1, 1].plot(history.history['mape'])
axs[1, 1].set_title('mape')

axs[1, 2].plot(history.history['cosine_proximity'])
axs[1, 2].set_title('cosine_proximity')

"""
f = open("LSTM_pred_values.txt", "a")
f.write("For LSTM prediction (results, errors): \n")
f.write("Y_test:")
f.write(str(valid['Close'].values))
f.write("\n\n")

f.write("results_output:")
f.write(str(valid['Predictions'].values))
f.write("\n\n")

f.write("errors:")
f.write(str(valid['Close']-valid['Predictions']))
f.write("\n\n")

f.close()
"""


plt.show()