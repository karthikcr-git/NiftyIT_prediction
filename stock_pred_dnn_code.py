import keras
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

#scaling for the inputs
from sklearn.preprocessing import MinMaxScaler
feature_scaler = MinMaxScaler(feature_range=(0, 1))
traget_scaler = MinMaxScaler(feature_range=(0, 1))

#df = pd.read_csv('stock_data_thin.csv') #stock_data_thin, stock_data_10y
df = pd.read_csv('stock_data_10y.csv')
df = df[['Date','Open','Close']]
df = df[(df.Date!='-') & (df.Open!='-') & (df.Close!='-')]

#print(df.head())

df['Date_c'] = pd.to_datetime(df.Date,format='%d-%b-%y') #stock_data_10y
df['Date'] = pd.to_datetime(df.Date,format='%d-%b-%y') #stock_data_10y

#df['Date_c'] = pd.to_datetime(df.Date,format='%d-%b-%Y') #stock_data_thin
#df['Date'] = pd.to_datetime(df.Date,format='%d-%b-%Y') #stock_data_thin
#df.index = df['Date']


#plot the original data for exploration
plt.figure(figsize=(16,8))
plt.suptitle('NIFTY IT Close - original', fontsize=16)
plt.plot(df['Date_c'][:,None],df['Close'], label='Close Price history')
#plt.show()

"""
#this was before scaling
df['Date_year'] = df['Date'].dt.year
df['Date_month'] = df['Date'].dt.month
df['Date_day'] = df['Date'].dt.day
"""
#change the date format to int64 for the calculations
df['Date'] = pd.to_datetime(df.Date).astype('int64')
df['Open'] = df['Open'].astype('float32')
df['Close'] = df['Close'].astype('float32')


#before scaling
#X = df.drop(columns=['Date_c','Close', 'High','Low'])
X = df.drop(columns=['Date_c','Close'])
Y = df[['Close']]

train_num=int(df.shape[0]*0.9)

X_train = X.values[0:train_num,:]
X_test = X.values[train_num:,:]

Y_train = Y.values[0:train_num,]
Y_test = Y.values[train_num:,]


#scaling the values of train and test
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.fit_transform(X_test)

Y_train = traget_scaler.fit_transform(Y_train)
Y_test = traget_scaler.fit_transform(Y_test)

model = keras.models.Sequential()
model.add(keras.layers.Dense(20, activation='relu', input_shape=(2,)))
model.add(keras.layers.Dense(18, activation='relu'))
model.add(keras.layers.Dense(21, activation='relu'))
model.add(keras.layers.Dense(1))

print(model.summary())

#scale the inputs before fitting the model


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mse','mae', 'mape', 'cosine_proximity'])
history = model.fit(X_train, Y_train, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

#test_data = np.array([2003,	854,	1710,	2,	1,	3]) #o/p: [[203744.11]]
#print(model.predict(test_data.reshape(1,6), batch_size=1)) #[[213644.72]]


score = model.evaluate(X_test, Y_test, verbose=0)
print("The model score is: ", score)

results_output=model.predict(X_test)

#errors = results_output-Y_test

#print("The errors are: ", errors)


#reverse the scaling on the result
results_output = traget_scaler.inverse_transform(results_output)
Y_test = traget_scaler.inverse_transform(Y_test)

"""
print("Y.values[train_num:,], results_output:")
print("**********************")
for i,j in zip(Y.values[train_num:,], results_output):
    print(i,j)
print("**********************")
"""

#change the date format back to %d-%b-Y from int64 for the plotting
df['Date'] = pd.to_datetime(df.Date)
df['Date'] = df['Date'].dt.strftime('%d-%b-%Y')

plt.figure()
plt.suptitle('DNN Predictions', fontsize=16)

#plt.plot(df['Date'].values[0:train_num][:,None],Y_train) #plots only the training part
plt.plot(df['Date_c'][:,None],Y) #plot the whole data set
plt.plot(df['Date_c'].values[train_num:][:,None],results_output)

#plt.show()

#print out the metrics
from sklearn.metrics import r2_score
score = r2_score(results_output, Y_test)
print('R-squared score for the test set:', round(score,4))

for i in history.history:
    print("\n\n", "-"*10)
    print(i,": ", "\nmax=",max(history.history[i]),"\nmin=", min(history.history[i]))
    print("-="*10)


#print out the graphs for the metrics
#plt.figure()
fig, axs = plt.subplots(2, 3)
plt.suptitle('Metrics DNN vs Epoches', fontsize=16)
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


#plot the errors:
plt.figure()
plt.suptitle("Errors diff plot: Y_test-results_output")
errors = Y_test-results_output
plt.plot(errors)

"""
f = open("DNN_pred_values.txt", "a")
f.write("For DNN prediction (y_test, results, errors): \n")
f.write("Y_test:")
f.write(str(Y_test))
f.write("\n\n")

f.write("results_output:")
f.write(str(results_output))
f.write("\n\n")

f.write("errors:")
f.write(str(errors))
f.write("\n\n")

f.close()
"""

#fig.show()
plt.show()
