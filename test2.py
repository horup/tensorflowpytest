import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dense, Activation

input_shape=(1,)
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Dense(1))

model.compile(optimizer='sgd', loss='mean_squared_error')


x_train = [1,2,3,1,2,3,1,2,3]
y_train = [10,20,30,10,20,30,10,20,30]

model.fit(x_train, y_train, epochs=20, batch_size=3)


res = model.predict([10000])

print(res)