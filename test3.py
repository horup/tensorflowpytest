import os
import numpy as np
from numpy.random import seed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dense, Activation


input_shape=(1,)
model = Sequential()
model.add(Dense(2, input_shape = (2,)))
model.add(Dense(2))
model.add(Dense(1))

model.compile(optimizer='Nadam', loss='mean_squared_error')

#x_train = np.array([[1,1], [1, 0.5], [1, 0.0], [1,0.75], [1,0.4], [1,0.55]])
#y_train = np.array([0, 1, 1, 0, 1, 0])

x_train = np.array([(1, 1), (1, 0.75), (1, 0.5), (1, 0.49), (1, 0.25), (1, 0.15), (1,0.0)])
y_train = np.array( [0,0,0,1, 1, 1, 1])

test = model.fit(x_train, y_train, epochs=1000)
print(model.weights)
print(model.predict( np.array([[1, 0.0]])))
print(model.predict( np.array([[1, 1.0]])))
print(model.predict( np.array([[1, 0.4]])))

#res = model.predict([10000])
#print(res)