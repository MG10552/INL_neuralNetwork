
# coding: utf-8

# In[1]:


import numpy as np

def output_function(sum_of_inputs):
    if sum_of_inputs > 0:
        return 1
    else:
        return 0

BIAS=1
input=np.array([BIAS, 1, 0])
weights = np.array([-30,20, 20])

result=output_function(np.dot(input, weights))
print(result)


# In[2]:


import numpy as np
import keras
from keras.models import Sequential, Input
from keras.layers import Activation, Dense
from keras.models import Model
          
input=np.array([[ 0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]])
          
print(input)

output=np.array([[0.],
                 [0.],
                 [0.],
                 [1.]])

print(output)


inputs = Input(shape=(2,))
predictions = Dense(1, activation='sigmoid', use_bias=False)(inputs)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input, output, epochs=20)

print("\n\nPrediction\n\n")
print(model.predict(np.array([[1., 1.],
                        [0., 0.]])))


# In[3]:


import numpy as np
import keras
from keras.models import Sequential, Input
from keras.layers import Activation, Dense

input=np.array([[ 0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]])
          
print(input)

output=np.array([[0.],
                 [0.],
                 [0.],
                 [1.]])

print(output)

model2 = Sequential()
model2.add(Dense(50, input_dim=2))
model2.add(Activation('tanh'))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

model2.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
model2.fit(input, output, epochs=20)


print("\n\nPrediction\n\n")
model2.predict(np.array([[1., 1.],
                        [0., 0.]]))


# In[6]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import numpy as np
   
X=np.array([[0. ],
            [0.1],
            [0.2],
            [0.3],
            [0.4]])

y=np.array([[0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5]])

valX = np.array([[0.5],
                 [0.6],
                 [0.7],
                 [0.8],
                 [0.9]])

valY = np.array([[0.6],
                 [0.7],
                 [0.8],
                 [0.9],
                 [1.]])


#tutaj należy zdefiniować model
#zwykłe warstwy - 10 neuronów wejściowych i jeden wyjściowy
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam')
#model2.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(X, y, epochs=20)

history = model.fit(X, y, epochs=40, validation_data=(valX, valY), shuffle=False)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# In[12]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
import numpy as np
 
X=np.array([[0. ],
            [0.1],
            [0.2],
            [0.3],
            [0.4]])

X = X.reshape((5, 1, 1))

y=np.array([[0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5]])

valX = np.array([[0.5],
                 [0.6],
                 [0.7],
                 [0.8],
                 [0.9]])

valX = valX.reshape((5, 1, 1))

valY = np.array([[0.6],
                 [0.7],
                 [0.8],
                 [0.9],
                 [1.]])


#tutaj należy zdefiniować model LSTM 10 neuronów wejściowych
#i jeden wyjściowy - uwaga trzeba zdefiniować argument input_size
input_size=10
inputs = Input(shape=(1,1))
z = LSTM(input_size)(inputs)
predictions=Dense(1, activation='sigmoid')(z)
model = Model(inputs=inputs,output=predictions)



#model = Sequential()
#model.add(LSTM(10, input_dim=1))
#model.add(Activation('tanh'))
#model.add(LSTM(1))
#model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam')
#model2.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(X, y, epochs=20)


model.compile(loss='mse', optimizer='adam')

history = model.fit(X, y, epochs=1200, validation_data=(valX, valY), shuffle=False)

pyplot.plot(history.history['loss'][500:])
pyplot.plot(history.history['val_loss'][500:])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# In[39]:


import numpy as np
from gensim import models
import keras
from keras.models import Sequential, Input
from keras.layers import Activation, Dense
from gensim.models.word2vec import *
import subprocess
import csv
import sklearn

output_file = 'C:/Users/s10552/Desktop/INL/Z2/frazy_ML.txt'
input = []
output = []
#subprocess.call(concraft_command2.format(input_file, output_file), shell=True)
with open(output_file, "r", encoding="utf-8") as f:
   # for line in f:
    #    print(line)
    reader = csv.reader(f, dialect="excel", delimiter="\t")
    for row in reader:
        adj, noun, typ=row
        if adj != 'adj':
            try:
                input.append(np.append(wmodel[adj],wmodel[noun]))
                if type == "L":
                    output.append([0])
                else: 
                    output.append([1])
            except:
                pass

print(len(input))            
X=np.array(input)

print(X.shape)
Y=np.array(output)
print(Y.shape)
#print(X)

model2 = Sequential()
model2.add(Dense(10, input_dim=200))
model2.add(Activation('tanh'))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

model2.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
model2.fit(X, Y, epochs=20)

#pred = cross_val_predict(estimator=crf, X=X, y=Y,cv=10)
#report = classification_report()

#print("\n\nPrediction\n\n")
#model2.predict(np.array(X(0:200)))
report = classification_report()
print('complete')



#vector_size=100
#wmodel = Word2Vec.load('C:/Users/s10552/Desktop/INL/Z2/nkjp+wiki-forms-all-'+str(vector_size)+'-skipg-ns')
#print('done')
#print(wmodel["kot"])

