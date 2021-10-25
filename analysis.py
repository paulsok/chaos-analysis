#!opt/anaconda3/bin/python python
# -*- coding: utf-8 -*-
import heapq
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


systems = {0: 'cubic', 1: 'cyclostationary', 2: 'freitas',
           3: 'gen. henon', 4: 'gopy', 5: 'granulocyte',
           6: 'henon', 7: 'ikeda', 8: 'izhikevich', 9: 'logistic',
           10: 'lorenz', 11: 'mackey_glass', 12: 'sine',
           13: 'randomwalk_arma', 14: 'randomwalk', 15: 'lorenz'
           }

model_mlp = keras.models.load_model('/models/mlp.h5')
model_shallow = keras.models.load_model('/models/shnet.h5')
model_deep = keras.models.load_model('/models/deep.h5')

data = np.stack((np.loadtxt('/time series/777_T_Bz=-35A_Bx=-0.08A_12K.txt'),
                 np.loadtxt('/time series/777_T_Bz=-5A_Bx=0.08A_12K.txt'),
                 np.loadtxt('/time series/777_T_Bz=5A_Bx=0.08A_5K.txt')))

# scale the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# classification with trained networks
X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
predictions_mlp = model_mlp.predict(X)

for i in range(data.shape[0]):
    temp_ts = predictions_mlp[i]
    max_idx = heapq.nlargest(3, range(len(temp_ts)), temp_ts.take)
    print(f'File #{i}: {temp_ts[max_idx[0]]} - {temp_ts[max_idx[1]]} - {temp_ts[max_idx[2]]}')
    print(systems.get(max_idx[0]), '-', systems.get(max_idx[1]), '-', systems.get(max_idx[2]))
    print('=*' * 50)
