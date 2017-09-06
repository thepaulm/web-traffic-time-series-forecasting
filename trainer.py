#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from timeit import default_timer as timer
import pickle


basepath = 'data'


class DataSet(object):
    def __init__(self, basepath=basepath):
        self.basepath = basepath
        self.data = pd.read_csv('{basepath}/train_1.csv'.format(basepath=basepath))

        self.index = None
        self.scaler = None
        self.x = None
        self.y = None

    def process_index(self, index, observations, predictions):
        '''
        Do preprocessing for a given index. Sets index, npdata, x, y.
        '''
        self.index = index

        npdata = self.data[index:index+1].fillna(0).as_matrix()
        npdata = npdata.squeeze()[1:].astype('float32')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        npdata = self.scaler.fit_transform(npdata.reshape(npdata.shape + (1,))).squeeze()

        outX = []
        outY = []
        for a in range(0, len(npdata) - (observations + predictions)):
            d = npdata[a:a + observations + predictions]
            outX.append(d[:observations])
            outY.append(d[observations:])
        outX = np.array(outX)
        outY = np.array(outY)
        outX = outX.reshape(outX.shape + (1,))
        outY = outY.reshape(outY.shape + (1,))
        self.x = outX
        self.y = outY.squeeze(axis=2)
        self.npdata = npdata

    def num_keys(self):
        return len(self.data)


def save_name(obs, pred, min, max, units, cells, lr, epochs):
    return "o%.4d_p%.4d_m%.4d_M%.4d_u%.3d_c%.3d_lr%e_e%.4d" % \
            (obs, pred, min, max, units, cells, lr, epochs)


class TrainContext(object):
    def __init__(self, model, scaler, history, train_time, sname):
        self.model = model
        self.scaler = scaler
        self.history = history
        self.train_time = train_time
        self.sname = sname

    def save(self):
        self.model.save_weights(self.sname + '_weights.h5')
        with open(self.sname + '.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            pickle.dump(self.history, f)
            pickle.dump(self.train_time, f)

    @staticmethod
    def load(obs, pred, min, max, units, cells, lr, epochs):
        ctx = TrainContext(None, None, None, None, None)
        sname = save_name(obs, pred, min, max, units, cells, lr, epochs)
        ctx.sname = sname
        ctx.model = make_model(obs, pred, units, cells)
        ctx.model.load_weights(sname + '_weights.h5')
        with open(sname + '.pkl', 'rb') as f:
            ctx.scaler = pickle.load(f)
            ctx.history = pickle.load(f)
            ctx.train_time = pickle.load(f)
        return ctx


def make_model(obs, pred, units, cells):
    model = Sequential()
    input_shape = (obs, ) + (1, )
    if cells == 1:
        model.add(LSTM(units, input_shape=input_shape))
    else:
        model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
        for _ in range(cells-2):
            model.add(LSTM(units, return_sequences=True))
        model.add(LSTM(units))

    model.add(Dense(pred))
    return model


def train_model(data, min=None, max=None, units=64, cells=1, lr=1e-3, epochs=32, verbose=False):
    '''create and fit the LSTM network'''

    if min is not None or max is not None:
        if min is None:
            min = 0
        if max is None:
            max = len(data.x)
        x = data.x[min:max]
        y = data.y[min:max]
    else:
        min = 0
        max = len(data.x)
        x = data.x
        y = data.y

    obs = x.shape[1]
    pred = y.shape[1]

    model = make_model(obs, pred, units, cells)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.optimizer.lr = lr

    start = timer()
    history = model.fit(x, y, epochs=epochs, batch_size=1, verbose=verbose)
    end = timer()
    return TrainContext(model, data.scaler, history.history, end-start,
                        save_name(obs, pred, min=min, max=max, units=units, cells=cells, lr=lr,
                                  epochs=epochs))


def main():
    pass

if __name__ == '__main__':
    main()