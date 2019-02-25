from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l2 as l2_reg


def standard_model(param_dict):
    """
        Standard model for text processing
    """
    model = Sequential()
    model.add(Embedding(input_dim=param_dict['alphabet'], output_dim=1024, input_length=param_dict['length_sen']))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=64, kernel_size=20, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=128, kernel_size=10, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=256, kernel_size=5, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], return_sequences=True, kernel_regularizer=l2_reg(param_dict['l2_reg']),
                   recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], kernel_regularizer=l2_reg(param_dict['l2_reg']), recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(Dense(256, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Dense(5, activation='softmax', kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    return model


def extended_standard_model(param_dict):
    """
        Extended standard model for text processing (deeper/more parameters)
    """
    model = Sequential()
    model.add(Embedding(input_dim=param_dict['alphabet'], output_dim=1024, input_length=param_dict['length_sen']))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=128, kernel_size=20, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=256, kernel_size=10, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=512, kernel_size=5, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=1024, kernel_size=3, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], return_sequences=True, kernel_regularizer=l2_reg(param_dict['l2_reg']),
                   recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], return_sequences=True, kernel_regularizer=l2_reg(param_dict['l2_reg']),
                   recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], kernel_regularizer=l2_reg(param_dict['l2_reg']), recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(Dense(512, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Dense(5, activation='softmax', kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    return model


def widely_extended_standard_model(param_dict):
    """
        More extended standard model for text processing (deeper/more parameters)
    """
    model = Sequential()
    model.add(Embedding(input_dim=param_dict['alphabet'], output_dim=1024, input_length=param_dict['length_sen']))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=128, kernel_size=25, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=256, kernel_size=20, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=512, kernel_size=10, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=1024, kernel_size=5, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Conv1D(filters=2048, kernel_size=3, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(LSTM(units=1024, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], return_sequences=True, kernel_regularizer=l2_reg(param_dict['l2_reg']),
                   recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(LSTM(units=1024, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], return_sequences=True, kernel_regularizer=l2_reg(param_dict['l2_reg']),
                   recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(LSTM(units=1024, activation='tanh', recurrent_activation='hard_sigmoid', dropout=param_dict['drop_rate'],
                   recurrent_dropout=param_dict['drop_rate'], kernel_regularizer=l2_reg(param_dict['l2_reg']), recurrent_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(Dense(1024, activation=None, kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=param_dict['drop_rate']))
    model.add(Dense(5, activation='softmax', kernel_regularizer=l2_reg(param_dict['l2_reg'])))
    return model
