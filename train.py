from tqdm import tqdm

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

from models import standard_model, extended_standard_model

import timeit
import time
import json
import numpy as np
import pickle
import random as rd
import argparse


def check_equal_ivo(lst):
    """
        Check if all elements in list are equal.
        :param lst: list to check
        :return: true, if all elements are equal, false if not
    """
    return not lst or lst.count(lst[0]) == len(lst)


def line_counter(file):
    """
        Calculates the number of lines of given file.
        :param file: Path to file
        :return: Number of lines of the given file
    """
    return sum(1 for i in open(file, mode="r", encoding="utf-8"))


def extract_subset(data_list, label_list, bin_limit=None, fraction=None, mode='abs_max', recover_unextracted=False,
                   shuffle_input=True):
    """
        Extract a subset from given set
        :param data_list: list with samples
        :param label_list: list with labels
        :param bin_limit: max elements per category, only used when mode == 'abs_limit'
        :param fraction: extracts a fraction of the given dataset, only used when mode == 'fraction'
        :param mode: defines how the subset is extracted, choices are 'abs_limit', 'abs_max' and 'fraction'
        :param recover_unextracted: If true, elements which not move to subset still be available and will not be
            deleted, otherwise deleted
        :param shuffle_input: if true, input data gets shuffled before extraction
        :return: extracted subsets and their length, optionally the recovered unextracted samples
    """

    # shuffle input
    if shuffle_input:
        to_shuffle = list(zip(data_list, label_list))
        rd.shuffle(to_shuffle)
        data_list, label_list = zip(*to_shuffle)
        data_list, label_list = list(data_list), list(label_list)
        del to_shuffle
    sub_texts = []
    sub_labels = []
    if recover_unextracted:
        recover_texts = []
        recover_labels = []

    # calculate category limit
    if mode == 'abs_limit':
        limit = bin_limit
    elif mode == 'abs_max':
        limit = min(list(Counter(label_list).values()))
    elif mode == 'fraction':
        limit = int(np.rint((len(data_list) // 5) * fraction))
    else:
        raise RuntimeError('Wrong mode given, choose between: \'abs_limit\', \'abs_max\' and \'fraction\'.')

    # extract subsets
    star_counts = [0, 0, 0, 0, 0]
    entries = len(data_list)
    for i in range(entries):
        if (all(w == limit for w in star_counts)) and not recover_unextracted:
            break
        if star_counts[label_list[-1]] < limit:
            sub_texts.append(data_list.pop())
            star_counts[label_list[-1]] += 1
            sub_labels.append(label_list.pop())
        else:
            if recover_unextracted:
                recover_texts.append(data_list.pop())
                recover_labels.append(label_list.pop())
            else:
                data_list.pop()
                label_list.pop()

    # check if all categories have same size
    assert check_equal_ivo(star_counts), 'Not enough data available to generate even datasets for' \
                                         ' all ratings -> {}!'.format(star_counts) + ' Increase over all dataset' \
                                                                                     ' size or decrease category limit!'

    if recover_unextracted:
        return sub_texts, sub_labels, star_counts[0], recover_texts, recover_labels
    return sub_texts, sub_labels, star_counts[0]


# ArgumentParser
parser = argparse.ArgumentParser(description='The following  parameters can be assigned:')

# Parse dataset
parser.add_argument('-jid', '--job_id', type=int, required=True,
                    help='A specific job id for every job.')
parser.add_argument('-al', '--alphabet', type=int, default=30000,
                    help='Specifies the alphabet size.')
parser.add_argument('-do', '--drop_rate', type=float, default=0.2,
                    help='Specifies the dropout rate.')
parser.add_argument('-l2', '--l2_reg', type=float, default=0.01,
                    help='Specifies the value for l2 regularization.')
parser.add_argument('-max_n_samples', '--maximum_number_of_samples', type=int, default=0,
                    help='Specifies how much samples are used for training (crop dataset). '
                         'Set to 0 if no cropping should be done.')
parser.add_argument('-mod', '--model', type=str, default='standard', choices=['standard', 'extended', 'w_extended'],
                    help='Specifies the network model architecture. See model.py for more information!')
parser.add_argument('-eps', '--epochs', type=int, default=200,
                    help='Specifies for how much epochs the training is applied.')
parser.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='Specifies the batch size for training and evaluation.')
parser.add_argument('-ds', '--dataset', type=str, required=True,
                    help='Specifies the path to the dataset, especially to the review.json file.')
                    
# parse to dict
argv = parser.parse_args()
param_dict = vars(argv)

# start runtime timer and get actual systemtime
start = timeit.default_timer()
act_time = time.strftime("%Y%m%d-%H%M%S")

# set numpy generator seed to reproduce results
np.random.seed(1337)

# set random seed to reproduce results
rd.seed(1337)

# Load the reviews and parse JSON
print('Reading json file ...')
if param_dict['maximum_number_of_samples'] == 0:
    param_dict['maximum_number_of_samples'] = np.inf
runner = 0
num_lines = line_counter(param_dict['dataset'])
if param_dict['maximum_number_of_samples'] < num_lines:
    num_lines = param_dict['maximum_number_of_samples']
reviews = []
with open(param_dict['dataset'], mode="r", encoding="utf-8") as json_file:
    for line in tqdm(json_file, total=num_lines):
        if runner == param_dict['maximum_number_of_samples']:
            break
        reviews.append(json.loads(line))
        runner += 1
print('Sucessfully loaded!\n')

# Get a balanced sample of positive and negative reviews
texts = [review['text'] for review in reviews]

# Convert our 5 classes into 2 (negative or positive)
stars = [(int(np.rint(review['stars'])) - 1) for review in reviews]

# free mem
del reviews

print('Extract balanced subsets ...')
train_texts, train_labels, train_set_size = extract_subset(texts, stars)
print('Sucessfully extracted!\n')

# free mem
del texts, stars

print('Create test set ...')
test_texts, test_labels, test_set_size, train_texts, train_labels = extract_subset(train_texts, train_labels,
                                                                                   fraction=0.2, mode='fraction',
                                                                                   recover_unextracted=True)
print('Sucessfully created!\n')

print('Create val set ...')
val_texts, val_labels, val_set_size, train_texts, train_labels = extract_subset(train_texts, train_labels,
                                                                                fraction=0.2, mode='fraction',
                                                                                recover_unextracted=True)
print('Sucessfully created!\n')

print('Generate Tokens ...')
tokenizer = Tokenizer(num_words=param_dict['alphabet'])
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
length_sen = 0
for se in sequences:
    if len(se) > length_sen:
        length_sen = len(se)
length_sen += (100 - (length_sen % 100))
while length_sen < 1000:
    length_sen += (100 - (length_sen % 100))
data = pad_sequences(sequences, maxlen=length_sen)
param_dict['length_sen'] = length_sen
val_sequences = tokenizer.texts_to_sequences(val_texts)
val_data = pad_sequences(val_sequences, maxlen=length_sen)
print('Sucessfully!\n')

# define model
if param_dict['model'] == 'standard':
    model = standard_model(param_dict)
elif param_dict['model'] == 'extended':
    model = extended_standard_model(param_dict)
elif param_dict['model'] == 'w_extended':
    model = extended_standard_model(param_dict)

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train model ...')
# define callbacks
callback = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(factor=0.5)]
# train model
history = model.fit(data, to_categorical(y=np.array(train_labels), num_classes=5),
                    validation_data=(val_data, to_categorical(y=np.array(val_labels))),
                    epochs=param_dict['epochs'], batch_size=param_dict['batch_size'], callbacks=callback)
print('Sucessfully trained!\n')

with open('training_log_' + act_time + '_' + str(param_dict['job_id']) + '.txt', 'w+') as log_writer:
    loss_string = (' ' * 4) + 'Model loss:'
    acc_string = (' ' * 4) + 'Model Accuracy:'
    val_loss_string = (' ' * 4) + 'Model validation loss:'
    val_acc_string = (' ' * 4) + 'Model validation accuracy:'
    log_writer.write('Training summary:\n')
    spacer = ''
    for idx, ep in enumerate(history.epoch):
        if idx != 0:
            spacer = (' ' * 15)
        loss_string += spacer + ' <Epoch ' + str(ep) + ': ' + str(history.history['loss'][ep]) + '>\n'
        if idx != 0:
            spacer = (' ' * 19)
        acc_string += spacer + ' <Epoch ' + str(ep) + ': ' + str(history.history['acc'][ep]) + '>\n'
        if idx != 0:
            spacer = (' ' * 26)
        val_loss_string += spacer + ' <Epoch ' + str(ep) + ': ' + str(history.history['val_loss'][ep]) + '>\n'
        if idx != 0:
            spacer = (' ' * 30)
        val_acc_string += spacer + ' <Epoch ' + str(ep) + ': ' + str(history.history['val_acc'][ep]) + '>\n'

    log_writer.write(loss_string + '\n')
    log_writer.write(acc_string + '\n')
    log_writer.write(val_loss_string + '\n')
    log_writer.write(val_acc_string + '\n')
    log_writer.write('Parameters:\n')

    for k, v in history.params.items():
        log_writer.write((' ' * 4) + str(k) + ': ' + str(v) + '\n')
        
    log_writer.write((' ' * 4) + 'Actual time: ' + str(act_time) + '\n')
    log_writer.write((' ' * 4) + 'Train Set Size per category: ' + str(train_set_size) + '\n')
    log_writer.write((' ' * 4) + 'Model architecture: ' + str(param_dict['model']) + '\n')
    log_writer.write((' ' * 4) + 'Alphabet: ' + str(param_dict['alphabet']) + '\n')
    log_writer.write((' ' * 4) + 'Length Sentences: ' + str(param_dict['length_sen']) + '\n')
    log_writer.write((' ' * 4) + 'Dropout Rate: ' + str(param_dict['drop_rate']) + '\n')
    log_writer.write((' ' * 4) + 'L2 Regularization: ' + str(param_dict['l2_reg']) + '\n')
    log_writer.write((' ' * 4) + 'Epochs: ' + str(param_dict['epochs']) + '\n')
    log_writer.write((' ' * 4) + 'Batch size: ' + str(param_dict['batch_size']) + '\n\n')
    log_writer.write('Model architecture:\n')
    model.summary(print_fn=lambda x: log_writer.write((' ' * 4) + x + '\n'))

# Model evaluation with testing data
print('Evaluate model ...')
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=param_dict['length_sen'])
eval_ob = model.evaluate(x=test_data, y=to_categorical(y=np.array(test_labels), num_classes=5),
                         batch_size=param_dict['batch_size'])
with open('training_log_' + act_time + '_' + str(param_dict['job_id']) + '.txt', 'a') as log_writer:
    log_writer.write('\n')
    log_writer.write('Evaluation on test set:\n')
    log_writer.write((' ' * 4) + 'Loss on test set: ' + str(eval_ob[0]) + '\n')
    log_writer.write((' ' * 4) + 'Accuracy on test set: ' + str(eval_ob[1]))
print('Sucessfully!\n')

# save the tokenizer and model
print('Save checkpoint ...')
with open("keras_tokenizer_" + act_time + '_' + str(param_dict['job_id']) + ".pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model.save("yelp_sentiment_model_" + act_time + '_' + str(param_dict['job_id']) + ".hdf5")
print('Sucessfully saved!\n')

stop = timeit.default_timer()
print('Runtime: {:.3f}'.format(stop - start) + 's')
