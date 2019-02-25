from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import argparse

# ArgumentParser
parser = argparse.ArgumentParser(description='The following  parameters can be assigned:')
# Parse dataset
parser.add_argument('-tok', '--tokenizer', type=str, required=True,
                    help='Path to the used tokenizer for text processing in training.')
parser.add_argument('-mod', '--model', type=str, required=True,
                    help='Path to the model file for the network architecture and its parameters')
parser.add_argument('-ml', '--maxlen', type=int, required=True,
                    help='Specifies the maximum input length for sentences. Taken from training log file,'
                         ' Parameters section > Length Sentences')
# parse to dict
argv = parser.parse_args()
param_dict = vars(argv)

# load the tokenizer and the model
with open(param_dict['tokenizer'], "rb") as f:
    tokenizer = pickle.load(f)

# load model
model = load_model(param_dict['model'])

# data to classify
input_texts = ["I hate this restaurant!", "I love these burgers!"]

# tokenize texts
sequences = tokenizer.texts_to_sequences(input_texts)
data = pad_sequences(sequences, maxlen=param_dict['maxlen'])

# get predictions for each of your new texts
predictions = model.predict(data)
print(np.argmax(a=predictions, axis=1) + 1)
