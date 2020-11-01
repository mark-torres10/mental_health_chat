####### 

# Implementing a simple Seq2Seq model, in Keras, for speech creation

#######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers , activations , models , preprocessing, utils
from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
tf.random.set_seed(1)
from google.colab import drive
import time
import os
import gdown
import sys

# set pandas viewing options
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option("max_colwidth", None)
#pd.reset_option("max_colwidth")

# the source of our data is: https://github.com/nbertagnolli/counsel-chat

# load our weights
gdown.download('https://drive.google.com/uc?export=download&id=1212a1k_GxYbvh-CKF6m9oLwAb_ZBRkZe','chatbot_seq2seq_v3.h5',True);

# Run this cell to download our dataset
chat_data = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")

# get our X (patient questions) and y (therapist answer) columns
X = chat_data["questionText"]
y = chat_data["answerText"]

# text preprocessing
def preprocess_text(phrase): 
  phrase = re.sub(r"\xa0", "", phrase) # removes "\xa0"
  phrase = re.sub(r"\n", "", phrase) # removes "\n"
  phrase = re.sub("[.]{1,}", ".", phrase) # removes duplicate "."s
  phrase = re.sub("[ ]{1,}", " ", phrase) # removes duplicate spaces

  return phrase

X = X.apply(preprocess_text)
y = y.apply(preprocess_text)

# split up questions and answers into Q&A pairs:
question_answer_pairs = []

MAX_LENGTH = 100 # the maximum length for our sequences

for (question, answer) in zip(X, y):
  question = preprocess_text(question) 
  answer = preprocess_text(answer)

  # split up question and answer into their constituent sentences

  question_arr = question.split(".")
  answer_arr = answer.split(".")

  # get the maximum number of question/answer pairs we can form,
  # which will be the shorter of len(question_arr) and len(answer_arr)

  max_sentences = min(len(question_arr), len(answer_arr))

  for i in range(max_sentences):
    q_a_pair = []

    # get maximum sentence length
    max_q_length = min(MAX_LENGTH, len(question_arr[i]))
    max_a_length = min(MAX_LENGTH, len(answer_arr[i]))

    # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
    question_to_append = question_arr[i][0:max_q_length]
    q_a_pair.append(question_to_append)

    answer_to_append = "<START> " + answer_arr[i][0:max_a_length] + " <END>"
    q_a_pair.append(answer_to_append)

    question_answer_pairs.append(q_a_pair)

# re-create questions, answers
questions = [arr[0] for arr in question_answer_pairs]
answers = [arr[1] for arr in question_answer_pairs]

target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex)
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# create encoder input data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
encoder_input_data = pad_sequences(tokenized_questions,
                                   maxlen=maxlen_questions,
                                   padding='post')

# create decoder input data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
decoder_input_data = pad_sequences(tokenized_answers,
                                   maxlen=maxlen_answers,
                                   padding='post')

# create decoder output data (note: this step might be particularly time and space intensive, since the resulting matrix is large)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_output_data = to_categorical(padded_answers, VOCAB_SIZE)

# create our Seq2Seq model:
enc_inputs = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inputs)
_, state_h, state_c = LSTM(200, return_state=True)(enc_embedding)
enc_states = [state_h, state_c]

dec_inputs = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inputs)
dec_lstm = LSTM(200, return_state=True, return_sequences=True)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(dec_outputs)

# compile our model
model = Model([enc_inputs, dec_inputs], output)

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

model.summary()

# train model
# model.fit([encoder_input_data, decoder_input_data], decoder_output_data)

# (use weights from previous training)
path_to_weight = "chatbot_seq2seq_v3.h5"

model.load_weights(path_to_weight)

# set up our evaluation step:
def make_inference_models():
    dec_state_input_h = Input(shape=(200,))
    dec_state_input_c = Input(shape=(200,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                             initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
        inputs=[dec_inputs] + dec_states_inputs,
        outputs=[dec_outputs] + dec_states)
    print('Inference decoder:')
    dec_model.summary()
    print('Inference encoder:')
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
    enc_model.summary()
    return enc_model, dec_model

def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                         maxlen=maxlen_questions,
                         padding='post')
    
enc_model, dec_model = make_inference_models()

# to test our chatbot:

#question = "Could you give me some advice?" # example question
question = sys.argv[1] # user input, as string

for _ in range(1):
    states_values = enc_model.predict(
        str_to_tokens(question))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq]
                                              + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' \
                or len(decoded_translation.split()) \
                > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)









