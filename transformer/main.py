# install packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
tf.random.set_seed(1)
import os

import preprocess_data
from preprocess_data import preprocess_text
import transformer
from transformer import scaled_dot_product_attention, MultiHeadAttention, create_padding_mask, create_look_ahead_mask, 
PositionalEncoding, encoder_layer, encoder, decoder_layer, decoder, Transformer, loss_function


# set pandas viewing options
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option("max_colwidth", None)
#pd.reset_option("max_colwidth")

# the source of our data is: https://github.com/nbertagnolli/counsel-chat

# load pretrained weights:
import gdown 
gdown.download('https://drive.google.com/uc?export=download&id=1rR0HAOKgs0yGAyZwqeJkX1U3W8234BgR','chatbot_transformer_v4.h5',True)

# load in our data with this code chunk: 
chat_data = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")

# create X, y variables
X = chat_data["questionText"]
y = chat_data["answerText"]

# run cleaning function
X = X.apply(preprocess_text)
y = y.apply(preprocess_text)

# get our question + answer pairs
question_answer_pairs = create_q_a_pairs(X, y)

# tokenize our data

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    [arr[0] + arr[1] for arr in question_answer_pairs], target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2


# maximum sentence length
MAX_LENGTH = 100 # chosen arbitrarily

# get questions, answers (tokenized, filtered, padded version)
questions, answers = tokenize_and_filter([arr[0] for arr in question_answer_pairs], 
                                         [arr[1] for arr in question_answer_pairs])


print('Vocab size: {}'.format(VOCAB_SIZE))
print('Number of samples: {}'.format(len(questions)))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)

# train NN
tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

model = Transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

# Compile our model
learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
	# ensure labels have shape (batch_size, MAX_LENGTH - 1)
	y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
	return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# train model
EPOCHS = 20
#model.load_weights(checkpoint_path)
#model.fit(dataset, epochs=EPOCHS)
# use pretrained version
model.load_weights("chatbot_transformer_v4.h5")

# set up evaluation of our model:
def evaluate(sentence):
  sentence = preprocess_text(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence

sentence = "What will make me happy?" #@param {type:"string"}
print("--------------------")
#output = predict(sentence)
predict(sentence)
print("--------------------")

