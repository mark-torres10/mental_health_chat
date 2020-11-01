import re
import tensorflow_datasets as tfds
import tensorflow as tf

def preprocess_text(phrase): 

	"""
		Preprocess our text
	"""
	phrase = re.sub(r"\xa0", "", phrase) # removes "\xa0"
	phrase = re.sub(r"\n", "", phrase) # removes "\n"
	phrase = re.sub("[.]{1,}", ".", phrase) # removes duplicate "."s
	phrase = re.sub("[ ]{1,}", " ", phrase) # removes duplicate spaces
	return phrase

def create_q_a_pairs(X, y):

	"""
		Create Q&A pairs for our data
	"""

	# run this code chunk, to store all of our question/answer pairs
	question_answer_pairs = []

	# loop through each combination of question + answer
	for (question, answer) in zip(X, y):

	  # clean up text inputs

	  # example: 
	  # question = "I am not feeling well today. I feel sad."
	  # answer = "Tell me more about how you feel. What have you been up to today?"

	  question = preprocess_text(question) 
	  answer = preprocess_text(answer)

	  # split by .
	  # example
	  # question_arr = ["I am not feeling well today", "I feel sad"]
	  # answer_arr = ["Tell me more about how you feel", "What have you been up to?"]
	  question_arr = question.split(".")
	  answer_arr = answer.split(".")

	  # get the maximum length, which will be the shorter of the two
	  max_sentences = min(len(question_arr), len(answer_arr))

	  # for each combination of question + answer, pair them up
	  for i in range(max_sentences):

	    # set up Q/A pair
	    q_a_pair = []

	    # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
	    q_a_pair.append(question_arr[i])
	    q_a_pair.append(answer_arr[i])

	    # append to question_answer_pairs
	    question_answer_pairs.append(q_a_pair)

	return question_answer_pairs

def tokenize_and_filter(inputs, outputs):
  """
    Tokenize, filter, and pad our inputs and outputs
  """

  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    [arr[0] + arr[1] for arr in question_answer_pairs], target_vocab_size=2**13)

  # store results
  tokenized_inputs, tokenized_outputs = [], []

  # loop through inputs, outputs
  for (sentence1, sentence2) in zip(inputs, outputs):

    # tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    
    # check tokenized sentence max length
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)

  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen = MAX_LENGTH, padding = "post")
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen = MAX_LENGTH, padding = "post")
    
  return tokenized_inputs, tokenized_outputs

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	# example of how we can adjust learning rate
	#sample_learning_rate = CustomSchedule(d_model=128)

	#plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
	#plt.ylabel("Learning Rate")
	#plt.xlabel("Train Step")
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps**-1.5)

	return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)





