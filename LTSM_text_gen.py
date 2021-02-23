"""
Basic fixed form text generator which learns from text_data[line #][line words], adapted from twitter generator:
Based on: https://www.analyticsvidhya.com/blog/2020/10/elon-musk-ai-text-generator-with-lstms-in-tensorflow-2/
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
import os


# Function to clean the text
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    # text = text.replace('\%','')
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join(filter(lambda x:x[0]!="@", text.split()))
    return text


class TextGen:

    def __init__(self, training_data):

        self.data = training_data
        # self.num_lines = len(self.data)
        # self.max_line_length = max([len(self.data[_]) for _ in range(len(self.data))])

        # Updates internal vocabulary (of tokenizer) based on a list of texts, establishing the word-to-int mapping
        self.tokenizer = Tokenizer()  # note the default filters likely handle much of the clean text function
        self.tokenizer.fit_on_texts(self.data)
        self.total_words = len(self.tokenizer.word_index) + 1
        self.input_sequences = []
        self.max_sequence_length = 0
        self.encode_lines_as_int_sequences()  # updates input_sequences to be all padded subsequences of sequential words for every line

        self.xs, self.labels = self.input_sequences[:, :-1], self.input_sequences[:, -1]  # pairs of (sub)sequences with final element(s) excluded, given as label(s)
        self.ys = tf.keras.utils.to_categorical(self.labels, num_classes=self.total_words)
        self.model = self.build_model()

    def encode_lines_as_int_sequences(self):
        for line in self.data:
            token_list = self.tokenizer.texts_to_sequences([line])[0]  # Transforms each text in [line] to a sequence of integers.
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                self.input_sequences.append(n_gram_sequence)
        # End up with a series of num_words_on_line encoded sequences each 1 larger than the next
        # e.g. for 2 lines of 3, 4 words respectively: input_sequences = [[4], [4, 2], [4, 2, 1], [2], [2, 23], [2, 23, 9], [2, 23, 9, 12]]
        self.max_sequence_length = max([len(x) for x in self.input_sequences])
        self.input_sequences = np.array(pad_sequences(self.input_sequences, maxlen=self.max_sequence_length, padding='pre'))
        # makes all sub-sequences equal length by adding leading 0s, and adds a dimension to store all sequences; (results in square matrix)

    def build_model(self, verbose=False, compile=True):
        """ Fixed LTSM model instantiation with embedding layer with 80 embedding length """
        model = Sequential()
        # model.add(Embedding(self.total_words, 80, input_length=self.max_sequence_length-1, mask_zero=True))  # as we've padded with zeros in the sequences
        model.add(Embedding(self.total_words, 80, input_length=self.max_sequence_length-1))  # as we've padded with zeros in the sequences
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(50))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(Dense(self.total_words/20))
        model.add(Dense(self.total_words, activation='softmax'))
        if verbose:
            model.summary()
        if compile:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, epochs, verbosity=1, checkpoint_callback=None):
        if checkpoint_callback is None:
            self.model.fit(self.xs, self.ys, epochs=epochs, verbose=verbosity)
        else:
            self.model.fit(self.xs, self.ys, epochs=epochs, verbose=verbosity, callbacks=[checkpoint_callback])

    def generate_text(self, seed_text, num_words_to_generate):
        """Given the (presumably trained) state of the model, produces guesses at text completion given seed_text,
        num_words_generated long. Ex:

        """
        if isinstance(seed_text, list):  # quasi overload allows for ["lists of", "seed text", "strings"]
            predictions = []
            for sd_txt in seed_text:
                # print("\n", self.generate_text(sd_txt, num_words_to_generate), "\n")
                predictions.append(self.generate_text(sd_txt, num_words_to_generate))
            print("\n\nGenerated texts: [seed text | generated text]")
            for pred in predictions:
                print("\n", pred, "\n")
            return
        assert isinstance(seed_text, str), f"Seed text needs to be a string of list of strings (e.g. \"to be or not to be \") \n seed_text: {seed_text}"

        return_chirp = seed_text + " |"
        token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
        num_words_in_chirp = len(token_list) + num_words_to_generate - 1
        token_list = pad_sequences([token_list], maxlen=num_words_in_chirp, padding='pre')
        for _ in range(num_words_to_generate):
            token_list = self.tokenizer.texts_to_sequences([return_chirp])[0]
            token_list = pad_sequences([token_list], maxlen=num_words_in_chirp, padding='pre')
            predicted = np.argmax(self.model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            return_chirp += " " + output_word
        return return_chirp


