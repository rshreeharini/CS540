from flask import Flask
from flask import jsonify
from flask import request
import keras
import nltk
import random
import json
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
# from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from flask_cors import CORS,cross_origin
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
word_stemmer = LancasterStemmer()

from flask import Flask, render_template

app = Flask(__name__)
cors = CORS(app, resources={r"/cs": {"origins": "http://localhost:port"}})

def get_model():
    global model,graph,session
    graph = tf.get_default_graph()
    session = keras.backend.get_session()
    init = tf.global_variables_initializer()
    session.run(init)
    model = tf.keras.models.load_model('model_cs_cnn.h5')
    print("Model Loaded!!!")
    
def bag_of_words(user_query, vocabulary):
    bag = [0 for _ in range(len(vocabulary))]

    s_words = nltk.word_tokenize(user_query)
    s_words = [word_stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(vocabulary):
            if se == w:
                bag[i] = 1

    return numpy.array(bag)
def pre_process():
    with open("train.json",encoding="utf-8") as data:
        contents = json.load(data)
    try:
        with open("cache.pickle", "rb") as f:
            vocabulary, labels, training, output = pickle.load(f)
    except:
        vocabulary = []
        labels = []
        docs_x = []
        docs_y = []

        # Extract words from the patterns, extract labels (tags)

        for intent in data["intents"]:
            for pattern in intent["patterns"]:

                tokenized_words = nltk.word_tokenize(pattern) 
                vocabulary.extend(tokenized_words) 
                docs_x.append(tokenized_words)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        # Stem the words. This process will bring the words to its root (the main meaning)

        vocabulary = [word_stemmer.stem(w.lower()) for w in vocabulary if w != "?"]  #Stemming the words after converting them to lower case letters
        vocabulary = sorted(list(set(vocabulary)))

        labels = sorted(labels)

      

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag_of_words = []

            wrds = [word_stemmer.stem(w) for w in doc]

            for w in vocabulary:
                if w in wrds:
                    bag_of_words.append(1)
                else:
                    bag_of_words.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag_of_words)
    #         print(training)
            output.append(output_row)

      

        training = numpy.array(training)
        output = numpy.array(output)

        # Write the pre-processed data into the pickle file to use it subsequent time.
        with open("cache.pickle", "wb") as f:
            pickle.dump((vocabulary, labels, training, output), f)
    return training,output,vocabulary,labels,contents
print("LOADING KERAS MODEL")
get_model()

@app.route('/cs',methods=['GET','POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def cs():
    message=request.get_json(force=True)
    name=message['name']
    training,output,vocabulary,labels,contents=pre_process()
    with graph.as_default():
        set_session(session)
        prediction = model.predict([bag_of_words(name, vocabulary).reshape(1,304)])
    max_prob = numpy.argmax(prediction)
    if prediction[0][max_prob] > 0.5:
        tags = labels[int(max_prob)]
        responses = []
        for i in contents["intents"]:
            if i["tag"] == tags:
                responses = i["responses"]
        answer=random.choice(responses)
    else:
        answer="I didn't get that. Try again."
    response={
            'greeting':answer
        }

    
    
    return jsonify(response)
