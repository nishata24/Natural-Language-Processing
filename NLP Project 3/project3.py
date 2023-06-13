# NLP Project 3 (Default Project)
# Yuri Hu and Nishat Ahmed 

# To use this program, run the following command:
# python project3.py <Training Labels File> <Testing File> <Optional Arguments>

import tensorflow as tf
import numpy as np
import argparse
import os 
import math 
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

vocab_size = 4000
embedding_dim = 64
max_length = 1000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# Function to create lists for the training and testing data
def create_data(train_file, test_file):
    # Get lines in training data
    train_data = open(train_file, 'r').read().splitlines()
    articles = []
    labels = []

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Process training data
    for line in train_data:
        # Split line and append to labels list
        label = line.split()
        labels.append(label[1])
        
        with open(label[0], 'r') as f:
            # Read file and append to articles list
            article = f.read().replace('\n','').replace('\t','').replace('  ', ' ').replace('``', '').replace("''", '').replace('"', '').strip()
            
            for item in stop_words:
                tok = ' ' + item + ' '
                article = article.replace(tok, ' ')
                article = article.replace(' ', ' ')
            
            articles.append(article)

    # Process testing data
    test_data = open(test_file, 'r').read().splitlines()
    test_articles = []
    
    for line in test_data:
        with open(line, 'r') as f:
            # Read file and append to test_articles list
            article = f.read().replace('\n','').replace('\t','').replace('  ', ' ').replace('``', '').replace("''", '').replace('"', '').strip()
            
            for item in stop_words:
                tok = ' ' + item + ' '
                article = article.replace(tok, ' ')
                article = article.replace(' ', ' ')
            
            test_articles.append(article)

    return articles, labels, test_articles

# Function to tokenize and pad the training and testing data
def train_and_cat(articles, labels, test_articles, vocab_size, embedding_dim, max_length, trunc_type, padding_type, oov_tok, training_portion, learning_rate, batch_size):
    train_size = int(len(articles) * training_portion)
    train_articles, train_labels = articles[:train_size], labels[:train_size]
    tuning_articles, tuning_labels = articles[train_size:], labels[train_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_articles)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    tuning_sequences = tokenizer.texts_to_sequences(tuning_articles)
    tuning_padded = pad_sequences(tuning_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    output = {i: j for j, i in label_tokenizer.word_index.items()}

    training_label_seq = label_tokenizer.texts_to_sequences(train_labels)
    tuning_label_seq = label_tokenizer.texts_to_sequences(tuning_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    num_epochs = 35

    model.fit(train_padded, np.array(training_label_seq), epochs=num_epochs,
              validation_data=(tuning_padded, np.array(tuning_label_seq)), verbose=2, batch_size=batch_size)

    predictions = []
    test_sequences = tokenizer.texts_to_sequences(test_articles)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred = model.predict(test_padded, batch_size=batch_size)

    for i in range(len(test_articles)):
        index = np.argmax(pred[i])
        predictions.append(output[index])

    return predictions

# Function to write predictions to file
def write_predictions(predictions, test_file):
    output_name = input("Enter the name of the desired text categorized predictions file: ")

    with open(output_name, 'w') as output_file, open(test_file, 'r') as test_file:
        test_file_lines = test_file.read().splitlines()

        for i, line in enumerate(test_file_lines):
            cat = predictions[i]
            output_line = f"{line} {cat.capitalize()}\n"
            output_file.write(output_line)

    return

parser = argparse.ArgumentParser(description='Train and categorize documents')
parser.add_argument('train_input', type=str, help='Name of the labeled list of training documents')
parser.add_argument('test_input', type=str, help='Name of the list of the testing documents to be categorized')
parser.add_argument('--vocab_size', type=int, default=10000, help='Number of words to keep in the vocabulary')
parser.add_argument('--embedding_dim', type=int, default=32, help='Dimensionality of the embedding')
parser.add_argument('--max_length', type=int, default=100, help='Maximum length of a document')
parser.add_argument('--trunc_type', type=str, default='post', help='Truncation type for padding (default: post)')
parser.add_argument('--padding_type', type=str, default='post', help='Padding type (default: post)')
parser.add_argument('--oov_tok', type=str, default='<OOV>', help='Token for out-of-vocabulary words')
parser.add_argument('--training_portion', type=float, default=0.8, help='Proportion of documents to use for training (default: 0.8)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 50)')
args = parser.parse_args()

articles, labels, test_articles = create_data(args.train_input, args.test_input)
predictions = train_and_cat(articles, labels, test_articles, args.vocab_size, args.embedding_dim, args.max_length, args.trunc_type, args.padding_type, args.oov_tok, args.training_portion, args.learning_rate, args.batch_size)
write_predictions(predictions, args.test_input)


