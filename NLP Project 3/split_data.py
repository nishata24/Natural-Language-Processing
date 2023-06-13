# NLP Project 3 (Default Project)
# Yuri Hu and Nishat Ahmed 

import sys
import os
import math
import random
import nltk

import sys
import os
import nltk
import math

# check if enough arguments are given by user
if len(sys.argv) != 7:
    print("Use in the following manner: python split_data.py <Training Labels File> <Output Training File> <Output Validation File> <Output Test File> <percentage_train (in decimal)> <percentage_val (in decimal)>")
    sys.exit(-1)

# Get the filename of the training labels file and the percentage of data to keep in training set
training_labels_file = sys.argv[1]
dir = os.path.dirname(os.path.realpath(training_labels_file))
percentage_train = float(sys.argv[5])
percentage_val = float(sys.argv[6])

# Open the training labels file to read
with open(training_labels_file, 'r') as fid:
    type_counts = {}
    for line in fid:
        cur_file, cur_label = nltk.word_tokenize(line)[:2]
        type_counts[cur_label] = type_counts.get(cur_label, 0) + 1

# Count the number of items of each type in the training labels file
for t in type_counts:
    type_counts[t] = math.floor(type_counts[t] * percentage_train)

with open(training_labels_file, 'r') as fid_i, \
     open(sys.argv[2], 'w') as fid_training, \
     open(sys.argv[3], 'w') as fid_val, \
     open(sys.argv[4], 'w') as fid_test, \
     open(sys.argv[3] + "Labels", 'w') as fid_val_answers, \
     open(sys.argv[4] + "Labels", 'w') as fid_test_answers:
    for line in fid_i:
        cur_file, cur_label = nltk.word_tokenize(line)[:2]
        if type_counts[cur_label] > 0:
            fid_training.write(line)
            type_counts[cur_label] -= 1
        elif type_counts[cur_label] <= 0:
            # create validation set
            if random.random() < percentage_val/(1-percentage_train):
                fid_val.write(cur_file + "\n")
                fid_val_answers.write(line)
            else:
                # create test set
                fid_test.write(cur_file + "\n")
                fid_test_answers.write(line)