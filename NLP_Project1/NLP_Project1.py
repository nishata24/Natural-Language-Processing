from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log
import string 

#allow user to input 2 files (files are assumed to be in the same diectory as the program)
file1 = input("Enter the train file name: ")
file2 = input("Enter the test file name: ")

#read training file names 
try: 
    file_train = open(file1, 'r')
except IOError:
    print("File could not be opened")
    exit()
# split lines in train file into a list
train_lines = file_train.read().splitlines()
# declare stemmer
stemmer = PorterStemmer()
# dictionary to count files per category
cat_count = {}
# dictionary to store word counts per category
word_cat = {}
# dictionary to store token counts per category
tok_cat = {}
for line in train_lines:
    # split line into a list 
    line_list = line.split()
    # read training file
    read_train = open(line_list[0], 'r')
    # tokenize words in training file
    train_tok = word_tokenize(read_train.read())
    # category of the file
    category = line_list[1]
    # keep count of categories
    if category in cat_count:
        cat_count[category] += 1
    else:
        cat_count[category] = 1
    # loop through words in training file
    for word in train_tok:
        # apply stemmer to word
        word = stemmer.stem(word)
        # count number of times word appears in category
        if (word, category) in word_cat:
            word_cat[(word, category)] += 1
        else:
            word_cat[(word, category)] = 1
        # count number of tokens in all categories
        if category in tok_cat:
            tok_cat[category] += 1
        else:
            tok_cat[category] = 1
#total number of files
total_files = sum(cat_count.values())
#store unique categories
categories_unique = tok_cat.keys()

# read test file names
try: 
    file_test = open(file2, 'r')
except IOError:
    print("File could not be opened")
    exit()
# split lines in test file into a list
test_lines = file_test.read().splitlines()
# list to store predictions
predictions = []
# loop through test files
for line in test_lines:
    # read test file
    read_test = open(line, 'r')
    # tokenize words in test file
    test_tok = word_tokenize(read_test.read())
    # dictionary to store probabilities of each category
    prob_cat = {}
    # find vocabulary size of test file
    for tok in test_tok:
        # apply stemmer to word
        tok = stemmer.stem(tok)
        if tok in list(string.punctuation):
            pass
        else:
            if tok in prob_cat:
                prob_cat[tok] += 1
            else:
                prob_cat[tok] = 1
    # vocabulary size
    vocab_size = len(prob_cat)
    # dictionary to store log probabilities of each category
    log_prob_cat = {}
    # loop through categories
    for category in categories_unique:
        # initialize log probability of category
        total_log_prob = 0
        # prior probability of category
        prior_prob = cat_count[category]/total_files
        # normalize coefficient for additive smoothing
        k = .057
        norm_coeff = tok_cat[category] + k*vocab_size
        # compute log probabilities of categories
        for (word, count) in prob_cat.items():
            if (word, category) in word_cat:
                count_word = word_cat[(word, category)] + k
            else:
                count_word = k
            total_log_prob += count*log(count_word/norm_coeff)
        log_prob_cat[category] = total_log_prob + log(prior_prob)
    # determine category using maximum likelihood estimation
    max_prob = max(log_prob_cat, key = log_prob_cat.get)
    # write each line to output file and append to predictions list
    predictions.append(line + " " + max_prob + "\n")
# allow user to input output file name
output_file = input("Enter the output file name: ")
# open output file
file_out = open(output_file, 'w')
# write predictions to output file
for prediction in predictions:
    file_out.write(prediction)
# close output file
file_out.close()
