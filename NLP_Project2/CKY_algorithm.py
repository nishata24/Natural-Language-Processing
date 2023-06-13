# Nishat Ahmed 
# NLP Project 2

import string 
import sys 

# node class to store non-terminal data
class Node:
    def __init__(self, value, left, right = None):
        self.value = value
        self.left = left
        self.right = right

# function to print parse tree
def tree(node, level): 
    for i in range(0, level):
        sys.stdout.write("  ")
    if node.right == None:
        print("[" + node.value + " " + node.left + "]")
    else:
        print("[" + node.value + " ")
        tree(node.left, level+1)
        tree(node.right, level+1)
        for i in range(0, level):
            sys.stdout.write("  ")
        print("]")


#function to create a string of a list 
def tree_list(node):
	if node.right == None:
		string = "[" + node.value + " " + node.left + "]"
		return string
	else:
		string = "[" + node.value + " " + tree_list(node.left) + " " + tree_list(node.right) + "]"
		return string

filename = input("Name of file containing CNF grammar: ")
grammar_file = open(filename, "r")
# split grammar_file into lines
grammar = grammar_file.read().splitlines()
# split each line into a list of lists
rules = []
for line in grammar:
    line = line.split()
    symbols = []
    symbols.append(line[0])
    symbols.append(line[2])
    if(len(line) == 4):
        symbols.append(line[3])
    rules.append(symbols)
translator = str.maketrans('', '', string.punctuation)
# ask user if they textual parse trees to be displayed
display = input("Display parse trees? (y/n): ")
while(True):
    # read input from user 
    sentence = input("Enter a sentence to parse or type quit to exit: ")
    # remove punctuation and make sentence lowercase
    sentence = sentence.translate(translator).lower()
    # split sentence into words
    words = sentence.split()
    # handle quit case
    if sentence == "quit":
        print("ending program")
        exit(0)
    # get number of words in sentence
    word_count = len(words)
    # create empty table to be used for parsing
    table = [[[] for i in range(word_count - j)] for j in range(word_count)]
    # put nodes containing terminals in the first row of table
    index = 0
    for i in words:
        for rule in rules:
            if i == rule[1]:
                table[0][index].append(Node(rule[0], i))
        index += 1
    # CKY algorithm
    for i in range(2, word_count + 1):
        for j in range(0, (word_count - i) + 1):
            for k in range(1, i):
                right_part = i - k
                left_cell = table[k-1][j]
                right_cell = table[right_part-1][j+k]
                for rule in rules:
                    left_nodes = []
                    for h in left_cell:
                        if h.value == rule[1]:
                            left_nodes.append(h)
                    if left_nodes:
                        right_nodes = []
                        for h in right_cell:
                            if len(rule) == 3:
                                if h.value == rule[2]:
                                    right_nodes.append(h)
                        # create nodes which contain left_nodes and right_nodes as children
                        for left_node in left_nodes:
                            for right_node in right_nodes:
                                table[i-1][j].append(Node(rule[0], left_node, right_node))
    # print parse tree
    start_node = "S"
    nodes = []
    # read first column of table from bottom to top
    for node in table[-1][0]:
        if node.value == start_node:
            nodes.append(node)
    # if nodes list is not empty, valid parse tree
    if nodes:
        iteration = 1
        print("VALID SENTENCE")
        if display == "y":
            for n in nodes: 
                print("Valid Parse #" + str(iteration) + ": ")
                print(tree_list(n))
                tree(n, 0)
                iteration += 1
        if display == "n":
            for n in nodes:
                print("Valid Parse #" + str(iteration) + ": ")
                print(tree_list(n))
                iteration += 1
        print("Number of valid parses: " + str(iteration-1))
    else:
        print("No valid parse tree")


