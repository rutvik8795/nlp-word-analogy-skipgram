import os
import pickle
import numpy as np
import re
from scipy import spatial

model_path = './models/'
loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model3'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

f=open("./word_analogy_dev.txt", "r")
contents =f.read()
pairs = re.split('\n', contents )
contents = contents.replace('"','')
pairs = contents.split('\n')
word_pairs = [[]]
option_pairs = [[]]
for pair in pairs:
    words = pair.split('||')
    words[0] = words[0].split(",")
    word_pairs.append(words[0])
    words[1] = words[1].split(',')
    option_pairs.append(words[1])

diffWords = [[] for i in range(1,len(word_pairs))]
diffOptions = [[] for i in range(1, len(option_pairs))]
diffWordsIndex = 0;
diffOptionsIndex = 0;
for wordIdx in range(1,len(word_pairs)):
    for idx in range(len(word_pairs[wordIdx])):
        wordPair = word_pairs[wordIdx][idx].split(':')
        word_id1 = dictionary[wordPair[0]]
        word_id2 = dictionary[wordPair[1]]
        embedding1 = embeddings[word_id1]
        embedding2 = embeddings[word_id2]
        embedding3 = embedding1-embedding2
        diffWords[diffWordsIndex].append(embedding3)
    diffWordsIndex = diffWordsIndex+1

for optionIdx in range(1,len(option_pairs)):
    for optionsIdx in range(len(option_pairs[optionIdx])):
        optionPair = option_pairs[optionIdx][optionsIdx].split(':')
        option_id1 = dictionary[optionPair[0]]
        option_id2 = dictionary[optionPair[1]]
        embeddingOption1 = embeddings[option_id1]
        embeddingOption2 = embeddings[option_id2]
        embeddingOptions3  = embeddingOption1 - embeddingOption2
        diffOptions[diffOptionsIndex].append(embeddingOptions3)
    diffOptionsIndex = diffOptionsIndex+1

diffWordsMean = [[]for i in range(len(diffWords))]
for idx in range(len(diffWords)):
    sum = [0 for i in range(len(diffWords[0][0]))]
    for innerIdx in range(len(diffWords[idx])):
        sum = sum + diffWords[idx][innerIdx]
    sum = sum/3
    diffWordsMean[idx].append(sum)

diffOptionsCosine = [[]for i in range(len(diffOptions))]
for idx in range(len(diffOptions)):
    for innerIdx in range(len(diffOptions[idx])):
        diffOptionsCosine[idx].append(1 - spatial.distance.cosine(diffWordsMean[idx], diffOptions[idx][innerIdx]))


maxMinIndices = [[] for i in range(len(diffOptionsCosine))]
for idx in range(len(diffOptionsCosine)):
    min = diffOptionsCosine[idx][0]
    max = diffOptionsCosine[idx][0]
    minIdx = 0
    maxIdx = 0
    for innerIdx in range(len(diffOptionsCosine[idx])):
        if diffOptionsCosine[idx][innerIdx] > max:
            max = diffOptionsCosine[idx][innerIdx]
            maxIdx = innerIdx
        if diffOptionsCosine[idx][innerIdx] < min:
            min = diffOptionsCosine[idx][innerIdx]
            minIdx = innerIdx
    maxMinIndices[idx].append(minIdx)
    maxMinIndices[idx].append(maxIdx)

word_pairs.remove([])
option_pairs.remove([])

leastMost = [[] for i in range(len(option_pairs))]
for idx in range(len(maxMinIndices)):
    for innerIdx in range(len(maxMinIndices[idx])):
        leastMost[idx].append(option_pairs[idx][maxMinIndices[idx][innerIdx]])

#print(leastMost)

printStr =""
for idx in range(len(option_pairs)):
    str =""
    for innerIdx in range(len(option_pairs[idx])):
        str = str+ '"'+option_pairs[idx][innerIdx]+'"'+" "
    for innerMinMaxIdx in range(len(leastMost[idx])):
        str = str+'"'+leastMost[idx][innerMinMaxIdx]+'"'+" "
    printStr = printStr+str+'\n'

predictionFile = open("./nce_word_analogy_dev3.txt","a")
predictionFile.write(printStr)
predictionFile.close()










"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

word_id

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
