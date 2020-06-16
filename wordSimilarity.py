import os
import pickle
import numpy as np
import re
from scipy import spatial
import zipfile
import collections
import tensorflow as tf

model_path = './models/'
loss_model = 'cross_entropy'

model_filepath = os.path.join(model_path, 'word2vec_%s.model3'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))


wordIdFirst = dictionary['first']
wordIdAmerican = dictionary['american']
wordIdWould = dictionary['would']
embeddingFirst = embeddings[wordIdFirst]
embeddingAmerican = embeddings[wordIdAmerican]
embeddingWould = embeddings[wordIdWould]

f=open("./word_analogy_test.txt", "r")
contents =f.read()
pairs = re.split('\n', contents )
contents = contents.replace('"','')
pairs = contents.split('\n')
word_pairs = [[]]
for pair in pairs:
    words = pair.split('||')
    words[0] = words[0].split(",")
    word_pairs.append(words[0])
dict1 = []
for wordIdx in range(1,len(word_pairs)):
    for idx in range(len(word_pairs[wordIdx])):
        wordPair = word_pairs[wordIdx][idx].split(':')
        word_id1 = dictionary[wordPair[0]]
        word_id2 = dictionary[wordPair[1]]
        if word_id1 not in dict1:
            dict1.append(word_id1)
        if word_id2 not in dict1:
            dict1.append(word_id2)

embeddingsFAW = [embeddingFirst, embeddingAmerican, embeddingWould]
maxWords = []
for faw in range(len(embeddingsFAW)):
    maxCosine = 0;
    maxWordsIds = []
    for i in range(20):
        maxWordId = 0
        for wordId in dict1:
            if((1-spatial.distance.cosine(embeddingsFAW[faw], embeddings[wordId])) > maxCosine and embeddingsFAW[faw].all() != embeddings[wordId].all() and (not embeddings[wordId] in maxWordsIds)):
                maxCosine = (1-spatial.distance.cosine(embeddingsFAW[faw], embeddings[wordId]))
                maxWordId = wordId
        maxWordsIds.append(maxWordId)
    maxWords.append(maxWordsIds)

print(len(maxWords[0]))
def read_data(filename):
  #Extract the first file enclosed in a zip file as a list of words
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary_size = 100000
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

words = read_data('text8.zip')
print('Data size', len(words))

_,_,_,reverse_dictionary = build_dataset(words)

print("Nearest to first: ")
for wordId in maxWords[0]:
    print(reverse_dictionary[wordId])
print("Nearest to American: ")
for wordId in maxWords[1]:
    print(reverse_dictionary[wordId])
print("Nearest to would: ")
for wordId in maxWords[2]:
    print(reverse_dictionary[wordId])



