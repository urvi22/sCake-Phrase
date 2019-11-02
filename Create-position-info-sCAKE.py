# 0.

import os
import re
import nltk
import string
import pandas as pd
from read_write_create import *
from nltk.stem import PorterStemmer
import unicodedata

global cwd, path, data_path

cwd = os.getcwd()
path = cwd
data_path = cwd + "/data"
create_folder(cwd, "positions")

numbers_ex = re.compile("[0-9]+(-[0-9]+)?")
punctuation_ex = re.compile("[^a-z ]")
roman_num_ex = re.compile("\\b[i|v|x|l|c|d|m]{1,3}\\b")
stopwords = read_list_from_file(path, "stopwords.txt")
ps = PorterStemmer()

print("create-position-info-sCake")
for every_file in (os.listdir(data_path)):

    print(every_file)
    text = read_text_from_file(data_path, every_file)

    ## pre-processing text
    text = text.strip()
    text = text.lower()
    text = re.sub(numbers_ex, '', text)
    text = re.sub(punctuation_ex, '', text)
    text = re.sub(roman_num_ex, '', text)

    words = nltk.word_tokenize(text)
    words = [i for i in words if i not in stopwords]
    words = [ps.stem(i) for i in words]

    # for i in range(len(words)):
    #     words[i] = words[i].encode('utf-8')
        #print(type(i))

   # print(words)

    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words)

    bigramFinder.apply_freq_filter(5)
    bi = list(bigramFinder.score_ngrams(bigrams.pmi))
    # bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram', 'PMI']).sort_values(
    #     by='PMI', ascending=False)

    # print(bigramPMITable[:50])

    trigrams = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    trigramFinder.apply_freq_filter(5)
    # trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)),
    #                                columns=['trigram', 'PMI']).sort_values(by='PMI', ascending=False)

    tri = list(trigramFinder.score_ngrams(trigrams.pmi))


    #     unicodedata.normalize('NFKD', t[0][0]).encode('ascii', 'ignore')
    #     unicodedata.normalize('NFKD', t[0][1]).encode('ascii', 'ignore')
    #     unicodedata.normalize('NFKD', t[0][2]).encode('ascii', 'ignore')

    #print(biG)
    w=0
    while w < len(words)-2:
        #print('bahar', w, len(words))
        for b in bi:
           # print('andar', w, len(words))
            if words[w] == b[0][0] and words[w + 1] == b[0][1]:

                f = 0
                for t in tri:
                    if b[0][0] == t[0][0] and b[0][1] == t[0][1]:
                        f = 1
                        words[w] = words[w] + '-' + words[w + 1] + '-' + words[w + 2]
                        #words[w] = words[w].encode('utf-8')
                        words.pop(w + 1)
                        words.pop(w + 1)
                        break

                if f is 0:
                    words[w] = words[w] + '-' + words[w + 1]
                    #words[w] = words[w].encode('utf-8')
                    words.pop(w + 1)

                break
        w += 1

    #
    # print(trigramPMITable[:50])
    #print(words)
    selected_words = list(set(words))
    #print(selected_words)

    # for w in range(len(selected_words)):
    #     if selected_words[w] is 'machin learn':
    #         print(w)

    ## end of pre-processing

    N = len(words) + 1
    posi = list()
    t = list()
    tf = list()

    for w in selected_words:
        posw = [i for i, word in enumerate(words) if w == word]
        w_freq = len(posw) + 1
        posw.append(N)
        t.append(w)
        tf.append(w_freq)
        posi.append(posw)

    # print(posi)

    data = dict()
    data["words"] = t
    data["tf"] = tf
    data["positions"] = posi

    df = pd.DataFrame(data=data)

    print(df.head())
    df.to_pickle(cwd + "/positions/" + every_file[:-4] + ".pkl")
