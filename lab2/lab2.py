# python tokenize_simple.py < Selma.txt  | sort | uniq | wc -w

import sys
import regex as re

def tokenize(text):
	words = re.findall('[A-Ö][\s\S]*?[\.!?]', text)
	words = [words[i][:-1].replace('\n', '').replace('\xad', '').lower() 
		for i in range(len(words))]
	return words

def xmlify(listofstrings):
	listofstrings = ["<s> " + listofstrings[i] + " </s>" for i in range(len(listofstrings))]
	return listofstrings

def tokenize1(text):
    words = re.findall("\p{L}+", text)
    return words

def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]

    frequency_bigrams = {}
    for bigram in bigrams:
        if bigram in frequency_bigrams:
            frequency_bigrams[bigram] += 1
        else:
            frequency_bigrams[bigram] = 1
    return frequency_bigrams

def mutual_info(words, freq_unigrams, freq_bigrams):
    mi = {}
    factor = len(words) * len(words) / (len(words) - 1)
    for bigram in freq_bigrams:
        mi[bigram] = (
            math.log(factor * freq_bigrams[bigram] /
                     (freq_unigrams[bigram[0]] *
                      freq_unigrams[bigram[1]]), 2))
    return mi

#TODO: "Estimate roughly the accuracy of your program."

# if __name__ == '__main__':
text = open(sys.argv[1], 'r', encoding='utf-8').read()
tokenized = tokenize(text)

#print(xmlify(tokenized))

words = tokenize1(text)
frequency_bigrams = count_bigrams(words)
nbr_of_bigrams = 0
for bigram in frequency_bigrams:
	nbr_of_bigrams += frequency_bigrams[bigram]

import math 

print("nbr of words = " + str(len(words)))
print("nbr of bigrams = " + str(nbr_of_bigrams))
print("possible nbr of bigrams = " + str(math.pow(len(words), 2)))
print("possible nbr of 4-grams = " + str(math.pow(len(words), 4)))

# Propose a solution to cope with bigrams unseen in the corpus. This topic will be discussed during the lab session.

frequency = count_unigrams(words)
mi = mutual_info(words, frequency, count_bigrams(words))

"""for bigram in sorted(mi.keys(), key=mi.get, reverse=True):
	if frequency_bigrams[bigram] < 1: continue
	print(mi[bigram], '\t', bigram, '\t',
			frequency[bigram[0]], '\t',
			frequency[bigram[1]], '\t',
			frequency_bigrams[bigram])"""

print("\nUnigram model")
print("======================================")
print("wi\tC(wi)\t#words\tP(wi)")
print("======================================")
print("stuff\nstuff\nstuff")
print("======================================")
print("Prob. unigrams: None")
print("Geometric mean prob.: None")
print("Entropy rate: None")
print("Perplexity: None")

print("\nBigram model")
print("======================================")
print("wi\twi+1\tCi,i+1\tC(i)\tP(wi+1|wi)")
print("======================================")
print("stuff\nstuff\nstuff")
print("======================================")
print("Prob. unigrams: None")
print("Geometric mean prob.: None")
print("Entropy rate: None")
print("Perplexity: None")