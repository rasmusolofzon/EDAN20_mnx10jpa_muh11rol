# python tokenize_simple.py < Selma.txt  | sort | uniq | wc -w

# frågor:
# for word in tokenize_words(sentence) - bra?
# P(<\s>) - räkna med?
#TODO: "Estimate roughly the accuracy of your program."
# Propose a solution to cope with bigrams unseen in the corpus. This topic will be discussed during the lab session.




import sys
import regex as re
import math

def tokenize_sentences(text):
	words = re.findall('[A-Ö][\s\S]*?[\.!?]', text)
	words = [words[i][:-1].replace('\n', '').replace('\xad', '').lower() 
		for i in range(len(words))]
	return words

def tokenize_words(text):
    words = re.findall("\p{L}+", text) 
    words = [words[i].lower() for i in range(len(words))]
    return words

def xmlify(listofstrings):
	listofstrings = ["<s> " + listofstrings[i] + " </s>" for i in range(len(listofstrings))]
	return listofstrings

def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency

def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2]) for inx in range(len(words) - 1)]        
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

def sentence_prob(sentence, frequency_unigrams):
    sentence_prob = 1
    wordcount = 0

    values = frequency_unigrams.values()
    for value in values:
        wordcount += value

    print(frequency_unigrams)
    print(wordcount)

    words = tokenize_words(sentence)
    for word in words[1:]:
        sentence_prob *= frequency_unigrams[word]/wordcount
    
    return sentence_prob



# if __name__ == '__main__':
text = open(sys.argv[1], 'r', encoding='utf-8').read()
# tokenized = tokenize_sentences(text)

# foo = 0
# for sent in tokenized:
#     foo += len(tokenize_words(sent))
#     print(tokenize_words(sent))
# print(foo)
# #print(xmlify(tokenized))

words = tokenize_words(text)
print(len(words))
freq_unigrams = count_unigrams(words)
freq_bigrams = count_bigrams(words)
#mi = mutual_info(words, freq_unigrams, count_bigrams(words))

nbr_of_bigrams = 0
for bigram in freq_bigrams:
	nbr_of_bigrams += freq_bigrams[bigram]



# for bigram in sorted(mi.keys(), key=mi.get, reverse=True):
# 	if frequency_bigrams[bigram] < 1: continue
# 	print(mi[bigram], '\t', bigram, '\t',
# 			frequency_unigrams[bigram[0]], '\t',
# 			frequency_unigrams[bigram[1]], '\t',
# 			frequency_bigrams[bigram])

# print("nbr of words = " + str(len(words)))
# print("nbr of bigrams = " + str(nbr_of_bigrams))
# print("possible nbr of bigrams = " + str(math.pow(len(words), 2)))
# print("possible nbr of 4-grams = " + str(math.pow(len(words), 4)))

print(sentence_prob('<s> det var en gång en katt som hette nils <\s>', freq_unigrams))

#print(freq_unigrams['detnils'])




