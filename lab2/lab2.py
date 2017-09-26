# python tokenize_simple.py < Selma.txt  | sort | uniq | wc -w

import sys
import regex as re
import math

def tokenize_sentences(text):
	words = re.findall('[A-Ö][\s\S]*?[\.!?]', text)
	words = [words[i][:-1].replace('\n', '').replace('\xad', '').lower() 
		for i in range(len(words))]
	return words

def tokenize_words(text):
    words = re.findall('[\p{L}<>/]+', text) 
    words = [words[i].lower() for i in range(len(words))]
    return words

def xmlify(listofstrings):
	listofstrings = ['<s> ' + listofstrings[i] + ' </s>' for i in range(len(listofstrings))]
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

def print_stats(words, freq_bigrams):
    print('\n')
    print('General statistics')
    print('======================================')
    nbr_of_bigrams = 0
    for bigram in freq_bigrams:
        nbr_of_bigrams += freq_bigrams[bigram]
    print('nbr of words = ' + str(len(words)))
    print('nbr of bigrams = ' + str(nbr_of_bigrams))
    print('possible nbr of bigrams = ' + str(math.pow(len(words), 2)))
    print('possible nbr of 4-grams = ' + str(math.pow(len(words), 4)))

def prob_unigrams(sentence, frequency_unigrams):
    sentence_prob = 1
    wordcount = 0

    values = frequency_unigrams.values()
    for value in values:
        wordcount += value

    print('\n')
    print('Unigram model')
    print('======================================')
    print('wi C(wi) #words P(wi)')
    print('======================================')

    words = tokenize_words(sentence)
    for word in words:
        word_freq = frequency_unigrams[word]
        word_prob = word_freq/wordcount
        sentence_prob *= word_prob
        print(word + ' ' + str(word_freq) + ' ' + str(wordcount) + ' ' + str(word_prob))
    
    n = len(words)
    prob_mean = math.pow(sentence_prob, 1/n)
    entropy_rate = -1/n*math.log(sentence_prob, 2)
    perplexity = pow(2, entropy_rate)
    print('======================================')
    print('Prob. unigrams: ' + str(sentence_prob))
    print('Geometric mean prob.: ' + str(prob_mean))
    print('Entropy rate: ' + str(entropy_rate))
    print('Perplexity: ' + str(perplexity))

    return sentence_prob


def prob_bigrams(sentence, frequency_unigrams, frequency_bigrams):
    sentence_prob = 1
    wordcount = 0

    values = frequency_unigrams.values()
    for value in values:
        wordcount += value

    print('\n')
    print('Bigram model')
    print('======================================')
    print('wi wi+1 Ci,i+1 C(i) P(wi+1|wi)')
    print('======================================')

    words = tokenize_words(sentence)

    for i in range(len(words)-1):

        bigram = (words[i], words[i+1]) 

        unigram_freq = freq_unigrams[words[i]]
        prev_unigram_freq = 0
        bigram_freq = 0
        bigram_prob = 1
        backoff = False

        if bigram in frequency_bigrams:
            bigram_freq = frequency_bigrams[bigram]
            prev_unigram_freq = frequency_unigrams[words[i-1]]
            bigram_prob *= bigram_freq / unigram_freq
        else:
            backoff = True
            bigram_prob = unigram_freq / wordcount
        
        sentence_prob *= bigram_prob
        print(bigram[0] + ' ' + bigram[1] + ' ' + str(bigram_freq) + ' ' + str(prev_unigram_freq) + ' ' 
             + (' *backoff' if backoff else '' ) + str(bigram_prob)) 

    n = len(words)
    prob_mean = math.pow(sentence_prob, 1/n)
    entropy_rate = -1/n*math.log(sentence_prob, 2)
    perplexity = pow(2, entropy_rate)
    print('======================================')
    print('Prob. unigrams: ' + str(sentence_prob))
    print('Geometric mean prob.: ' + str(prob_mean))
    print('Entropy rate: ' + str(entropy_rate))
    print('Perplexity: ' + str(perplexity))

    return sentence_prob


if __name__ == '__main__':
    text = open(sys.argv[1], 'r', encoding='utf-8').read()
    tokenized = xmlify(tokenize_sentences(text))

    words = []
    for sentence in tokenized:
        for word in tokenize_words(sentence):
            words.append(word)
    freq_unigrams = count_unigrams(words)
    freq_bigrams = count_bigrams(words)
    
    print_stats(words, freq_bigrams)
    
    test_set = [37, 548, 1065, 6735, 55829]
    for i in range(len(test_set)):
        prob_unigrams(tokenized[test_set[i]], freq_unigrams)
        prob_bigrams(tokenized[test_set[i]], freq_unigrams, freq_bigrams)

   