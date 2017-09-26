"""
Baseline chunker for CoNLL 2000
"""
__author__ = "Pierre Nugues"

import conll_reader


def count_pos(corpus):
    """
    Computes the part-of-speech distribution
    in a CoNLL 2000 file
    :param corpus:
    :return:
    """
    pos_cnt = {}
    for sentence in corpus:
        for row in sentence:
            if row['pos'] in pos_cnt:
                pos_cnt[row['pos']] += 1
            else:
                pos_cnt[row['pos']] = 1
    return pos_cnt


def train(corpus):
    """
    Computes the chunk distribution by pos
    The result is stored in a dictionary
    :param corpus:
    :return:
    """
    pos_cnt = count_pos(corpus)
    # We compute the chunk distribution by POS
    chunk_dist = {key: {} for key in pos_cnt.keys()}
    """
    chunk_dist = {
        'VBG':
            {},
        'TO':
            {},
        'NNP':
            {},
        ...
    }
    Fill in code to compute the chunk distribution for each part of speech

    corpus =
    ('form', 'pos', 'chunk')
    "Confidence NN B-NP
    in IN B-PP
    the DT B-NP
    pound NN I-NP
    is VBZ B-VP
    widely RB I-VP
    expected VBN I-VP
    to TO I-VP
    take VB I-VP
    another DT B-NP
    sharp JJ I-NP
    dive NN I-NP
    if IN B-SBAR
    trade NN B-NP
    figures NNS I-NP
    for IN B-PP
    September NNP B-NP
    , , O
    due JJ B-ADJP
    ..."
    """
    for sentence in corpus:
        for row in sentence:
            if row['chunk'] in chunk_dist[row['pos']]:
                chunk_dist[row['pos']][row ['chunk']] += 1
            else:
                chunk_dist[row['pos']][row ['chunk']] = 1

    """
    chunk_dist = {
        'VBG':
            { 'I-ADJP': 53, 'B-PP': 16, ..},
        'TO':
            { 'O': 7403, 'B-SBAR': 64, ..},
        'NNP':
            { 'B-VP': 81, ..},
        ...
    }
    Fill in code so that for each part of speech, you select the most frequent chunk.
    You will build a dictionary with key values:
    pos_chunk[pos] = most frequent chunk for pos
    """
    # We determine the best association
    pos_chunk = {}
    
    counts = []
    for pos in chunk_dist:
        chunktags_occ = dict(chunk_dist[pos].items())
        # for i in range(len(chunktags_occ)):
        # 	if ()
        #   		counts.append(val)
        #   	max(counts)	
        #print(chunktags_occ)
        print(pos, max(chunktags_occ, key = chunktags_occ.get), chunktags_occ[max(chunktags_occ, key = chunktags_occ.get)])
        pos_chunk[pos] = max(chunktags_occ, key = chunktags_occ.get)

    return pos_chunk


def predict(model, corpus):
    """
    Predicts the chunk from the part of speech
    Adds a pchunk column
    :param model:
    :param corpus:
    :return:
    """
    """
    We add a predicted chunk column: pchunk
    """
    for sentence in corpus:
        for row in sentence:
            row['pchunk'] = model[row['pos']]
    return corpus


def eval(predicted):
    """
    Evaluates the predicted chunk accuracy
    :param predicted:
    :return:
    """
    word_cnt = 0
    correct = 0
    for sentence in predicted:
        for row in sentence:
            word_cnt += 1
            if row['chunk'] == row['pchunk']:
                correct += 1
    return correct / word_cnt


if __name__ == '__main__':
    column_names = ['form', 'pos', 'chunk']
    train_file = './train.txt'
    test_file = './test.txt'

    train_corpus = conll_reader.read_sentences(train_file)
    train_corpus = conll_reader.split_rows(train_corpus, column_names)
    test_corpus = conll_reader.read_sentences(test_file)
    test_corpus = conll_reader.split_rows(test_corpus, column_names)
    #print(test_corpus)

    model = train(train_corpus)
    print('after train')
    predicted = predict(model, test_corpus)
    accuracy = eval(predicted)
    print("Accuracy", accuracy)
    f_out = open('out', 'w')
    # We write the word (form), part of speech (pos),
    # gold-standard chunk (chunk), and predicted chunk (pchunk)
    for sentence in predicted:
        for row in sentence:
            """print(row['form'] + ' ' + row['pos'] + ' ' +
                        row['chunk'] + ' ' + row['pchunk'] + '\n')"""
            f_out.write(row['form'] + ' ' + row['pos'] + ' ' +
                        row['chunk'] + ' ' + row['pchunk'] + '\n')
        f_out.write('\n')
    f_out.close()

"""
bygg modell > träna modellen med train-setet > förutsäg med modellen på test-setet

a b c d e 

a b c d E
a b c D e 
a b C d e
..
"""