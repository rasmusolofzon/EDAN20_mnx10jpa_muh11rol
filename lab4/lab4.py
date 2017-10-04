''' 
    note: (CoNLL-X == CoNLL2006) = True
    format desc: https://web.archive.org/web/20161105025307/http://ilk.uvt.nl/conll/
'''

import conll
import os

if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    train_file = 'datasets/swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = 'datasets/swedish_talbanken05_test.conll'

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    print(train_file, len(formatted_corpus))
    print(formatted_corpus[0])
    print(formatted_corpus[1])

    for sentence in formatted_corpus:
        for word in sentence:
            if '' inword.keys():

    


    '''
    [ #formaterat corpus
        [ #mening
            {ord},
            {1, SS, 2}, <--
            {2, VERB}
        ],
        [],
        []
    ]

    > söka efter ett verb
    > titta på verbets DEPREL
    > om(DEPREL == SS) skapa en tupel av verb-subjekt
    > om(tupeln inte finns) lägg in tupeln i nya dictionary:n
    > annars addera ett till värdet av sagda tupel

    { #dict:en vi ska skapa
        (VERB, SUBJEKT): X,
        (VERB, SUBJEKT): Y,
        (VERB, SUBJEKT): Z
    }
    '''