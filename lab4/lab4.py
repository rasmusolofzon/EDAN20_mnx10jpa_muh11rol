''' 
    note: (CoNLL-X == CoNLL2006) = True
    format desc: https://web.archive.org/web/20161105025307/http://ilk.uvt.nl/conll/
    functional categories (sv): http://stp.lingfil.uu.se/~nivre/swedish_treebank/GF.html
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
    #print(formatted_corpus[1])

    subj_verb_pairs = {}

    for sentence in formatted_corpus:
        for word in sentence:
            if word['deprel'] == 'SS':
                for head in sentence:
                    if head['id'] == word['head']:
                        if (word['form'], head['form']) in subj_verb_pairs.keys():
                            subj_verb_pairs[(word['form'], head['form'])] += 1
                        else:
                            subj_verb_pairs[(word['form'], head['form'])] = 1

    print("unique pairs " + str(len(subj_verb_pairs)))

    summa = 0
    for freq in subj_verb_pairs.values():
        summa += freq
    
    print("Total nbr of frequencies " + str(summa) + ", whoopdeefuckingdoo")


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