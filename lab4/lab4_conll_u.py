''' 
    note: (CoNLL-X == CoNLL2006) = True
    format desc: https://web.archive.org/web/20161105025307/http://ilk.uvt.nl/conll/
    functional categories (sv): http://stp.lingfil.uu.se/~nivre/swedish_treebank/GF.html
'''
import sys
import conll
import os

def get_sv_pairs(formatted_corpus):

    subj_verb_pairs = {}
    for sentence in formatted_corpus:
        for word in sentence:
            if word['deprel'] == 'nsubj':
                for head in sentence:
                    if head['id'] == word['head']:
                        if (word['form'].lower(), head['form'].lower()) in subj_verb_pairs.keys():
                            subj_verb_pairs[(word['form'].lower(), head['form'].lower())] += 1
                        else:
                            subj_verb_pairs[(word['form'].lower(), head['form'].lower())] = 1
    return subj_verb_pairs




def get_sov_triples(formatted_corpus):

    sov_triplets = {}
    for sentence in formatted_corpus:
        for word in sentence:
            if word['deprel'] == 'nsubj':
                for other_word in sentence:
                    #if head['id'] == word['head']: 
                    if other_word['head'] == word['head'] and other_word['deprel'] == 'obj': 
                        for head in sentence:
                            if head['id'] == word['head'] and head['id'] == other_word['head']:
                                if (word['form'].lower(), other_word['form'].lower(), head['form'].lower()) in sov_triplets.keys():
                                    sov_triplets[(word['form'].lower(), other_word['form'].lower(), head['form'].lower())] += 1
                                else:
                                    sov_triplets[(word['form'].lower(), other_word['form'].lower(), head['form'].lower())] = 1
    return sov_triplets



def reformate_corpus(formatted_corpus):
    reformatted_corpus = []
    for sentence in formatted_corpus:
        reformatted_sentence = {}
        for word in sentence:
            reformatted_sentence[word['id']] = word
        reformatted_corpus.append(reformatted_sentence)

    return reformatted_corpus
                


if __name__ == '__main__':
    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']
    language = sys.argv[1]
    print(language)
    train_file = 'datasets/' + language + '-ud-train.conllu'


    formatted_corpus = conll.split_rows(conll.read_sentences(train_file), column_names_u)


    reformatted_corpus = reformate_corpus(formatted_corpus)
    print(formatted_corpus)


    subj_verb_pairs = get_sv_pairs(formatted_corpus)
    sv_pair_freq = 0
    for freq in subj_verb_pairs.values():
        sv_pair_freq += freq
    print("\nTotal nbr of subject–verb pairs: " + str(sv_pair_freq))
    print("\nThe top 5 most common subject–verb pairs are: ")
    sorted_pairs = sorted(subj_verb_pairs.items(), key=lambda pair: pair[1])
    for i in range(1,6):
        print(sorted_pairs[len(sorted_pairs)-i])

    print('\n')

    sov_triplets = get_sov_triples(formatted_corpus)
    sov_triple_freq = 0
    for freq in sov_triplets.values():
        sov_triple_freq += freq
    print("\nTotal nbr of subject–verb–object triples: " + str(sov_triple_freq))
    print("\nThe top 5 most common subject–verb-object triples are: ")
    sorted_triplets = sorted(sov_triplets.items(), key=lambda triplet: triplet[1])
    for i in range(1,6):
        print(sorted_triplets[len(sorted_triplets)-i])




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