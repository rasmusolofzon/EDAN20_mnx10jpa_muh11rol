"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features

import time
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree



def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' #+ deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' #+ deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

if __name__ == '__main__':
    train_file = 'datasets/swedish_talbanken05_train.conll'
    test_file = 'datasets/swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    feature_names_1 = ['w_1_STACK', 'pos_1_STACK', 'w_1_QUEUE', 'pos_1_QUEUE', 'can_left_arc', 'can_reduce']
    feature_names_2 = feature_names_1 + ['w_2_STACK', 'pos_2_STACK','w_2_QUEUE', 'pos_2_QUEUE']
    feature_names_3 = feature_names_2 + ['w_TOP_plus_1', 'pos_TOP_plus_1', 'pos_1_STACK_h', 'lex_1_STACK_rs']


    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0

    nonprojectives = []

    X_dict = [];
    Y_symbols = [];

    for sentence in formatted_corpus:
        #print(sentence)
        sent_cnt += 1
        #if sent_cnt % 1000 == 0:
        #    print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        while queue:
            # Extract the features
            feature_vector = features.extract(stack, queue, graph, feature_names_3, sentence)
            X_dict.append(feature_vector)

            # Obtain the next transition
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)

            # Save the ground truth
            Y_symbols.append(trans)

            #print(trans)
            #print('\n')


        stack, graph = transition.empty_stack(stack, graph)
        #print('Equal graphs:', transition.equal_graphs(sentence, graph))

        # Create a list of non-projective sentences
        '''if not transition.equal_graphs(sentence, graph):
            nonprojectives.append(sentence)'''

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        #print(transitions)
        #print(graph)
            
    # Obtain the shortest non-projective sentence
    '''shortest = min(nonprojectives, key=len)
    nonprojective = ''
    for word in shortest:
        if word['form'] != 'ROOT':
            nonprojective += word['form'] + ' '
    print(nonprojective)'''

    #print(sentences[3])
    #for line in formatted_corpus[3]: print(line)
    
    
    #print(X)
    #print(Y)
    print('Nbr of feature vectors in X: ' + str(len(X_dict)))
    print('Nbr of gold standard values in Y: ' + str(len(Y_symbols)))





    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)

    classes = ['la', 'ra', 'sh', 're']
    
    Y, dict_classes, inv_dict_classes = features.encode_classes(Y_symbols)

    """
    Y = []
    for y_symbol in Y_symbols:
        Y.append(classes.index(y_symbol))
    """


    #print(X)
    #print(Y)
