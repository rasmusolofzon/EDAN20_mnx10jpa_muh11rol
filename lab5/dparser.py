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


def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))

    # We assign each name a number
    dict_classes = dict(enumerate(classes))

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[y_symbol] for y_symbol in y_symbols]
    return y, dict_classes, inv_dict_classes



if __name__ == '__main__':
    train_file = 'datasets/swedish_talbanken05_train.conll'
    test_file = 'datasets/swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    feature_names_1 = ['stack0_FORM', 'stack0_POS', 'queue0_FORM', 'queue0_POS', 'can_leftarc', 'can_reduce']
    feature_names_2 = feature_names_1 + ['stack1_FORM', 'stack1_POS', 'queue1_FORM', 'queue1_POS']
    feature_names_3 = feature_names_2 + ['sent_stack0fw_FORM', 'sent_stack0fw_POS', 'sent_stack1h_POS', 'sent_stack1rs_FORM']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0

    nonprojectives = []

    X_dict_1 = []
    X_dict_2 = []
    X_dict_3 = []

    Y_symbols = []

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
            feature_vector = features.extract(stack, queue, graph, feature_names_1, sentence)
            X_dict_1.append(feature_vector)
            feature_vector = features.extract(stack, queue, graph, feature_names_2, sentence)
            X_dict_2.append(feature_vector)
            feature_vector = features.extract(stack, queue, graph, feature_names_3, sentence)
            X_dict_3.append(feature_vector)

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
    print('Nbr of feature vectors in X3: ' + str(len(X_dict_3)))
    print('Nbr of gold standard values in Y: ' + str(len(Y_symbols)))

    classes = ['la', 'ra', 'sh', 're']
    vec = DictVectorizer(sparse=True)
    X_1 = vec.fit_transform(X_dict_1)
    X_2 = vec.fit_transform(X_dict_2)
    X_3 = vec.fit_transform(X_dict_3)
    Y, dict_classes, inv_dict_classes = encode_classes(Y_symbols)
