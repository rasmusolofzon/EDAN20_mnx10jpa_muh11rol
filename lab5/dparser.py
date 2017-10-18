"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree

import pickle

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
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        # only do arc checks for words in stacks
        for word in stack:
            if (word['id'] == queue[0]['head'] or word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def parse_ml(stack, queue, graph, trans):
    
    # Right arc
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'    

    # Left arc
    if stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'

    # Reduce
    if stack and trans[:2] == 're' and transition.can_reduce(stack, graph):
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'

    # Shift
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


def extract_features(sentences, feature_names):

    #sent_cnt = 0
    #nonprojectives = []

    X_dict = []
    Y_symbols = []

    for sentence in sentences:
        #print(sentence)
        #sent_cnt += 1
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
            feature_vector = features.extract(stack, queue, graph, feature_names, sentence)
            #print(feature_vector)
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
    
    return X_dict, Y_symbols


def create_ml_models(sentences, feature_names):
    # Extract the features from the train sentences
    print('Extracting the features from the train set...')
    X_dict, Y_symbols = extract_features(formatted_train_sentences, feature_names)
        
    # Learn the list of 'feature name -> value' mappings in the dict, then transform the dict to a vector
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
   # Translate the gold standard values to numbers
    le = LabelEncoder()
    le.fit(Y_symbols)
    classes = list(le.classes_)
    Y = le.transform(Y_symbols)

    #print(len(classes))
    print('Feature vectors created!\n')

    #print('Nbr of feature vectors in X: ' + str(len(X_dict)))
    #print('Nbr of gold standard values in Y: ' + str(len(Y_symbols)))

    # Start the training phase
    print("Training the model...")
    # Fit the model according to the given training data
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    model = classifier.fit(X, Y)
    print("Training done!\n")

    return vec, le, classifier # we might need these later on: dict_classes, inv_dict_classes


def predict(sentences, feature_names, dict_vectorizer, label_encoder, classifier):
    open('system_output.txt', 'w').write(str(''))

    # graphs = []
    for sentence in sentences:

        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        while queue:
            # Extract the feature vector
            feature_vector = features.extract(stack, queue, graph, feature_names, sentence)
            
            # Vectorize the feature_vector
            #print(feature_vector)
            X = dict_vectorizer.transform(feature_vector)

            # predict the transition
            Y = classifier.predict(X)
            trans = label_encoder.inverse_transform(Y)[0]
            #print(trans)
            
            # in training phase: "stack, queue, graph, trans = reference(stack, queue, graph)""
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)


        stack, graph = transition.empty_stack(stack, graph)

        # graphs.append(graph)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
            word['deprel'] = graph['deprels'][word['id']]

        write_sentence_to_file(sentence[1:])
    return # graphs



def write_sentence_to_file(sentence):
    with open('system_output.txt', 'a') as system_output_file:
        for word in sentence:
            #if word['id'] != 0:
            line = word['id'] + '\t' + word['form'] + '\t' + word['lemma'] + '\t' + word['cpostag'] + '\t' + word['postag'] + '\t' \
               + word['feats'] + '\t' + word['head'] + '\t' + word['deprel'] + '\t' + '_' + '\t' + '_' + '\n'
            system_output_file.write(line)
        system_output_file.write('\n')
    return


if __name__ == '__main__':
    train_file = 'datasets/swedish_talbanken05_train.conll'
    test_file = 'datasets/swedish_talbanken05_test_blind.conll'

    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    
    train_sentences = conll.read_sentences(train_file)
    formatted_train_sentences = conll.split_rows(train_sentences, column_names_2006)
    
    test_sentences = conll.read_sentences(test_file)
    formatted_test_sentences = conll.split_rows(test_sentences, column_names_2006_test)

    feature_names_1 = ['stack0_FORM', 'stack0_POS', 'queue0_FORM', 'queue0_POS', 'can_leftarc', 'can_reduce']
    feature_names_2 = feature_names_1 + ['stack1_FORM', 'stack1_POS', 'queue1_FORM', 'queue1_POS']
    feature_names_3 = feature_names_2 + ['sent_stack0fw_FORM', 'sent_stack0fw_POS', 'sent_stack1h_POS', 'sent_stack1rs_FORM']
    feature_names = feature_names_3 # change n in feature_name_[n] to select feature vector

    try:
        classifier = pickle.load( open("classifier.p", "rb"))
        le = pickle.load( open("le.p", "rb"))
        vec = pickle.load( open("vec.p", "rb"))
    except:
        vec, le, classifier = create_ml_models(formatted_train_sentences, feature_names)  

        pickle.dump(classifier, open("classifier.p", "wb"))
        pickle.dump(le, open("le.p", "wb"))
        pickle.dump(vec, open("vec.p", "wb"))
  
    print("Now it is time to start the parsing with machine learning!\n")

    '''
        try:
            result_graphs = pickle.load(open("result_graphs.p", "rb"))
        except:
            # Start the parsing with machine learning
            result_graphs = predict(formatted_test_sentences, feature_names, vec, le, classifier)

            pickle.dump(result_graphs, open("result_graphs.p", "wb"))

        print(type(result_graphs))
    '''

    predict(formatted_test_sentences, feature_names, vec, le, classifier)
    


    