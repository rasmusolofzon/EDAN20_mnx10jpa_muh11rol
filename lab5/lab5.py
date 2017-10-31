import os
import transition
import conll
import features
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from subprocess import call


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


def parse_ml(stack, queue, graph, trans, deprel):
    
    # Right arc
    if stack and trans == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, deprel)
        return stack, queue, graph

    # Left arc
    if stack and trans == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, deprel)
        return stack, queue, graph

    # Reduce
    if stack and trans == 're' and transition.can_reduce(stack, graph):
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph

    # Shift
    stack, queue, graph = transition.shift(stack, queue, graph)
    
    return stack, queue, graph


def create_stack_queue_graph(sentence):

    stack = []
    queue = list(sentence)
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    transitions = []

    return stack, queue, graph, transitions


def create_train_set(sentences, feature_names):

    #nonprojectives = []

    X_dict = []
    y_symbols = []

    for sentence in sentences:

        # Create and prepare a stack, a queue and a graph
        stack, queue, graph, transitions = create_stack_queue_graph(sentence)

        while queue:
            # Extract the features
            feature_vector = features.extract(stack, queue, graph, feature_names, sentence)
            X_dict.append(feature_vector)
            #print(feature_vector)

            # Obtain the next transition using gold standard parsing
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)

            # Save the ground truth
            y_symbols.append(trans)
            #print(trans)

        # Handle the rest of the stack and clear it
        stack, graph = transition.empty_stack(stack, graph)
        
        #print('Equal graphs:', transition.equal_graphs(sentence, graph))
        #print(graph)
        
        """
        # Create a list of non-projective sentences
        if not transition.equal_graphs(sentence, graph):
            nonprojectives.append(sentence)

    # Obtain the shortest non-projective sentence
    shortest = min(nonprojectives, key=len)
    nonprojective = ''
    for word in shortest:
        if word['form'] != 'ROOT':
            nonprojective += word['form'] + ' '
    print(nonprojective)
    """

    return X_dict, y_symbols


def create_ml_model(sentences, feature_names):

    # Create a train set by extracting the features from the train sentences
    print('Extracting features from the train sentences...')
    X_dict, y_symbols = create_train_set(sentences, feature_names)
        
    # Learn the 'feature name -> value' mappings in X_dict and transform it to a one-hot encoding sparse matrix
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    
    # Translate the gold standard values to numbers
    le = LabelEncoder()
    y = le.fit_transform(y_symbols)
    #print(list(le.classes_))
    print('Feature vectors created!\n')

     # Fit the model according to the given training data
    print('Training the model...')
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    model = classifier.fit(X, y)
    print('Training done!\n')

    # Use the model to predict the train set and obtain its accuracy classification score
    y_predicted = classifier.predict(X)
    score = accuracy_score(y, y_predicted, normalize=False)
    nbr_of_predictions = len(y_predicted)
    accuracy_rate = round(score / nbr_of_predictions * 100, 2)
    print('The accuracy classification score for the model when predicting the translations generated by Nivre\'s parser ' +
        'on the train set is:\n  ' + str(score) + ' / ' + str(nbr_of_predictions) + ' * 100 = ' + str(accuracy_rate) + '%\n')
    
    return vec, le, classifier


def predict_graphs(sentences, feature_names, feature_set_index, dict_vectorizer, label_encoder, classifier, output_dir):
    
    # Create the output file
    print('Creating the output file...')
    file_name = output_dir + os.sep + 'system_output_' + str(feature_set_index) + '.txt'
    system_output_file = open(file_name, 'w').write(str(''))

    for sentence in sentences:

        # Create and prepare a stack, a queue and a graph
        stack, queue, graph, transitions = create_stack_queue_graph(sentence)

        while queue:
            # Extract the feature vector
            feature_vector = features.extract(stack, queue, graph, feature_names, sentence)
            #print(feature_vector)

            # Vectorize the feature_vector
            X = dict_vectorizer.transform(feature_vector)

            # Predict the transition and dependency relation for the word
            y = classifier.predict(X)
            trans = label_encoder.inverse_transform(y)[0]
            #print(trans)
            
            # Update the stack, the queue and the graph
            stack, queue, graph = parse_ml(stack, queue, graph, trans[:2], trans[3:])

        # Handle the rest of the stack and clear it
        stack, graph = transition.empty_stack(stack, graph)

        #print('Heads: ' + graph['heads'])
        #print('Deprels: ' + graph['deprels'])
        
        # Add the 'head' and the 'deprel' attributes to the sentence
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
            word['deprel'] = graph['deprels'][word['id']]

        # Write the sentence to the output file
        write_sentence_to_file(sentence[1:], file_name)

    # Close the output file
    #system_output_file.close()
    print('Output file created!\n')
    
    return


def write_sentence_to_file(sentence, file_name):
    
    with open(file_name, 'a') as system_output_file:
        for word in sentence:
            line = word['id'] + '\t' + word['form'] + '\t' + word['lemma'] + '\t' + word['cpostag'] + '\t' + word['postag'] \
                + '\t' + word['feats'] + '\t' + word['head'] + '\t' + word['deprel'] + '\t' + '_' + '\t' + '_' + '\n'
            system_output_file.write(line)
        system_output_file.write('\n')
    
    return


if __name__ == '__main__':
    
    # Define the directories for the datasets, program data and program output
    datasets_dir = 'datasets'
    data_dir = 'program_data'
    output_dir = 'output_data'

    # Define the train set, test set and evaluation files
    train_file = datasets_dir + os.sep + 'swedish_talbanken05_train.conll'
    test_file = datasets_dir + os.sep + 'swedish_talbanken05_test_blind.conll'
    eval_file = datasets_dir + os.sep + 'swedish_talbanken05_test.conll'
    
    # Define the CoNLL evaluation script
    eval_script = 'eval.pl'

    # Prepare the train set
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    train_sentences = conll.read_sentences(train_file)
    formatted_train_sentences = conll.split_rows(train_sentences, column_names_2006)

    # Prepare the test set
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    test_sentences = conll.read_sentences(test_file)
    formatted_test_sentences = conll.split_rows(test_sentences, column_names_2006_test)

    # Define the feature sets
    feature_names = []
    feature_names.append(['stack0_FORM', 'stack0_POS', 'queue0_FORM', 'queue0_POS', 'can_leftarc', 'can_reduce'])
    feature_names.append(feature_names[0] + ['stack1_FORM', 'stack1_POS', 'queue1_FORM', 'queue1_POS'])
    feature_names.append(feature_names[1] + ['sent_stack0fw_FORM', 'sent_stack0fw_POS', 'sent_stack0ffw_FORM',
        'sent_stack0ffw_POS', 'sent_stack0pw_FORM', 'sent_stack0pw_POS'])

    # For each feature set...
    for i in range(len(feature_names)):
        # Define the index of the feature set
        feature_set_index = i + 1
        print('Processing feature set '.upper() + str(feature_set_index) + '...\n')
        # Obtain a vectorizer, a label encoder and a classifier
        try:
            # If already created, load them from the program data directory
            classifier = pickle.load( open(data_dir + os.sep + 'classifier_' + str(feature_set_index) + '.p', 'rb'))
            le = pickle.load( open(data_dir + os.sep + 'label_encoder_' + str(feature_set_index) + '.p', 'rb'))
            vec = pickle.load( open(data_dir + os.sep + 'vectorizer_' + str(feature_set_index) + '.p', 'rb'))
        except:
            # If not already created, create them and save them in the program data directory
            vec, le, classifier = create_ml_model(formatted_train_sentences, feature_names[i])  
            pickle.dump(classifier, open(data_dir + os.sep + 'classifier_' + str(feature_set_index) + '.p', 'wb'))
            pickle.dump(le, open(data_dir + os.sep + 'label_encoder_' + str(feature_set_index) + '.p', 'wb'))
            pickle.dump(vec, open(data_dir + os.sep + 'vectorizer_' + str(feature_set_index) + '.p', 'wb'))

        # Make the predictions of the graphs, i.e. the heads and dependency relations, for each sentence in the test set
        predict_graphs(formatted_test_sentences, feature_names[i], feature_set_index, vec, le, classifier, output_dir)

        # Measure the accuracy of the parser using the CoNLL evaluation script
        print('The accuracy classification score for the model when predicting the graphs of the test set is: ')
        output_file = output_dir + os.sep + 'system_output_' + str(feature_set_index) + '.txt'
        call(["perl", eval_script, "-g", eval_file, "-s", output_file, "-q"])
        print('\n\n')
        
    
    
