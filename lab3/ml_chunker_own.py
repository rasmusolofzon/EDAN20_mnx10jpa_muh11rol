"""
Machine learning chunker for CoNLL 2000
"""
__author__ = "Pierre Nugues + edits by Joel & Rasmus"

import time
import conll_reader
from sklearn.feature_extraction import DictVectorizer
# from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
# from sklearn.naive_bayes import GaussianNB
# from sklearn.grid_search import GridSearchCV
import pickle

def extract_features(sentences, w_size, feature_names, training_phase):
    """
    Builds X matrix and y vector
    X is a list of dictionaries and y is a list
    :param sentences:
    :param w_size:
    :return:
    """
    X_l = []
    y_l = []
    for sentence in sentences:
        X, y = extract_features_sent(sentence, w_size, feature_names, training_phase)
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l


def extract_features_sent(sentence, w_size, feature_names, training_phase):
    """
    Extract the features from one sentence
    returns X and y, where X is a list of dictionaries and
    y is a list of symbols
    :param sentence:
    :param w_size:
    :return:
    """

    # We pad the sentence to extract the context window more easily
    start = "BOS BOS BOS BOS\n"
    end = "\nEOS EOS EOS EOS"
    start *= w_size
    end *= w_size
    sentence = start + sentence
    sentence += end

    # Each sentence is a list of rows
    sentence = sentence.splitlines()
    padded_sentence = list()
    for line in sentence:
        line = line.split()
        padded_sentence.append(line)
    # print(padded_sentence)

    # We extract the features and the classes
    # X contains a list of features, where each feature vector is a dictionary
    # y is the list of classes
    X = list()
    y = list()
    for i in range(len(padded_sentence) - 2 * w_size):
        # x is a row of X
        x = list()
        # The words in lower case
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j][0].lower())
        # The POS
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j][1])
        
        # The chunks (Up to the word)
        # for j in range(w_size):
        #     x.append(padded_sentence[i + j][2])
        if training_phase:
            # The preceding 'predicted' chunks
            for j in range(w_size):
                x.append(padded_sentence[i + j][2])
                # x.append(padded_sentence[i + j + 1][2])
        # elif not training_phase:
            # print("nopsled, wie")
            '''
                here, need to predict and add continously. 
            
                # skissande
                if padded_sentence[i - 1][0].lower() == 'bos' and padded_sentence[i - 2][0].lower() == 'bos':
                    # predict chunk oldschool way
                elif padded_sentence[i - 1][0].lower() != 'bos' padded_sentence[i - 2][0].lower() == 'bos':
                    # predict chunk half-old, half-new: use predicted chunk of w_(i-1) but disregard 'bos' of w_(i-2)
                else:
                    # business as usual, predict with predicted chunks of w_(i-1) and w_(i-2)

                # could also express this as:
                if i == 0:
                    # 
                elif i == 1:
                    # 
                else:
                    # 
            
            
                har dock fortf problemet att nuvarande set-up är att skilja på 'extract features' och 'predicting' väldigt hårt.
                svårt som det är nu att trycka in 'extract (conditionally)'->'predict'->'extract (conditionally & w/ last prediction)' 
                    någonstans.
                möjliga lösningar på detta är att:
                    * försöka trycka in detta hårt och i och med det sabba den logiska uppdelning av metoder, 
                        introducera implicit och icke generaliserbart beteende
                    * skriva om funktionerna, i princip göra en 'predict_continously'-funktion (ish). 
                        Gör det tydligare vad som händer, och behöver inte formatera om så mycket existerande kod. Let's do this.
            '''
        
        # We represent the feature vector as a dictionary
        X.append(dict(zip(feature_names, x)))  # {'w_i-2': 'The', 'w_i-1': 'cat', 'w_i': 'ate', ... 't_i'}
        # print(X)
        # The classes are stored in a list
        y.append(padded_sentence[i + w_size][2])
    return X, y


def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))
    """
    Results in:
    ['B-ADJP', 'B-ADVP', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-NP', 'B-PP',
    'B-PRT', 'B-SBAR', 'B-UCP', 'B-VP', 'I-ADJP', 'I-ADVP', 'I-CONJP',
    'I-INTJ', 'I-NP', 'I-PP', 'I-PRT', 'I-SBAR', 'I-UCP', 'I-VP', 'O']
    """
    # We assign each name a number
    dict_classes = dict(enumerate(classes))
    """
    Results in:
    {0: 'B-ADJP', 1: 'B-ADVP', 2: 'B-CONJP', 3: 'B-INTJ', 4: 'B-LST',
    5: 'B-NP', 6: 'B-PP', 7: 'B-PRT', 8: 'B-SBAR', 9: 'B-UCP', 10: 'B-VP',
    11: 'I-ADJP', 12: 'I-ADVP', 13: 'I-CONJP', 14: 'I-INTJ',
    15: 'I-NP', 16: 'I-PP', 17: 'I-PRT', 18: 'I-SBAR',
    19: 'I-UCP', 20: 'I-VP', 21: 'O'}
    """

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}
    """
    Results in:
    {'B-SBAR': 8, 'I-NP': 15, 'B-PP': 6, 'I-SBAR': 18, 'I-PP': 16, 'I-ADVP': 12,
    'I-INTJ': 14, 'I-PRT': 17, 'I-CONJP': 13, 'B-ADJP': 0, 'O': 21,
    'B-VP': 10, 'B-PRT': 7, 'B-ADVP': 1, 'B-LST': 4, 'I-UCP': 19,
    'I-VP': 20, 'B-NP': 5, 'I-ADJP': 11, 'B-CONJP': 2, 'B-INTJ': 3, 'B-UCP': 9}
    """

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[element] for element in y_symbols]
    return y, dict_classes, inv_dict_classes

'''
def predict(test_sentences, feature_names, f_out, classifier):
    for test_sentence in test_sentences:
        X_test_dict, y_test_symbols = extract_features_sent(test_sentence, w_size, feature_names, training_phase = False)
        # Vectorize the test sentence and one hot encoding
        X_test = vec.transform(X_test_dict)
        # Predicts the chunks and returns numbers
        y_test_predicted = classifier.predict(X_test)
        # Converts to chunk names
        y_test_predicted_symbols = [dict_classes[i] for i in y_test_predicted]
        # Appends the predicted chunks as a last column and saves the rows
        rows = test_sentence.splitlines()
        rows = [rows[i] + ' ' + y_test_predicted_symbols[i] for i in range(len(rows))]
        for row in rows:
            f_out.write(row + '\n')
        f_out.write('\n')
    f_out.close()
'''

def predict_extract_continously(sentences, feature_names, f_out, classifier): 
    nbr_sent_clcltd = 0.0  
    total = len(sentences)
    rows = []
    for sentence in sentences:
        # We pad the sentence to extract the context window more easily
        start = "BOS BOS BOS BOS\n"
        end = "\nEOS EOS EOS EOS"
        start *= w_size
        end *= w_size
        sentence = start + sentence
        sentence += end

        # Each sentence is a list of rows
        sentence = sentence.splitlines()
        padded_sentence = list()
        for line in sentence:
            line = line.split()
            padded_sentence.append(line)
        # print(padded_sentence)  
        
        # We extract the features and the classes
        # X contains a list of features, where each feature vector is a dictionary
        # y is the list of classes
        X = list()
        y = list()
        for i in range(len(padded_sentence) - 2 * w_size):
            # x is a row of X
            x = list()
            # The words in lower case
            for j in range(2 * w_size + 1):
                x.append(padded_sentence[i + j][0].lower())
            # The POS
            for j in range(2 * w_size + 1):
                x.append(padded_sentence[i + j][1])
            
            if i == 0:
                # old-school prediction, w/o help of previous chunks
                # Vectorize the test sentence and one hot encoding
                X_iter_one = vec.transform(dict(zip(feature_names[:-2], x)))
                # Predicts the chunks and returns numbers
                y_iter_one_predicted = classifier.predict(X_iter_one)
                # print(y_iter_one_predicted)

                # TODO
                # Converts to chunk names
                y_iter_one_predicted_symbols = [dict_classes[j] for j in y_iter_one_predicted]

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_one_predicted_symbols[i]
                
                # f_out.write(row + '\n')
                rows.append(row + '\n')
                # add to some kind of dict (mayhaps)
                
                padded_sentence[i+2].append(y_iter_one_predicted_symbols[i])
                # x.append('bos')
                # x.append('bos')
                
            elif i == 1:
                # half-old-school prediction, w/ help of one previously predicted chunk
                x.append(padded_sentence[i+2-1][3])
                # print(x)
                # Vectorize the test sentence and one hot encoding
                cut_feat_names = feature_names[:-2]
                cut_feat_names.append(feature_names[-1])
                # print(cut_feat_names)
                X_iter_two = vec.transform(dict(zip(cut_feat_names, x)))
                # Predicts the chunks and returns numbers
                y_iter_two_predicted = classifier.predict(X_iter_two)
                # print(y_iter_two_predicted)

                # TODO: feed back result into padded_sentence,
                #           this means we now have two predicted chunks and can start predicting new-school

                # Converts to chunk names
                y_iter_two_predicted_symbols = [dict_classes[j] for j in y_iter_two_predicted]
                # print(y_iter_two_predicted_symbols)

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_two_predicted_symbols[0]

                # f_out.write(row + '\n')
                rows.append(row + '\n')
                

                padded_sentence[i+2].append(y_iter_two_predicted_symbols[0])
                
                # x.append(y_iter_one_predicted_symbols[i]) 
            else:
            
                # predict new-school
                x.append(padded_sentence[i+2-2][3])
                x.append(padded_sentence[i+2-1][3])
                # Vectorize the test sentence and one hot encoding
                X_iter_n = vec.transform(dict(zip(feature_names, x)))
                # Predicts the chunks and returns numbers
                y_iter_n_predicted = classifier.predict(X_iter_n)
                # print(y_iter_n_predicted)

                # Converts to chunk names
                y_iter_n_predicted_symbols = [dict_classes[j] for j in y_iter_n_predicted]
                # print(y_iter_n_predicted_symbols)

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_n_predicted_symbols[0]
                # f_out.write(row + '\n')
                rows.append(row + '\n')
                # add to some kind of dict (mayhaps)
                
                padded_sentence[i+2].append(y_iter_n_predicted_symbols[0])
                # print(row)
                # print(padded_sentence)
        rows.append('\n')
        nbr_sent_clcltd += 1.0
        if (nbr_sent_clcltd % 50) == 0:
            print(nbr_sent_clcltd / total)
    for row in rows:
        f_out.write(row)
        # print(row)

if __name__ == '__main__':
    start_time = time.clock()
    train_corpus = './train.txt'
    test_corpus = './test.txt'
    w_size = 2  # The size of the context window to the left and right of the word
    feature_names = ['word_n2', 'word_n1', 'word', 'word_p1', 'word_p2',
                     'pos_n2', 'pos_n1', 'pos', 'pos_p1', 'pos_p2',
                     'chunk_n2', 'chunk_n1']
    training_start_time = time.clock()

    try:
        classifier = pickle.load( open("classifier.p", "rb"))
        vec = pickle.load( open("vec.p", "rb"))
        dict_classes = pickle.load( open("dict_classes.p", "rb"))
        inv_dict_classes = pickle.load( open("inv_dict_classes.p", "rb"))
    except:
        train_sentences = conll_reader.read_sentences(train_corpus)

        print("Extracting the features...")
        X_dict, y_symbols = extract_features(train_sentences, w_size, feature_names, training_phase = True)
        # for i in range(40):
        #     print(X_dict[i])
        #print(train_sentences[0] + train_sentences[1])
        
        print("Encoding the features and classes...")
        # Vectorize the feature matrix and carry out a one-hot encoding
        vec = DictVectorizer(sparse=True)
        # print(X_dict[55])
        X = vec.fit_transform(X_dict)
        # The statement below will swallow a considerable memory
        # X = vec.fit_transform(X_dict).toarray()
        # print(vec.get_feature_names())
        pickle.dump(vec, open("vec.p", "wb"))

        y, dict_classes, inv_dict_classes = encode_classes(y_symbols)
        pickle.dump(dict_classes, open("dict_classes.p", "wb"))
        pickle.dump(inv_dict_classes, open("inv_dict_classes.p", "wb"))

        training_start_time = time.clock()
        print("Training the model...")
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
        # classifier = tree.DecisionTreeClassifier()
        # classifier = linear_model.Perceptron(penalty='l2', n_jobs=2)
        model = classifier.fit(X, y)
        print(model)

        pickle.dump(classifier, open("classifier.p", "wb"))

    test_start_time = time.clock()
    # We apply the model to the test set
    test_sentences = list(conll_reader.read_sentences(test_corpus))
    # print(test_sentences[0])
    # Here we carry out a chunk tag prediction and we report the per tag error
    # This is done for the whole corpus without regard for the sentence structure
    print("Predicting the chunks in the test set...")
    f_out = open('out_own', 'w')
    predict_extract_continously(test_sentences, feature_names, f_out, classifier)
    print("Done!")

    '''
        X_test_dict, y_test_symbols = extract_features(test_sentences, w_size, feature_names, training_phase = False)
        # Vectorize the test set and one-hot encoding
        X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
        y_test = [inv_dict_classes[i] if i in y_symbols else 0 for i in y_test_symbols]
        y_test_predicted = classifier.predict(X_test)
        print("Classification report for classifier %s:\n%s\n"
            % (classifier, metrics.classification_report(y_test, y_test_predicted)))

        # Here we tag the test set and we save it.
        # This prediction is redundant with the piece of code above,
        # but we need to predict one sentence at a time to have the same
        # corpus structure
        # print("Predicting the test set...")
        # f_out = open('out', 'w')
        # predict(test_sentences, feature_names, f_out, classifier)
    '''

    end_time = time.clock()
    print("Training time:", (test_start_time - training_start_time) / 60)
    print("Test time:", (end_time - test_start_time) / 60)


    ''' 
        word   POS     gold chunk      IOB (Inside-Outside-Begin)
        [['BOS', 'BOS', 'BOS'], 
        ['BOS', 'BOS', 'BOS'], 
        ['It', 'PRP', 'B-NP'], 
        ['is', 'VBZ', 'B-VP'], 
        ['also', 'RB', 'I-VP'], 
        ['pulling', 'VBG', 'I-VP'], 
        ['20', 'CD', 'B-NP'], 
        ['people', 'NNS', 'I-NP'], 
        ['out', 'IN', 'B-PP'], 
        ['of', 'IN', 'B-PP'], 
        ['Puerto', 'NNP', 'B-NP'], 
        ['Rico', 'NNP', 'I-NP'], 
        [',', ',', 'O'], 
        ['who', 'WP', 'B-NP'], 
        ['were', 'VBD', 'B-VP'], 
        ['helping', 'VBG', 'I-VP'], 
        ['Huricane', 'NNP', 'B-NP'], 
        ['Hugo', 'NNP', 'I-NP'], 
        ['victims', 'NNS', 'I-NP'], 
        [',', ',', 'O'], 
        ['and', 'CC', 'O'], 
        ['sending', 'VBG', 'B-VP'], 
        ['them', 'PRP', 'B-NP'], 
        ['to', 'TO', 'B-PP'], 
        ['San', 'NNP', 'B-NP'], 
        ['Francisco', 'NNP', 'I-NP'], 
        ['instead', 'RB', 'B-ADVP'], 
        ['.', '.', 'O'], 
        ['EOS', 'EOS', 'EOS'], 
        ['EOS', 'EOS', 'EOS']]          
    '''                                                                                                           
