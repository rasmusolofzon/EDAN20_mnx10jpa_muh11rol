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

def extract_features(sentences, w_size, feature_names):
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
        X, y = extract_features_sent(sentence, w_size, feature_names)
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l


def extract_features_sent(sentence, w_size, feature_names):
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
        
        # The preceding 'predicted' chunks
        for j in range(w_size):
            x.append(padded_sentence[i + j][2])
               
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
                # Converts to chunk names
                y_iter_one_predicted_symbols = [dict_classes[j] for j in y_iter_one_predicted]

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_one_predicted_symbols[i]
                rows.append(row + '\n')
                
                padded_sentence[i+2].append(y_iter_one_predicted_symbols[i])
            elif i == 1:
                # half-old-school prediction, w/ help of one previously predicted chunk
                x.append(padded_sentence[i+2-1][3])
                # Vectorize the test sentence and one hot encoding
                cut_feat_names = feature_names[:-2]
                cut_feat_names.append(feature_names[-1])
                # print(cut_feat_names)
                X_iter_two = vec.transform(dict(zip(cut_feat_names, x)))
                # Predicts the chunks and returns numbers
                y_iter_two_predicted = classifier.predict(X_iter_two)
                # Converts to chunk names
                y_iter_two_predicted_symbols = [dict_classes[j] for j in y_iter_two_predicted]

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_two_predicted_symbols[0]

                rows.append(row + '\n')

                padded_sentence[i+2].append(y_iter_two_predicted_symbols[0])
            else:
                # predict new-school
                x.append(padded_sentence[i+2-2][3])
                x.append(padded_sentence[i+2-1][3])
                # Vectorize the test sentence and one hot encoding
                X_iter_n = vec.transform(dict(zip(feature_names, x)))
                # Predicts the chunks and returns numbers
                y_iter_n_predicted = classifier.predict(X_iter_n)

                # Converts to chunk names
                y_iter_n_predicted_symbols = [dict_classes[j] for j in y_iter_n_predicted]

                row = ""
                for j in range(len(padded_sentence[i+2])):
                    row += padded_sentence[i+2][j] + ' '
                row += y_iter_n_predicted_symbols[0]
                rows.append(row + '\n')
                
                padded_sentence[i+2].append(y_iter_n_predicted_symbols[0])
        rows.append('\n')
        nbr_sent_clcltd += 1.0
        if (nbr_sent_clcltd % 50) == 0:
            print(nbr_sent_clcltd / total)
    for row in rows:
        f_out.write(row)

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
        X_dict, y_symbols = extract_features(train_sentences, w_size, feature_names)
        
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
    
    # Here we carry out a chunk tag prediction and we report the per tag error
    # This is done for the whole corpus without regard for the sentence structure
    print("Predicting the chunks in the test set...")
    f_out = open('out_own', 'w')
    predict_extract_continously(test_sentences, feature_names, f_out, classifier)
    print("Done!")

    end_time = time.clock()
    print("Training time:", (test_start_time - training_start_time) / 60)
    print("Test time:", (end_time - test_start_time) / 60)
