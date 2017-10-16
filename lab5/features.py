"""
Returns the features in a dictionary format compatible with scikit-learn. 
(You have a code example of feature encoding in this format in the chunking program.)
"""

import transition

def extract(stack, queue, graph, feature_names, sentence):

    #print(feature_names)
    
    features = {}
    
    # the word and the part of speech extracted from the first element in the stack
    if len(stack) >= 1:
        features['w_1_STACK'] = stack[0]['form']
        features['pos_1_STACK'] = stack[0]['postag']
    else:
        features['w_1_STACK'] = 'nil'
        features['pos_1_STACK'] = 'nil'
    # the word and the part of speech extracted from the first element in the queue
    if len(queue) >= 1:
        features['w_1_QUEUE'] = queue[0]['form']
        features['pos_1_QUEUE'] = queue[0]['postag']
    else:
        features['w_1_QUEUE'] = 'nil'
        features['pos_1_QUEUE'] = 'nil'
    # constraints on the parser's actions: "can do left arc" and "can do reduce"
    features['can_left_arc'] = transition.can_leftarc(stack, graph)
    features['can_reduce'] = transition.can_reduce(stack, graph)
    
    # Only for the second feature set
    if len(feature_names) >= 10:
        # the word and the part of speech extracted from the second element in the stack
        if len(stack) >= 2:
            features['w_2_STACK'] = stack[1]['form']
            features['pos_2_STACK'] = stack[1]['postag']
        else:
            features['w_2_STACK'] = 'nil'
            features['pos_2_STACK'] = 'nil'
        # the word and the part of speech extracted from the second element in the queue
        if len(queue) >= 2:
            features['w_2_QUEUE'] = queue[1]['form']
            features['pos_2_QUEUE'] = queue[1]['postag']
        else:
            features['w_2_QUEUE'] = 'nil'
            features['pos_2_QUEUE'] = 'nil'

    # Only for the third feature set
    if len(feature_names) >= 14:
        # the part of speech and the word form of the word following the top of the stack in the sentence order
        if len(stack) >= 1:
            for i in range(len(sentence)):
                if sentence[i]['form'] == stack[0]['form']:
                    if len(sentence) >= i+2:
                        features['w_TOP_plus_1'] = sentence[i+1]['form']
                        features['pos_TOP_plus_1'] = sentence[i+1]['postag']
                    else: 
                        features['w_TOP_plus_1'] = 'nil'
                        features['pos_TOP_plus_1'] = 'nil'
                    break
        else:
            features['w_TOP_plus_1'] = 'nil'
            features['pos_TOP_plus_1'] = 'nil'
        # the part-of-speech of the head of the second value on the stack
        if len(stack) >= 2:
            for i in range(len(sentence)):
                if sentence[i]['id'] == stack[1]['head']:
                    features['pos_1_STACK_h'] = sentence[i]['postag']
                    break
        else: 
            features['pos_1_STACK_h'] = 'nil'
            
        # the word (lexical value) of the right sibling of the second value on the stack 
        if len(stack) >= 2:
            for i in range(len(sentence)):
                if sentence[i]['head'] == stack[1]['head'] and sentence[i]['form'] != stack[1]['form']:
                    features['lex_1_STACK_rs'] = sentence[i]['form']
                    break
                else:
                    features['lex_1_STACK_rs'] = 'nil'        
        else: 
            features['lex_1_STACK_rs'] = 'nil'
        

        #print(features)
        #if len(features) < 14:
            #print(features)

    """
    print([features['pos_1_STACK'], features['pos_2_STACK'], features['w_1_STACK'], features['w_2_STACK'],
            features['pos_1_QUEUE'], features['pos_2_QUEUE'], features['w_1_QUEUE'], features['w_2_QUEUE'],
            features['can_reduce'], features['can_left_arc']])
    print('\n')
    """

    return features


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
    pass


    #features_1 = ['w_1_STACK', 'pos_1_STACK', 'w_1_QUEUE', 'pos_1_QUEUE', 'can_left_arc', 'can_reduce']
    #features_2 = features_1 + ['w_2_STACK', 'pos_2_STACK','w_2_QUEUE', 'pos_2_QUEUE']
                
    '''
        ['w_1_STACK', 'pos_1_STACK', 'w_2_STACK', 'pos_2_STACK', 
        'w_1_QUEUE', 'pos_1_QUEUE', 'w_2_QUEUE', 'pos_2_QUEUE',
        'can_left_arc', 'can_reduce']
    '''

    #features_3 = features_2 + ['w_TOP_plus_1', 'pos_TOP_plus_1', 'pos_1_STACK_h', 'lex_1_STACK_rs']
                
    '''
        ['w_1_STACK', 'pos_1_STACK', 'w_2_STACK', 'pos_2_STACK', 
        'w_1_QUEUE', 'pos_1_QUEUE', 'w_2_QUEUE', 'pos_2_QUEUE',
        'can_left_arc', 'can_reduce', 
        'w_', 'pos_', 
        'pos_1_STACK_h', 'lex_1_STACK_rs']
    '''
     #POS STACK 1 h, LEX STACK 1 rs (good performance in http://www.aclweb.org/anthology/C/C10/C10-1093.pdf )
                #word['head'] where sentence['id'] == id[TOP_form]+1
                #POS STACK 1 h = the part-of-speech of the head of the second value on the stack
                #LEX STACK 1 rs = the lexical value of the right sibling of the second value on the stack 
                #http://maltparser.org/userguide.html



"""
            if __name__ == '__main__':


                features = "poop"

                X_l = []
                for sentence in sentences:
                    X, y = extract_features_sent(sentence, w_size, feature_names)
                    X_l.extend(X)
                    y_l.extend(y)
                    return X_l, y_l
                    """

'''
...
for sentence in corpus:
    create empty stack
    put all words from sentence in queue
    create empty graph (to fill in)
    dictCompatibleWithScikit = extract(stack, queue, graph, feature_1, sentence)
    listWithAllFeatureVectors.append(dictCompatibleWithScikit)

logisticRegClassifier
logisticRegClassifier.fit(listWithAllFeatureVectors)

yield success
...
'''
#Generate the three scikit-learn models using the code models from the chunking labs. 


# You will evaluate the model accuracies (not the parsing accuracy) using 
# the classification report produced by scikit-learn and the correctly classified instances 