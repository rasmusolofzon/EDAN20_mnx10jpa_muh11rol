"""
Returns the features in a dictionary format compatible with scikit-learn. 
(You have a code example of feature encoding in this format in the chunking program.)
"""

import transition

def extract(stack, queue, graph, feature_names, sentence):

    #print(feature_names)
    
    features = {}

    # the word form and the part-of-speech extracted from the first element in the stack
    if len(stack) >= 1:
        features['stack0_FORM'] = stack[0]['form']
        features['stack0_POS'] = stack[0]['postag']
    else:
        features['stack0_FORM'] = 'nil'
        features['stack0_POS'] = 'nil'

    # the word form and the part-of-speech extracted from the first element in the queue
    if len(queue) >= 1:
        features['queue0_FORM'] = queue[0]['form']
        features['queue0_POS'] = queue[0]['postag']
    else:
        features['queue0_FORM'] = 'nil'
        features['queue0_POS'] = 'nil'

    # constraints on the parser's actions: whether it can do 'left arc' and 'reduce'
    features['can_leftarc'] = transition.can_leftarc(stack, graph)
    features['can_reduce'] = transition.can_reduce(stack, graph)
    

    # Only for the second feature set
    if len(feature_names) >= 10:

        # the word form and the part-of-speech extracted from the second element in the stack
        if len(stack) >= 2:
            features['stack1_FORM'] = stack[1]['form']
            features['stack1_POS'] = stack[1]['postag']
        else:
            features['stack1_FORM'] = 'nil'
            features['stack1_POS'] = 'nil'

        # the word form and the part-of-speech extracted from the second element in the queue
        if len(queue) >= 2:
            features['queue1_FORM'] = queue[1]['form']
            features['queue1_POS'] = queue[1]['postag']
        else:
            features['queue1_FORM'] = 'nil'
            features['queue1_POS'] = 'nil'

        """
        print([features['stack0_POS'], features['stack1_POS'], features['stack0_FORM'], features['stack1_FORM'],
        features['queue0_POS'], features['queue1_POS'], features['queue0_FORM'], features['queue1_FORM'],
        features['can_reduce'], features['can_leftarc']])
        print('\n')
        """

    # Only for the third feature set
    if len(feature_names) >= 16:
        
        # the word form and the part-of-speech of the word following the top of the stack in the sentence order
        if len(stack) >= 1:
            for i in range(len(sentence)):
                if sentence[i]['form'] == stack[0]['form']:
                    


                    if len(sentence) >= i+2:
                        features['sent_stack0fw_FORM'] = sentence[i+1]['form']
                        features['sent_stack0fw_POS'] = sentence[i+1]['postag']
                    else:
                        #varför sker detta aldrig???
                        features['sent_stack0fw_FORM'] = 'nil'
                        features['sent_stack0fw_POS'] = 'nil'
                    
                    if len(sentence) >= i+3:
                        features['sent_stack0ffw_FORM'] = sentence[i+2]['form']
                        features['sent_stack0ffw_POS'] = sentence[i+2]['postag']
                    else:
                        # Detta sker!!
                        features['sent_stack0ffw_FORM'] = 'nil'
                        features['sent_stack0ffw_POS'] = 'nil'
                    
                    if i != 0:
                        features['sent_stack0pw_FORM'] = sentence[i-1]['form']
                        features['sent_stack0pw_POS'] = sentence[i-1]['postag']
                    else:
                        #varför sker detta samtidigt som det funkar utan if-satsen ovan?
                        features['sent_stack0pw_FORM'] = 'nil'
                        features['sent_stack0pw_POS'] = 'nil'
                    break
        else:
            features['sent_stack0fw_FORM'] = 'nil'
            features['sent_stack0fw_POS'] = 'nil'
            features['sent_stack0ffw_FORM'] = 'nil'
            features['sent_stack0ffw_POS'] = 'nil'
            features['sent_stack0pw_FORM'] = 'nil'
            features['sent_stack0pw_POS'] = 'nil'
        



        """

        # the part-of-speech of the head of the second value on the stack
        if len(stack) >= 2:
            for i in range(len(sentence)):
                if sentence[i]['id'] == stack[1]['head']:
                    features['sent_stack1h_POS'] = sentence[i]['postag']
                    break
        else: 
            features['sent_stack1h_POS'] = 'nil'
            
        # the word form of the right sibling of the second value on the stack 
        if len(stack) >= 2:
            for i in range(len(sentence)):
                if sentence[i]['id'] > stack[1]['id'] and sentence[i]['head'] == stack[1]['head']:
                    features['sent_stack1rs_FORM'] = sentence[i]['form']
                    break
                else:
                    features['sent_stack1rs_FORM'] = 'nil'        
        else: 
            features['sent_stack1rs_FORM'] = 'nil'
        
        """

        #print(features)
        if len(features) < 16:
            print(features)

    return features




if __name__ == '__main__':
    pass


"""   
# the word form and the part-of-speech extracted from the first element in the stack
'stack0_FORM'
'stack0_POS'

# the word form and the part-of-speech extracted from the first element in the queue
'queue0_FORM'
'queue0_POS'

# the word form and the part-of-speech extracted from the second element in the stack
'stack1_FORM'
'stack1_POS'

# the word form and the part-of-speech extracted from the second element in the queue
'queue1_FORM'
'queue1_POS'

# constraints on the parser's actions: whether it can do 'left arc' and 'reduce'
'can_leftarc'
'can_reduce'

# the word form and the part-of-speech of the word following the top of the stack in the sentence order
'sent_stack0fw_FORM'
'sent_stack0fw_POS'

# the part-of-speech of the head of the second value on the stack
'sent_stack1h_POS'

# the word form of the right sibling of the second value on the stack 
'sent_stack1rs_FORM'
"""


"""
feature_names_1 = ['stack0_FORM', 'pos_1_STACK', 'w_1_QUEUE', 'pos_1_QUEUE', 'can_left_arc', 'can_reduce']
feature_names_2 = feature_names_1 + ['w_2_STACK', 'pos_2_STACK','w_2_QUEUE', 'pos_2_QUEUE']
feature_names_3 = feature_names_2 + ['w_TOP_plus_1', 'pos_TOP_plus_1', 'pos_1_STACK_h', 'lex_1_STACK_rs']

#POS STACK 1 h, LEX STACK 1 rs (good performance in http://www.aclweb.org/anthology/C/C10/C10-1093.pdf )
#word['head'] where sentence['id'] == id[TOP_form]+1
#POS STACK 1 h = the part-of-speech of the head of the second value on the stack
#LEX STACK 1 rs = the lexical value of the right sibling of the second value on the stack 
#http://maltparser.org/userguide.html
"""

"""
for sentence in corpus:
    create empty stack
    put all words from sentence in queue
    create empty graph (to fill in)
    dictCompatibleWithScikit = extract(stack, queue, graph, feature_1, sentence)
    listWithAllFeatureVectors.append(dictCompatibleWithScikit)

logisticRegClassifier
logisticRegClassifier.fit(listWithAllFeatureVectors)

yield success
"""

# Generate the three scikit-learn models using the code models from the chunking labs. 
# You will evaluate the model accuracies (not the parsing accuracy) using 
# the classification report produced by scikit-learn and the correctly classified instances 