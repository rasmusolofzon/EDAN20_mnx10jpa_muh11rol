"""
Returns the features in a dictionary format compatible with scikit-learn. 
(You have a code example of feature encoding in this format in the chunking program.)
"""

import transition

def extract(stack, queue, graph, feature_names, sentence):
    
    features = {}

    # the word form and the part-of-speech extracted from the first element in the stack
    if 'stack0_FORM' and 'stack0_POS' in feature_names:
        if len(stack) >= 1:
            features['stack0_FORM'] = stack[0]['form']
            features['stack0_POS'] = stack[0]['postag']
        else:
            features['stack0_FORM'] = 'nil'
            features['stack0_POS'] = 'nil'

    # the word form and the part-of-speech extracted from the first element in the queue
    if 'queue0_FORM' and 'queue0_POS' in feature_names:
        if len(queue) >= 1:
            features['queue0_FORM'] = queue[0]['form']
            features['queue0_POS'] = queue[0]['postag']
        else:
            features['queue0_FORM'] = 'nil'
            features['queue0_POS'] = 'nil'

    # constraints on the parser's actions: whether it can do 'left arc' and 'reduce'
    if 'can_leftarc' and 'can_reduce' in feature_names:
        features['can_leftarc'] = transition.can_leftarc(stack, graph)
        features['can_reduce'] = transition.can_reduce(stack, graph)
    
    # the word form and the part-of-speech extracted from the second element in the stack
    if 'stack1_FORM' and 'stack1_POS' in feature_names:
        if len(stack) >= 2:
            features['stack1_FORM'] = stack[1]['form']
            features['stack1_POS'] = stack[1]['postag']
        else:
            features['stack1_FORM'] = 'nil'
            features['stack1_POS'] = 'nil'

    # the word form and the part-of-speech extracted from the second element in the queue
    if 'queue1_FORM' and 'queue1_POS' in feature_names:
        if len(queue) >= 2:
            features['queue1_FORM'] = queue[1]['form']
            features['queue1_POS'] = queue[1]['postag']
        else:
            features['queue1_FORM'] = 'nil'
            features['queue1_POS'] = 'nil'
    
    
    if 'sent_stack0fw_FORM' and 'sent_stack0fw_POS' and 'sent_stack0ffw_FORM' and 'sent_stack0ffw_POS' and \
        'sent_stack0pw_FORM' and 'sent_stack0pw_POS' in feature_names:
        if len(stack) >= 1:
            id = int(stack[0]['id'])
            # the word form and the part-of-speech of the word following the top of the stack in the sentence order
            if len(sentence) >= id+2:
                features['sent_stack0fw_FORM'] = sentence[id+1]['form']
                features['sent_stack0fw_POS'] = sentence[id+1]['postag']
            else:
                features['sent_stack0fw_FORM'] = 'nil'
                features['sent_stack0fw_POS'] = 'nil'
            # the word form and the part-of-speech of the word two positions ahead of the top of the stack in the sentence order
            if len(sentence) >= id+3:
                features['sent_stack0ffw_FORM'] = sentence[id+2]['form']
                features['sent_stack0ffw_POS'] = sentence[id+2]['postag']
            else:
                features['sent_stack0ffw_FORM'] = 'nil'
                features['sent_stack0ffw_POS'] = 'nil'
            # the word form and the part-of-speech of the word preceding the top of the stack in the sentence order
            if id != 0:
                features['sent_stack0pw_FORM'] = sentence[id-1]['form']
                features['sent_stack0pw_POS'] = sentence[id-1]['postag']
            else:
                features['sent_stack0pw_FORM'] = 'nil'
                features['sent_stack0pw_POS'] = 'nil'
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
        #if len(features) < 16:
            #print(features)

        """
        print([features['stack0_POS'], features['stack1_POS'], features['stack0_FORM'], features['stack1_FORM'],
        features['queue0_POS'], features['queue1_POS'], features['queue0_FORM'], features['queue1_FORM'],
        features['can_reduce'], features['can_leftarc']])
        print('\n')
        """

    return features



if __name__ == '__main__':
    pass

