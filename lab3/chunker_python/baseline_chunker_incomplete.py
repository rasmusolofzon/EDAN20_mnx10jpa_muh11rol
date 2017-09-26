# varför punkter och komman i listan?
# varför fel på första prediktionen?


import conll_reader

# Computes the part-of-speech distribution in a CoNLL 2000 file
def count_pos(corpus):
	pos_cnt = {}
	for sentence in corpus:
		for row in sentence:
			if row['pos'] in pos_cnt:
				pos_cnt[row['pos']] += 1
			else:
				pos_cnt[row['pos']] = 1
	return pos_cnt


def train(corpus):
	pos_cnt = count_pos(corpus)

	# Compute the chunk distribution by POS
	chunk_dist = {key: {} for key in pos_cnt.keys()}
	for sentence in corpus:
		for row in sentence:
			if row['chunk'] in chunk_dist[row['pos']]:
				chunk_dist[row['pos']][row ['chunk']] += 1
			else:
				chunk_dist[row['pos']][row ['chunk']] = 1
	
	# Determine the best association
	pos_chunk = {}
	for pos in chunk_dist:
		val = ''
		max = 0
		for chunk in chunk_dist[pos]:
			if chunk_dist[pos][chunk] > max:
				max = chunk_dist[pos][chunk]
				val = chunk
		pos_chunk[pos] = val

	return pos_chunk


def predict(model, corpus):
	# Predicts the chunk from the part of speech 
	for sentence in corpus:
		for row in sentence:
			row['pchunk'] = model[row['pos']]
	return corpus

# Evaluates the predicted chunk accuracy
def eval(predicted):
	word_cnt = 0
	correct = 0
	for sentence in predicted:
		for row in sentence:
			word_cnt += 1
			if row['chunk'] == row['pchunk']:
				correct += 1
	return correct / word_cnt


if __name__ == '__main__':

    column_names = ['form', 'pos', 'chunk']
    train_file = '../train.txt'
    test_file = '../test.txt'

    train_corpus = conll_reader.read_sentences(train_file)
    train_corpus = conll_reader.split_rows(train_corpus, column_names)
    test_corpus = conll_reader.read_sentences(test_file)
    test_corpus = conll_reader.split_rows(test_corpus, column_names)

    model = train(train_corpus)

    predicted = predict(model, test_corpus)
    accuracy = eval(predicted)
    print("Accuracy", accuracy)
    f_out = open('out', 'w')
    # We write the word (form), part of speech (pos),
    # gold-standard chunk (chunk), and predicted chunk (pchunk)
    for sentence in predicted:
        for row in sentence:
            f_out.write(row['form'] + ' ' + row['pos'] + ' ' +
                        row['chunk'] + ' ' + row['pchunk'] + '\n')
        f_out.write('\n')
    f_out.close()

	column_names = ['form', 'pos', 'chunk']
	train_file = '../train.txt'
	test_file = '../test.txt'

	train_corpus = conll_reader.read_sentences(train_file)
	train_corpus = conll_reader.split_rows(train_corpus, column_names)
	test_corpus = conll_reader.read_sentences(test_file)
	test_corpus = conll_reader.split_rows(test_corpus, column_names)

	model = train(train_corpus)

	predicted = predict(model, test_corpus)
	accuracy = eval(predicted)
	print("Accuracy", accuracy)
	f_out = open('out', 'w')

	# We write the word (form), part of speech (pos),
	# gold-standard chunk (chunk), and predicted chunk (pchunk)

	for sentence in predicted:
		for row in sentence:
			f_out.write(row['form'] + ' ' + row['pos'] + ' ' +
						row['chunk'] + ' ' + row['pchunk'] + '\n')
		f_out.write('\n')
	f_out.close()

