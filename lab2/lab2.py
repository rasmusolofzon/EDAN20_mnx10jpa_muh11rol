# python tokenize_simple.py < Selma.txt  | sort | uniq | wc -w

import sys
import regex as re

def tokenize(text):
	# words = re.findall(’\p{L}+’, text)
	words = re.findall('[A-Z].*?[\.!?]', text)
	words = [words[i][:-1].lower() for i in range(len(words))]
		#words[i] = words[i][:-1].lower()
		#print(type(word))
	return words


# if __name__ == '__main__':
text = open(sys.argv[1], 'r').read()
tokenized = tokenize(text)
print(tokenized)