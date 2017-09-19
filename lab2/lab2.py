# python tokenize_simple.py < Selma.txt  | sort | uniq | wc -w

import sys
import regex as re

def tokenize(text):
	words = re.findall('[A-Ö][\s\S]*?[\.!?]', text)
	words = [words[i][:-1].replace('\n', '').replace('\xad', '').lower() 
		for i in range(len(words))]
	return words

def xmlify(listofstrings):
	listofstrings = ["<s> " + listofstrings[i] + " </s>" for i in range(len(listofstrings))]
	return listofstrings

#TODO: "Estimate roughly the accuracy of your program."

# if __name__ == '__main__':
text = open(sys.argv[1], 'r', encoding='utf-8').read()
tokenized = tokenize(text)
print(xmlify(tokenized))

962203