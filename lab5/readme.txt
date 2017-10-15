projective/non-projective sentence?

Feature sets
	1. (TOP, FIRST)
	2. (w_1top, POS_1top, w_2top, POS_2top
		w_1first, POS_1first, 
			w_2first, POS_2first)
	3. (2, featX, featY)

Encode constraints as features? 
	(Hint: Boolean features)

Extracting features
-------------------

1. w_1top, POS_1top, w_2top, POS_2top
2. (w_1top, POS_1top, w_2top, POS_2top
	w_1first, POS_1first, 
		w_2first, POS_2first)
3. (2, featX, featY)
	where 
		featX = (TOP-1(sentence ordering))(POS+form)
		featY = ?
all: Boolean parameters,
	la = "can do left arc"
	ra = "can do right arc"

feature 	# of parameters
iteration
1		6
2		10
3		14

"This means that the purpose of this assignment 
	is to generate three scikit-learn models 
	for the labelled graphs."



Report
------


Nivre's parser
--------------
Given a manually-annotated dependency graph, what are the conditions on the stack and the current input list -- the queue -- to execute left-arc, right-arc, shift, or reduce? Start with left-arc and right-arc, which are the simplest ones.

execute right-arc:
- the stack and the queue (input vector) must not be empty
execute left-arc: 
- the stack and the queue (input vector) must not be empty
- the word on the top of the stack must not already have a head, i.e. there must not be an arc from another word to the word on the top of the stack
execute reduce: 
- the stack must not be empty and the word on the top of the stack must have a head, i.e. there must be an arc from a word to the word on the top of the stack
execute shift: 
- the queue (input vector) must not be empty 


The parser can only deal with projective sentences. In the case of a nonprojective one, the parsed graph and the manually-annotated sentence are not equal. Examine one such sentence and explain why it is not projective. Take a short one (the shortest).

The sentence is non-projective as it contains a non-projective link, i.e. a link not only separated by direct or indirect dependents of 'Head' and 'Dep'. In this case the non-projective link is the one between the word "behöver" and the question mark. In the graph this is visually illustrated as two arcs crossing each other.

Nivre's parser sets constraints to actions. Name a way to encode these constraints as features. Think of Boolean features.