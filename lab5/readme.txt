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
		featX = (TOP-1)(POS+form)
			(sentence ordering)
		featY = ?
all: Boolean parameters,
	la = "can do left arc"
	ra = "can do right arc"

feature 	# of parameters
iteration
1		6
2		10
3		14
		(correct)

"This means that the purpose of this assignment 
	is to generate three scikit-learn models 
	for the labelled graphs."