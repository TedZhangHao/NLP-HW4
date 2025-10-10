# 1.
(a) I have chosen Berkeley Neural Parser (BNP) to have some experiments. BNP is the most widely-used English parser for short sentence, and it is based on the neural network, achieved SOTA on corpus like Penn Treebank.

The parse trees generated are not always binary trees as we using CNF in the class. For example, when parsing "the only one", BNP directly parses it into "DET JJ(adj) NN" instead of "NP -> Det NP" and "NP -> JJ NN". To be more specific, the parse tree generted by BNP are more flatter and less hierarchical, and it normally has more direct constituents, where each node might have more children. 

More surprisingly, for those sentences that are ambiguous, (e.g., prepositional attachment), BNP directly merge them together as in one rule, which enforce people to distinguish what exactly is the true structure by themselves.   

(b) 
(1) A special case is that if I ask it to parse a gramaticaly wrong sentence such as "I play the game eat some food", where "and" is ignored, and it can still parse this sentence and assign "VBP" (verb present) to "play" and "VB" to "eat", which means that BNP cannot detect that there is mistake within the sentence.
(2) BNP is not good at processing prepositional phrase attachment. For example, in the sentence "I saw the man in the park with a telescope", BNP choose to merge "saw" (VBD), "the man in the park" (NP), and "with a telescope" (PP) together, instead of assigning  PP to VP (VBD+NP) or to NP first.
(3) BNP also deals bad with coordination ambiguity such as "I saw the old man and woman". It cannot decide to assign "old" to "man and woman" or only "man".
(4) BNP is not dealing well with polysemy. For example, "The old man the boat", where "man" is the verb, can be parsed into 2 NPs.

However, BNP deals very well with complex structures such as nested clauses and long dependency. For example, in the sentence "The cat that the dog, which was barking loudly, chased, was fast", BNP successfully assigned "fast" with "the cat".

(c)
"The more you push, the less you take."
BNP failed parse "the less you take" ("the more", "the less") as an unknown phrase.