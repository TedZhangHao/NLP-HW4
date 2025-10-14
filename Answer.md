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

# 2.
(a) 
[1] Dependency grammar does not generate a syntax tree, instead, it describes the structure of sentence by using the dependant relationship between each words. Every token is directly connected with its corresponding head token. There is no context-free nonterminals such as NP or VP. The syntax information is only represented by the dependency arcs. For example, nsubj represents the noun subject, and dobj represents direct object.
[2] Link grammar parser constructs a syntactic structure represented as links between pairs of words. Those links are named with the relationship btween words. For example, MV connects verbs (and adjectives) to modifying phrases like adverbs, prepositianl pjrases, time expressions, etc. By checking 'Show all linkages', all the possible linkages (syntactic structures) are shown, By checking 'Show constituent tree', it can generate trees (represented by parenthesis) as we saw in the HW1. However, this parser is not good at propernouns. If I input 'Papa ate the caviar with a spoon', it falls in 'No complete linkagesn found'.
[3] The link provided is not accessible, and there is no relative online demo. However, I have read many relative documents to have a thorough view of this parser. As opposed to dependency grammar, Head-driven phrase structure grammar (HPSG) focuses more on lexicon details and defines phrase structures by a set of rules that describe how different types of heads combine with their complements and specifiers. A central innovation in HPSG is its use of feature structures to represent syntactic, semantic, and morphological information. Attribute-value matrices encode intricate linguistic information as features. HPSG also uses unification, a process in which different feature structures are merged under constraint that all relevant syntactic and semantic information is consistently maintained.  
[4] Combinatory Category Grammar (CCG) is an efficient parable and linguistically expressive grammar fomalism, which generates consistency-based structures (as opposed to a dependency grammar). It is not like traditional phrase structure grammar using rules, instead, CCG assumes every word contains information of how to be combined with other words. Each word is defined by a categroy in CCG, which specifies its grammar type and what kind of adjacent words can be used to integrate a larger structure. When using CCG provided in https://github.com/chrzyki/candc, an example of sentence 'The government plans to raise the tax.' is as follows. 
`(<T S[dcl] 1 2> (<T S[dcl] 0 2> (<T NP 0 1> (<T N 0 2> (<L N/N NP[nb]/N NP[nb]/N The N/N>) (<L N N/N N/N government N>))) (<T S[dcl]\NP 1 2> (<L (S[dcl]\NP)/(S[to]\NP) N N plans (S[dcl]\NP)/(S[to]\NP)>) (<T S[to]\NP 1 2> (<L (S[to]\NP)/(S[b]\NP) (NP\NP)/NP (NP\NP)/NP to (S[to]\NP)/(S[b]\NP)>) (<T S[b]\NP 1 2> (<L (S[b]\NP)/NP N/N N/N raise (S[b]\NP)/NP>) (<T NP 1 1> (<T N 0 2> (<L N/N N/N N/N income N/N>) (<L N N N tax N>))))))) (<L . . . . .>))`
Specification:
`<T Category start end>` nonterminal node (tree), category of combination, start and end position.
`<L ....>` terminal node (leaf)
`S[dcl]` a decalritive sentence
`S[to]` infinite "to do sth."
`S[b]` verb phrase
`/` forward-looking A/B B->A `The|NP/N     government|N   →   NP`, which means that 'The' requires a Noun on the right to become a NP.
`\` backword-looking A\B A->B `plans|(S[dcl]\NP)/NP   income tax|NP   →   S[dcl]\N`, which means that 'plans' needs a NP on the left to form a declaritive sentence. 
By applying the functions, words can be combined to construct a larger structre. However, the result is complicated due to the format.

(b)
I experiment Denpendency grammer with Chinese and find out that it still only shows the relationship between pairs of preterminals. However, in the context of Chinese, dependent grammar sometimes assigns two words with only one preterminal, which is contrast with the version of parsing English sentences. For example, it assigns VERB to '觉得', which could be translated as 'think'. This is rational because '觉得' itself is a disyllabic verb. However, this could cause misleading. For instance, I simply test it with a question sentence 'do you think it is proper of you doing this?', which is translated as '你觉得你这么做好吗'. It assigns '做好' with VERB, but '做好' is actually not a disullabic verb. Instead, the sentence should be seperated syntactically as '你觉得你这么做' (do you think that you doing this) and '好吗' (is proper), which mean '做好' is VERB+adj instead of a verb like '觉得'. What's more, when I test with '你最近怎么样' (how are you recently), where '最近' (recently) is ADV, '怎么样' (semantically similar to 'how') is ADJP, and an auxiliary verb is omitted orally. However, parser assign '最近' as NOUN and '怎么样' as ADV, which are not correct. Based on these cases, the parses are not very accurate as it has done on English sentences.