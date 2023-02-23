from kerMIT.structenc.dpte import partialTreeKernel
import kerMIT.operation as op
from kerMIT.tree_encode import parse
from stanfordcorenlp import StanfordCoreNLP
from kerMIT.tree import Tree

def getDST(pt_encoder, sentence):
    parsed_sentence = parse(sentence, nlp=nlp, annotator='depparse', tokens_as_leaves=False)
    print(parsed_sentence)
    tree = Tree(string=parsed_sentence)

    dt = pt_encoder.ds(tree, store_substructures=True)
    print(dt)
    return dt

DIMENSION = 8192
LAMBDA = 1.0
MU = 1.0

operation = op.fast_shuffled_convolution

nlp = StanfordCoreNLP('/stanford-corenlp-full-2018-10-05')
pt_encoder = partialTreeKernel(dimension=DIMENSION, LAMBDA=LAMBDA, MU=MU, operation=operation)

operation=op.fast_shuffled_convolution
s = "How do you know? All this is their information again."
dt = getDST(pt_encoder, s)


