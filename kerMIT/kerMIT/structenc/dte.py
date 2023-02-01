__author__ = 'Fabio Massimo Zanzotto, Lorenzo Ferrone, Leonardo Ranaldi'

# TO DO.................rivedere bene delle cose con prof. ZNZ8


import numpy as np
import time
import hashlib
import gc

from kerMIT.tree import Tree
from kerMIT.structenc.dse import DSE
#import operation as op
from kerMIT import operation as op
#from semantic_vector import SemanticVector


class DTE(DSE):
    """The Distributed Tree Encoder is applied to trees
    and represents trees in the space of subtrees as defined
    in `Duffy\&Collins (2001)`_
    by means of reduced vectors. It has been firstly implemented
    in the `Distributed Tree Kernels, ICML 2012 Zanzotto\&Dell'Arciprete`_.

    Let define trees as the root label :math:`r` and the list of its subtrees:

    .. math:: \\sigma = (r,[t_1,...,t_k])

    where :math:`t_i` are trees. Leaves have an empty list of subtrees.
    Terms of Equation :eq:`main` are then the following:


    1. :math:`\\vec{\\sigma} = \\Phi(\\sigma) =
    \\begin{cases} \\vec{r} & \\text{if } \\sigma \\text{ is a leaf}\\\\ (\\vec{r} \\circ (\\Phi(t_1) \\circ \ldots \\circ \\Phi(t_n)))
    & \\text{otherwise}\\end{cases}` where :math:`\\vec{r}` is the vector associated to the label of :math:`r` and :math:`\circ` is the chosen vector composition operation;

    2. :math:`\\omega_\\sigma = \\sqrt \\lambda^n` where :math:`\\lambda` is the factor penalizing large subtrees
    and :math:`n` is the number  of nodes of the subtree :math:`\\sigma`

    3. :math:`S(t)` is the set containing the relevant subtrees of :math:`t` where a relevant subtree :math:`\\sigma` of :math:`t`
    is any subtree rooted in any node of :math:`t` with the only constraint that, if a node of  is contained in the subtree,
    all the siblings of this node in the initial tree have to be in the subtree

    To keep things simple, terms of Equation :eq:`main` are also presented with a running example.
    Given a tree in the classical parenthetical form:

    ``(NP (DT the) (JJ) (NN ball))``

    :math:`\\vec{\\sigma}` is obtained by transforming node labels in the related
    vectors and  by putting the operation :math:`\\circ` instead of blanks:

    .. math::    \\vec{\\sigma} = (\\vec{NP} \\circ (\\vec{DT} \\circ \\vec{the}) \\circ (\\vec{JJ})\\circ (\\vec{NN} \\circ \\vec{ball}))

    This is an example of construction of the :math:`S(t)` for a very simple tree (taken from Moschitti's personal page in Trento):
.. figure:: /_static/_img/TK_Moschitti.gif
   :scale: 50 %
   :name: fig-tk
   :target: ../../_static/_img/TK_Moschitti.gif
   :align: center
   :alt: Tree Kernel Space



.. _Distributed Tree Kernels, ICML 2012 Zanzotto\&Dell'Arciprete: https://dl.acm.org/doi/10.5555/3042573.3042592
.. _Duffy\&Collins (2001): https://dl.acm.org/doi/10.5555/2980539.2980621


    :param LAMBDA: the penalizing factor :math:`\lambda` for large subtrees
    :param dimension: the dimension :math:`d` of the embedding space :math:`R^d`
    :param operation: the basic operation :math:`\circ` for composing two vectors
    """

    def __init__(self, LAMBDA = 1., dimension=4096, operation=op.fast_shuffled_convolution):
        self.LAMBDA = LAMBDA
        self.dimension = dimension
        self.operation = operation
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.random_cache = {}
        self.result = np.zeros(self.dimension)
        self.spectrum = np.zeros(self.dimension)

    def cleanCache(self):
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.random_cache = {}
        gc.collect()

    def distributedVector(self, s):
        if s in self.random_cache:
            return self.random_cache[s]
        # h = int(hashlib.md5(s.encode()).hexdigest(),16) % 100000000              #probably too slow and not necessary ??
        # h = abs(mmh3.hash(s)) % 1000000

        h = abs(op.hash(s)) % 4294967295

        # h = np.abs(hash(s))         #devo hashare s in qualche modo (controllare che basti) e
        np.random.seed(h)            #inizializzare np.random.seed()
        v_ = op.random_vector(self.dimension,normalized=False)
        self.random_cache[s] = v_
        return v_



    def sRecursive(self, tree):
        result = np.zeros(self.dimension)
        if tree.isPreTerminal():
            result = np.sqrt(self.LAMBDA)*self.operation(self.operation(self.distributedVector(tree.root),
                                                                        self.distributedVector("separator")),
                                    self.distributedVector(tree.children[0].root))
        elif not tree.isTerminal():
            rootVector = self.distributedVector(tree.root)
            separator = self.distributedVector("separator")
            result = self.operation(rootVector, separator)
            for child in tree.children:
                vecChildren = (np.sqrt(self.LAMBDA)*self.distributedVector(child.root) + self.sRecursive(child))
                result = self.operation(result, vecChildren)
            result = np.sqrt(self.LAMBDA) * result
        self.spectrum = self.spectrum + result

        return result

    def ds(self, tree):
        """This general method is implemented by dt(tree)
        """
        return self.dt(tree)

    def dt(self, tree):
        """takes as input a tree and produces

        :param tree:
        :return:
        """
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum

    def dsf_with_weight(self, tree):
        return self.dtf_and_weight(tree)

    def dsf(self, tree):
        return self.dtf(tree)

    def substructures(self, tree):
        return self.subtrees(tree)

    def subtrees(self, tree):
        """

        :param tree: tree in parienthetic form
        :return: all subtrees
        """
        setOfTrees = []
        setOfSubTrees = []
        if tree.isPreTerminal():
            tree.children[0].wasTerminal = True
            setOfTrees.append(tree)
            setOfSubTrees.append(tree)
        else:
            baseChildrenList = [[]]
            for child in tree.children:
                newBaseChildrenList = []
                c_setOfTrees, c_setOfSubtrees = DTE.subtrees(self,child)
                setOfSubTrees = setOfSubTrees + c_setOfSubtrees
                for treeSubChild in c_setOfTrees:
                   for treeSub in baseChildrenList:
                       newBaseChildrenList.append(treeSub + [treeSubChild])
                baseChildrenList = newBaseChildrenList
            for children in baseChildrenList:
                newTree = Tree(root=tree.root,children=children, id=tree.id())
                setOfTrees.append(newTree)
                setOfSubTrees.append(newTree)

        setOfTrees.append(Tree(root=tree.root, id=tree.id()))
        return setOfTrees,setOfSubTrees


    def dtf_and_weight(self, tree):
        """given a tree :math:`\\sigma` produces the pair :math:`(\\vec{\\sigma},\\omega_\\sigma)`:

            1. :math:`\\vec{\\sigma} = \\Phi(\\sigma) = \\begin{cases} \\vec{r} & \\text{if } \\sigma \\text{ is a leaf}\\\\ (\\vec{r} \\circ (\\Phi(t_1) \\circ \ldots \\circ \\Phi(t_n)))  & \\text{otherwise}\\end{cases}` where :math:`\\vec{r}` is the vector associated to the label of :math:`r` and :math:`\circ`

            2. :math:`\\omega_\\sigma = \\sqrt \\lambda^n` where :math:`\\lambda` is the factor penalizing large subtrees and :math:`n` is the number  of nodes of the subtree :math:`\\sigma`

         Examples in the general description of the class.

        :param tree: :math:`\\sigma`, which is a Tree
        :return: :math:`(\\vec{\\sigma},\\omega_\\sigma)`, which is a vector (Array or Tensor) and a scalar
        """
        if tree.isTerminal():
            penalizing_value = 1
            if not tree.wasTerminal:
                penalizing_value = np.sqrt(self.LAMBDA)
            self.dtf_cache[tree] = (self.distributedVector(tree.root),penalizing_value)
        elif tree in self.dtf_cache:
            return self.dtf_cache[tree]
        else:
            penalizing_value = np.sqrt(self.LAMBDA)
            vec = self.distributedVector(tree.root)
            separator = self.distributedVector("separator")
            vec = self.operation(vec,separator)
            for c in tree.children:

                (vecChildren,penalizing_value_in) = self.dtf_and_weight(c)
                vec = self.operation(vec, vecChildren)
                penalizing_value *= penalizing_value_in
            self.dtf_cache[tree] = (vec,penalizing_value)
        return self.dtf_cache[tree]

    def dtf(self, tree):
        return self.dtf_and_weight(tree)[0]

if __name__ == "__main__":

    ss = "(S (@S (NP (NP (@NP (DT The) (NN wait)) (NN time)) (PP (IN for) (NP (@NP (DT a) (JJ green)) (NN card)))) (VP (AUX has) (VP (@VP (VBN risen) (PP (IN from) (NP (@NP (NP (CD 21) (NNS months)) (TO to)) (NP (CD 33) (NNS months))))) (PP (IN in) (NP (@NP (DT those) (JJ same)) (NNS regions)))))) (. .))"
    ss = '(NOTYPE##ROOT(NOTYPE##NP(NOTYPE##S(NOTYPE##NP(NOTYPE##NNP(NOTYPE##Children)))(NOTYPE##VP-REL(NOTYPE##VBG-REL(NOTYPE##W))(NOTYPE##CC(NOTYPE##and))(NOTYPE##VBG(NOTYPE##waving))(NOTYPE##PP(NOTYPE##IN(NOTYPE##W))(NOTYPE##NP(NOTYPE##NN(NOTYPE##camera))))))))'
    ss = ss.replace(")", ") ").replace("(", " (")

    t = Tree(string=ss)



    kernel = partialTreeKernel(dimension=8192, LAMBDA= 0.6, operation=op.fast_shuffled_convolution)

    v = kernel.sRecursive(t)
    w = kernel.dt(t)

    print (v)
    print (w)


