__author__ = 'Leonardo Ranaldi'

from kerMIT.tree import Tree
from kerMIT.dtk import DT
from kerMIT import operation

class TreeKernel:

    def __init__(self, LAMBDA=1.):
        self.LAMBDA = LAMBDA

    def prodCompare(self, a, b):
        """

        :param a: Tree one
        :param b: Tree two
        :return:  Distance between a and b
        """
        if a.root != b.root:
            return False
        if len(a.children) != len(b.children):
            return False
        if (a.children is None) or (b.children is None):
            return False
        return all(x.root == y.root for (x,y) in zip(a.children, b.children))   #if all pairs of children are equal

    def delta(self, a, b):
        k = 0

        if (a.children is None) or (b.children is None):
            k = 0

        else:
            if len(a.children) == len(b.children):
                if len(a.children) == 1 and a.children[0].isTerminal() and b.children[0].isTerminal():
                    if a == b:
                        k = 1
                else:
                    if self.prodCompare(a,b):
                        k = 1
                        for child_a, child_b in zip(a.children, b.children):
                            k = k*(1 + self.LAMBDA*self.delta(child_a, child_b))

        return k

    def evaluate(self, a, b):
        sum = 0
        for aa in a.allNodes():
            for bb in b.allNodes():
                sum = sum + self.delta(aa, bb)
        return sum


if __name__ == "__main__":
    s1 = "(S (@S (NP (NP (@NP (DT The) (NN wait)) (NN time)) (PP (IN for) (NP (@NP (DT a) (JJ green)) (NN card)))) (VP (AUX has) (VP (@VP (VBN risen) (PP (IN from) (NP (@NP (NP (CD 21) (NNS months)) (TO to)) (NP (CD 33) (NNS months))))) (PP (IN in) (NP (@NP (DT those) (JJ same)) (NNS regions)))))) (. .))"
    s2 = "(@S)"


    t1 = Tree(string=s1)
    t2 = Tree(string=s2)

    tk = TreeKernel()

    dtk = DT(dimension = 8192)

    for tt in t1.allNodes():

        print (tt.root, tk.evaluate(tt, tt), dtk.kernel(tt, tt))


    print (tk.evaluate(t1, t2))
    print (dtk.kernel(t1, t2))

