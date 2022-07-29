from kerMIT.structenc.dse import DSE
from kerMIT.structenc.dte import DTE
import kerMIT.operation as op
from kerMIT.tree import Tree
import numpy as np


class partialTreeKernel(DTE):
    def __init__(self, LAMBDA=1., MU=1., dimension=4096, operation=op.fast_shuffled_convolution, terminal_factor=1.):
        super(partialTreeKernel, self).__init__(LAMBDA=LAMBDA, dimension=dimension, operation=operation)
        print('using partial')

        self.spectrum = np.zeros(self.dimension)

        self.__set_terminal_factor(terminal_factor)
        # TODO perche' MAXPOWER non e' parametrico? anche su java e' 10
        self.__set_mu(MU, MAX_POWER=10)

    def __set_terminal_factor(self, terminal_factor=1.0):
        self._terminal_factor = np.sqrt(terminal_factor)

    def terminal_factor(self) -> float:
        return self._terminal_factor

    def __set_mu(self, MU: float, MAX_POWER: int):
        self._MAX_POWER = MAX_POWER
        self._mu = MU
        self._mus = [self._mu ** i for i in range(self._MAX_POWER)]

    def mu(self) -> float:
        return self._mu

    def mu_pow(self, exp: int) -> float:
        if exp < self._MAX_POWER:
            return self._mus[exp]
        else:
            return self._mu ** exp

    def sRecursive(self, node: Tree):
        """Recursive computation of function s(n) for the root of the input tree.
        s(n) sums all of the tree fragments rooted in n.
        :param node: the input tree
        :return s(node): sum of all of the tree fragments rooted in node
        """
        v = self.distributedVector(node.root)

        #print(f"at node {node}, before d(children) value of spectrum{self.spectrum}")
        # TODO non sembra mai lessicalizzato
        if node.isTerminal():
            result = (self._mu * self._terminal_factor) * v
        else:
            result = self._mu * v + self.operation(v,
                                                   self.operation(
                                                       self.distributedVector("separator"), self.d(node.children)
                                                   ))
        #print(f"at node {node}, after d(children) value of spectrum{self.spectrum}")
        result = np.sqrt(self.LAMBDA) * result
        self.spectrum = self.spectrum + result
        return result

    def d(self, trees):
        """
        Computation of D(trees)
        D(trees) sums all of the tree fragment forests rooted in any subset of nodes in c, to be attached to the parent node.
        :param trees: the list of children nodes for the parent node.
        :param sum: the object collecting the sum of s(n) for each node in the tree.
        :return D(trees):
        """
        # hashmap used for dynamic programming
        dvalues = {}

        result = np.zeros(self.dimension)
        # TODO sono pensati in parallelo o in sequenza? si puo fare la somma 1! volta e risolvere anche il dubbio su spectrum
        for k, c in enumerate(trees):
            result = result + self.__dRecursive(trees, k, dvalues) #spectrum passato originariamente qui

        return result

    def __dRecursive(self, trees, k, dvalues):
        """
        Computation of d(c_i). Dynamic programming is used for efficiency reasons.
        d(c_i) sums all the tree fragment forests rooted in c_i and any subset of nodes in c following c_i.
        :param c: the list of children nodes for the parent node.
        :param k: the current child index
        :param dvalues: the map used for dynamic programming
        :return d(c_i):
        """
        # TODO gestione hash oscura, perche' la key e' l'indice del figlio?
        if k in dvalues:
            print(f'k = {k} , trees[k] == {trees[k]}')
            return dvalues[k]

        s_trees_k = self.sRecursive(trees[k])

        if k < len(trees) - 1:
            total = self.__dRecursive(trees, k + 1, dvalues)
            for i in range(k + 2, len(trees)):
                total = total + self._mus[i - k - 1] * self.__dRecursive(trees, i, dvalues)

            result = s_trees_k + self.operation(s_trees_k, total)

        else:
            result = s_trees_k

        dvalues[k] = result
        return result

    def dt(self, tree):
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum

if __name__ == "__main__":
    ss = "(NP (DT The) (JJ wonderful) (NN time))"
    ss = ss.replace(")", ") ").replace("(", " (")
    t = Tree(string=ss)

    kernel = partialTreeKernel(dimension=8192, LAMBDA=0.6, operation=op.fast_shuffled_convolution)
    v = kernel.sRecursive(t)
    w1 = kernel.dt(t)
    print(v)
    print(w1)

    #print(kernel.kernel(t,frag))
