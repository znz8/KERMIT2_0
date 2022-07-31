import gc

from kerMIT.structenc.dse import DSE
import kerMIT.operation as op
from kerMIT.tree import Tree
import numpy as np
import itertools as it


class partialTreeKernel(DSE):
    def __init__(self, LAMBDA=1., MU=1., dimension=4096, operation=op.fast_shuffled_convolution, terminal_factor=1.):
        self.LAMBDA = LAMBDA
        self.dimension = dimension
        self.operation = operation
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.random_cache = {}
        self.result = np.zeros(self.dimension)
        self.spectrum = np.zeros(self.dimension)

        self.__set_terminal_factor(terminal_factor)
        # TODO perche' MAXPOWER non e' parametrico? anche su java e' 10
        self.__set_mu(MU, MAX_POWER=10)

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
        np.random.seed(h)  # inizializzare np.random.seed()
        v_ = op.random_vector(self.dimension, normalized=False)
        self.random_cache[s] = v_
        return v_

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

    def sRecursive(self, node: Tree, store_substructures=False):
        """Recursive computation of function s(n) for the root of the input tree.
        s(n) sums all of the tree fragments rooted in n.
        :param node: the input tree
        :return s(node): sum of all of the tree fragments rooted in node
        """
        v = self.distributedVector(node.root)
        penalizing_value = 1

        # TODO non sembra mai non lessicalizzato
        if node.isTerminal():
            penalizing_value = self._mu * self._terminal_factor
            result = penalizing_value * v
        else:
            result = self._mu * v + self.operation(v, self.operation(
                self.distributedVector("separator"),
                self.d(node.children, store_substructures)))

        # TODO quale e' qui il penalizing_value? mu non entra mai nella def del peso?
        penalizing_value = penalizing_value * np.sqrt(self.LAMBDA)
        result = penalizing_value * result

        self.spectrum = self.spectrum + result

        if store_substructures:
            self.dtf_cache[node] = (result, penalizing_value)

        return result

    def d(self, trees, store_substructures=False):
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
        for k, c in enumerate(trees):
            result = result + self.__dRecursive(trees, k, dvalues,
                                                store_substructures)  # spectrum passato originariamente qui

        return result

    def __dRecursive(self, trees, k, dvalues, store_substructures=False):
        """
        Computation of d(c_i). Dynamic programming is used for efficiency reasons.
        d(c_i) sums all the tree fragment forests rooted in c_i and any subset of nodes in c following c_i.
        :param c: the list of children nodes for the parent node.
        :param k: the current child index
        :param dvalues: the map used for dynamic programming
        :return d(c_i):
        """
        if k in dvalues:
            return dvalues[k]

        s_trees_k = self.sRecursive(trees[k], store_substructures)
        if k < len(trees) - 1:
            total = self.__dRecursive(trees, k + 1, dvalues, store_substructures)
            for i in range(k + 2, len(trees)):
                total = total + self._mus[i - k - 1] * self.__dRecursive(trees, i, dvalues, store_substructures)

            result = s_trees_k + self.operation(s_trees_k, total)
        else:
            result = s_trees_k

        dvalues[k] = result
        return result

    def ds(self, tree):
        return self.dpt(tree)

    def dpt(self, tree):
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum

    # TODO delete? impl diretta in java... ma memo substructs
    def dpt_v2(self, tt: Tree):
        result = np.zeros(self.dimension)
        for n in tt.allNodes():
            result += self.sRecursive(n, store_substructures=True)
        return result

    def dsf(self, structure: Tree, original: Tree):
        return self.dsf_with_weight(structure, original)[0]

    def dsf_with_weight(self, structure: Tree, original: Tree):
        superTree = self.findSuperTree(structure, original)

        if superTree is None:
            raise ValueError("Fragment not found in originary tree!")

        if structure.isTerminal():
            # In this case, return value will be mistakenly multiplied by lambdaSq,
            # so it must be divided by lambdaSq to compensate
            penalizing_value = np.sqrt(self.LAMBDA) * self._mu
            result = penalizing_value * self.distributedVector(structure.root)
        else:
            # The composition order is different from the one of the classic DTK, it is n#(c1#(c2#...#(cn-1#cn)...))
            result, penalizing_value = self.dsf_with_weight(structure.children[len(structure.children) - 1], superTree)
            for i in range(len(structure.children) - 2, -1, -1):
                result = self.operation(
                    self.dsf_with_weight(structure.children[i], superTree)[0],
                    result
                )

            penalizing_value = np.sqrt(self.LAMBDA)
            result = penalizing_value * self.operation(self.distributedVector(structure.root), result)

        return (result, penalizing_value)

    def findSuperTree(self, fragment: Tree, whole: Tree):
        if self.isSuperTree(fragment, whole):
            return whole

        superTree = None
        if not whole.isTerminal():
            for c in whole.children:
                if superTree is not None:
                    if self.findSuperTree(fragment, c) is not None:
                        raise Exception("Tree fragment may refer to multiple subtrees!")
                else:
                    superTree = self.findSuperTree(fragment, c)

        return superTree

    def isSuperTree(self, fragment: Tree, whole: Tree):
        if fragment.root != whole.root:
            return False

        if fragment.isTerminal():
            return True

        i = -1
        for c in fragment.children:
            while True:
                i += 1
                if i >= len(whole.children):
                    return False

                if self.isSuperTree(c, whole.children[i]):
                    break
        return True

    ## TODO implementazione diretta se dptf e' "parte" di tutto l'albero... non viene uguale a sopra
    def dptf_with_weight_v2(self, structure: Tree, original: Tree):
        self.spectrum = np.zeros(self.dimension)
        self.dtf_cache = {}

        self.sRecursive(original, store_substructures=True)
        return self.dtf_cache[structure]

    def substructures(self, structure: Tree):
        active_in_root, all = self.partialtrees(structure)
        return all

    def partialtrees(self, tree: Tree):
        if tree.isPreTerminal():
            return [Tree(root=tree.root), tree], []

        # all combination of trees can
        rooted_in_children = []
        all_pt = []
        for i in range(len(tree.children)):
            rooted_in_c, other = self.partialtrees(tree.children[i])
            rooted_in_children.append(rooted_in_c + [None])
            all_pt.extend(other)
            all_pt.extend(rooted_in_c)

        active = []
        for ptc in it.product(*rooted_in_children):
            children = [x for x in ptc if x is not None]
            if len(children) > 0:
                active.append(Tree(root=tree.root, children=children))
            else:
                active.append(Tree(root=tree.root))

        return active, all_pt


if __name__ == "__main__":
    ss = "(NP (DT The) (JJ wonderful) (NN time))"
    ss = '(NOTYPE##ROOT(NOTYPE##NP(NOTYPE##S(NOTYPE##NP(NOTYPE##NNP(NOTYPE##Children)))(NOTYPE##VP-REL(NOTYPE##VBG-REL(NOTYPE##W))(NOTYPE##CC(NOTYPE##and))(NOTYPE##VBG(NOTYPE##waving))(NOTYPE##PP(NOTYPE##IN(NOTYPE##W))(NOTYPE##NP(NOTYPE##NN(NOTYPE##camera))))))))'
    ss = ss.replace(")", ") ").replace("(", " (")
    t = Tree(string=ss)

    LAMBDA = 1
    kernel = partialTreeKernel(dimension=5, LAMBDA=LAMBDA, operation=op.fast_shuffled_convolution)
    print(kernel.partialtrees(t))

    (root_dptf, root_penalization) = kernel.dsf_with_weight(t.children[0], t)

    kernel = partialTreeKernel(dimension=5, LAMBDA=LAMBDA, operation=op.fast_shuffled_convolution)
    root_dptf_sRecursive = kernel.sRecursive(t)

    kernel = partialTreeKernel(dimension=5, LAMBDA=LAMBDA, operation=op.fast_shuffled_convolution)
    (root_dptf_2, root_penalization_2) = kernel.dptf_with_weight_v2(t.children[0], t)

    kernel = partialTreeKernel(dimension=5, LAMBDA=LAMBDA, operation=op.fast_shuffled_convolution)
    dpt = kernel.ds(t)

    kernel = partialTreeKernel(dimension=5, LAMBDA=LAMBDA, operation=op.fast_shuffled_convolution)
    dpt_2 = kernel.dpt_v2(t)

    print(root_dptf)
    print(root_dptf_sRecursive)
    print(root_dptf_2)

    print()
    print(dpt)
    print(np.sum([kernel.dtf_cache[k][0] for k in kernel.dtf_cache], axis=0))
    print("penalizing values ", [(k, kernel.dtf_cache[k][1]) for k in kernel.dtf_cache])
    # print(kernel.kernel(t,frag))

    """print()
    print(t)
    sub = t.children[0].children[0]
    print(sub)
    print(kernel.findSuperTree(sub, t))"""
