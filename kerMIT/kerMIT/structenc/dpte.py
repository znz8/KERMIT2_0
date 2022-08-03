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
            result = self._mu * v + self.operation(v,
                                                   self.operation(
                                                        self.distributedVector("separator"),
                                                        self.d(node.children, store_substructures)
                                                   )
                                                   )

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

        result = self.__dRecursive(trees, 0, dvalues, store_substructures)
        for k in range(1, len(trees)):
            result = result + self.__dRecursive(trees, k, dvalues, store_substructures)  # spectrum passato originariamente qui

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
                total = total + self.mu_pow(i - k - 1) * self.__dRecursive(trees, i, dvalues, store_substructures)

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

    # TODO delete? impl diretta in java... ma hai gia' calcolato spectrum come somma dei result (ciascuno e' s(n))
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

    def compute(self, t1: Tree, t2: Tree):
        """
        The evaluation of the common PTs rooted in nodes n1 and n2 requires the selection of the shared child subsets of the two nodes.
        For example (S (DT JJ N)) and (S (DT N N)) have (S [N)) (2 times) and (S [DT N)) in common.
        Let F = {f1, f2, .., f|F|} be a tree fragment space of type PTs and let the indicator function I_i(n) be equal to 1 if the target fi is rooted at node n and 0 otherwise.
        We define the PT kernel as:
        K(T1, T2) = \sum_{n1 \in N_{T1}}\sum_{n2 \in N_{T2}} \Delta(n1, n2)
        where N_{T1} and N_{T2} are the sets of nodes in T1 and T2 and \Delta(n1, n2) counts common fragments rooted at the n1 and n2 nodes:
        """

        sim = 0

        def exp_delta(n1: Tree, n2: Tree):
            """
            \Delta(n1, n2) counts common fragments rooted at the n1 and n2 nodes:
            \Delta(n1, n2) = \sum_{i = 1}^{|F|} I_i(n1)I_i(n2)

            We compute as in Moschitii (2006) as follows:
            If n1 and n2 are different then \Delta(n1, n2) = 0;
            else \Delta(n1, n2) = 1 + \sum_{J_1,J_2, l(J_1)=l(J_2)} \prod_{i=1}^{l(J_1)} Delta(c_{n1}[J_{1i}], c_{n2}[J_{2i}])

            where:
            J_1 =<J_{11}, J_{12}, J_{13}, ..> and J_2 = <J_{21}, J_{22}, J_{23}, ..> are index sequences associated with the ordered child sequences c_{n1} of n1 and c_{n2} of n2, respectively
            J_{1i} and J_{2i} point to the i-th children in the two sequences, and l(·) returns the sequence length.

            """
            if n1.root != n2.root:
                return 0

            sum = 1

            c_n1 = n1.children
            c_n2 = n2.children

            range_indexes = range(0, min(len(c_n1), len(c_n2)))
            for j1, j2 in it.product(range_indexes):
                prod = 1
                for i in range(0, len(j1)):
                    prod *= exp_delta(c_n1[j1[i]], c_n2[j2[i]])
                sum += prod

        def delta(n1: Tree, n2: Tree, LAMBDA, mu):
            """
            Eq. 3 can be distributed with respect to different types of sequences, e.g. those composed by p children:
            \Delta(n1, n2) = \mu(\lambda^2 + \sum_{p=1}^{lm} \Delta_p(c_{n1} , c_{n2}))
            """
            if n1.root != n2.root:
                return 0

            if n1.children is None or n2.children is None:
                return LAMBDA + mu ** 2

            lm = min(len(n1.children), len(n2.children))
            s = 0
            for p in range(0, lm):
                s += deltap(n1.children, n2.children, LAMBDA, mu)

            return LAMBDA * (mu ** 2 + s)

        def deltap(children1, children2, LAMBDA, mu):
            if len(children1) == 0:
                return 1

            s1, a = children1[:-1], children1[-1]
            s2, b = children2[:-1], children2[-1]

            if a.root != b.root:
                return 0

            return delta(a, b, LAMBDA, mu) * Dp(s1, s2, LAMBDA, mu)

        def Dp(s1, s2, LAMBDA, mu):
            s = 0
            for i in range(0, len(s1)):
                for r in range(0, len(s2)):
                    s += mu ** (len(s1)-1 - i + len(s2)-1 - r) * deltap(s1[0:i], s2[0:r], LAMBDA, mu)

            """s = deltap(s1[1:k], s2[1:l], LAMBDA, mu)
            s += mu * Dp(s1, s2, k, l-1, LAMBDA, mu)
            s += mu * Dp(s1, s2, k-1, l, LAMBDA, mu)
            s -= (mu**2) * Dp(s1, s2, k-1, l, LAMBDA, mu)"""
            return s

        for n1 in t1.allNodes():
            for n2 in t2.allNodes():
                sim += delta(n1, n2, LAMBDA=self.LAMBDA, mu=self._mu)

        return sim

    def substructures(self, structure: Tree):
        active_in_root, inactive_in_root = self.__partialtrees(structure)
        return active_in_root + inactive_in_root

    def __partialtrees(self, t: Tree):
        if t.isPreTerminal():
            return [Tree(root=t.root), t], []

        active_in_children = []
        inactive_pt = []
        for i in range(len(t.children)):
            active_in_c, other = self.__partialtrees(t.children[i])
            active_in_children.append(active_in_c + [None])

            inactive_pt.extend(other)
            inactive_pt.extend([x for x in active_in_c if x.depth() > 1])

        active_in_t = []

        """print("---------------")
        print(t.root)
        print(t.children)
        print("PARTIALS: {")
        for ptc in it.product(*active_in_children):
            print("\t", ptc)
        print("}")
        print("---------------")"""
        for ptc in it.product(*active_in_children):
            active_pt = [x for x in ptc if x is not None]
            if len(active_pt) > 0:
                active_in_t.append(Tree(root=t.root, children=active_pt))
            else:
                active_in_t.append(Tree(root=t.root))

        return active_in_t, inactive_pt






