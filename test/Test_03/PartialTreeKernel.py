import math

from kerMIT.tree import Tree

class PartialTreeKernel:
    def __init__(self, LAMBDA=1.0, mu=1, nodeCount = 0,deltaMatrix={}, deltaPMatrix = {}, dPMatrix = {}, nodeIndices = {}):
        self.LAMBDA = LAMBDA
        self.mu = mu
        self.nodeCount = nodeCount
        self.deltaMatrix = deltaMatrix
        self.deltaPMatrix = deltaPMatrix
        self.dPMatrix = dPMatrix
        self.nodeIndices = nodeIndices

    def value(self, a: Tree, b: Tree):
        sum = 0
        for n in self.allNodes(a):
            for m in self.allNodes(b):
                sum += self.delta(n,m)
        return sum

    def delta(self, a: Tree, b: Tree):
        if a.root != b.root:
            return 0

        if str(self.nodeIndices[a]) + ":" + str(self.nodeIndices[b]) in self.deltaMatrix:
            return self.deltaMatrix[str(self.nodeIndices[a]) + ":" + str(self.nodeIndices[b])]

        k = self.LAMBDA*self.LAMBDA

        len_a = 0
        len_b = 0

        if a.children is not None:
            len_a = len(a.children)

        if b.children is not None:
            len_b = len(b.children)

        lm = min(len_a, len_b)

        for p in range(1,lm+1):
            k+= self.deltaP(p, a.children, b.children)

        k = self.mu*k
        self.deltaMatrix[str(self.nodeIndices[a]) + ":" + str(self.nodeIndices[b])] = k

        return k

    def deltaP(self, p: int, c1: list, c2: list):
        if min(len(c1), len(c2)) < p:
            return 0

        key = str(p)
        for t1 in c1:
            key += ":"+str(self.nodeIndices[t1])

        for t2 in c2:
            key += ";"+str(self.nodeIndices[t2])

        if key in self.deltaPMatrix:
            return self.deltaMatrix[key]

        res = self.deltaP(p, c1[0:-1], c2)
        last = c1[-1]
        for n in c2:
            if n.root == last.root:
                res += self.delta(last, n) * self.DP(p-1, c1[0:-1], c2[0: c2.index(n)])

        self.deltaMatrix[key] = res

        return res
    def DP(self, p: int, c1: list, c2 : list):
        if p == 0:
            return 1
        else:
            if min(len(c1), len(c2)) < p:
                return 0

        key = str(p)

        for t1 in c1:
            key += ":"+str(self.nodeIndices[t1])

        for t2 in c2:
            key += ";"+str(self.nodeIndices[t2])

        if key in self.dPMatrix:
            return self.dPMatrix[key]

        res = self.LAMBDA * self.DP(p, c1[0:-1], c2)
        last = c1[-1]

        for n in c2:
            if n.root == last.root:
                res += math.pow(self.LAMBDA, (len(c2) - c2.index(n) -1)) * self.delta(last, n) * self.DP(p-1, c1[0:-1], c2[0: c2.index(n)])

        self.dPMatrix[key] = res

        return res

    def evaluate(self, arg0: Tree, arg1: Tree):
        return self.value(arg0, arg1)

    def allNodes(self, node: Tree):
        if (node not in self.nodeIndices):
            self.nodeIndices[node] = self.nodeCount
            self.nodeCount+=1

        all = []
        all.append(node)

        if node.children is not None:
            for child in node.children:
                all.extend(self.allNodes(child))

        return all


albero = Tree(string="(A (C (w1)) (D (w2) (w3) (w4)))")
albero2 = Tree(string="(B (D (w2) (w4)))")

PTK = PartialTreeKernel()

q = PTK.value(albero, albero2)
print(q)

print(q/math.sqrt(PTK.value(albero, albero) * PTK.value(albero2, albero2)))
