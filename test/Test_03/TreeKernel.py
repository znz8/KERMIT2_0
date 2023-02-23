from kerMIT.tree import Tree
class TreeKernel:
    def __init__(self,LAMBDA = 1.0, lexicalized=False, nodeCount=0, deltaMatrix={}, nodeIndices={}):
        self.LAMBDA = LAMBDA
        self.lexicalized = lexicalized
        self.nodeCount = nodeCount
        self.deltaMatrix = deltaMatrix
        self.nodeIndices = nodeIndices

    def value(self, a: Tree, b: Tree):
        sum = 0

        for n in a.allNodes():
            for m in b.allNodes():
                sum += self.delta(n,m)
        return sum

    def delta(self, a: Tree, b: Tree):
        k = 0
        if a not in self.nodeIndices:
            self.nodeIndices[a] = self.nodeCount
            self.nodeCount+=1

        if b not in self.nodeIndices:
            self.nodeIndices[b] = self.nodeCount
            self.nodeCount+=1


        if str(self.nodeIndices[a]) + ":" + str(self.nodeIndices[b]) in self.deltaMatrix:
            return self.deltaMatrix[str(self.nodeIndices[a]) + ':' + str(self.nodeIndices[b])]

        if a.children is not None and b.children is not None:                                   #Aggiunta atrimenti comparava il len(None)
            if len(a.children) == len(b.children):
                if len(a.children) == 1 and a.children[0].isTerminal() and b.children[0].isTerminal():
                    if self.lexicalized and a == b:
                        k = self.LAMBDA
                    else:
                        if self.productionCompare(a,b):
                            k = self.LAMBDA

                            for i in range(0, len(a.children)):
                                k = k*(1+self.delta(a.children[i], b.children[i]))

        self.deltaMatrix[str(self.nodeIndices[a]) + ":" + str(self.nodeIndices[b])] = k

        return k

    def productionCompare(self, a: Tree, b: Tree):
        if a.root != b.root:
            return False
        if len(a.children) != len(b.children) or len(a.children) == 0:
            return False

        for i in range(0, len(a.children)):
            if a.children[i].root != b.children[i].root:
                print('Ciao')
                return False
        return True

    '''
    def allNodes(self, node: Tree):
        all = []
        all.append(node)

        for child in node.children:
            all.append(allNodes(child))         #non ne sono sicuro

        return all
    '''

    def evaluate(self, arg0: Tree, arg1: Tree):
        return self.value(arg0,arg1)



albero = Tree(string="(ROOT (A (B C)) (D (E))))")
albero2 = Tree(string="(ROOT (A (B C)) (D (E))))")

TreeK = TreeKernel()

#print(albero2.children)
#print(len(albero2.children))

vals = TreeK.value(albero, albero2)
print(vals)

