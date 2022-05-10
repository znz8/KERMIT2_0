from kerMIT.structenc.dse import DSE


class partialTreeKernel(DTE):
    def __init__(self, LAMBDA = 1., MU = 1., dimension=4096, operation=op.fast_shuffled_convolution):
        super(partialTreeKernel, self).__init__(LAMBDA = LAMBDA, dimension = dimension, operation = operation)
        print ('using partial')
        self.result = np.zeros(self.dimension)
        self.spectrum = np.zeros(self.dimension)
        self.dvalues = {}
        self.mu = MU
        self.MAXPOWER = 10
        self.mus = [self.mu ** i for i in range(self.MAXPOWER)]

    def sRecursive(self, node):
        v = self.distributedVector(node.root)
        result = np.zeros(self.dimension)
        if node.isTerminal():
            result = self.mu * v
        else:
            result = self.operation(self.mu*v, self.d(node.children))

        result =  np.sqrt(self.LAMBDA) * result
        self.spectrum = self.spectrum + result
        return result

    def d(self, trees):

        result = self.dRecursive(trees, 0)
        for i, c in enumerate(trees):
            result = result + self.dRecursive(trees, i)

        return result

    def dRecursive(self, trees, i):
        if i in self.dvalues:
            return self.dvalues[i]

        sci = self.sRecursive(trees[i])
        result = np.zeros(self.dimension)
        if i < len(trees) - 1:
            total = self.dRecursive(trees, i+1)
            for k in range(i+2, len(trees)):
                total = total + self.mus[k - 1 - 1] * self.dRecursive(trees, k)

        else:
            result = sci
        self.dvalues[i] = result
        return result

    def dt(self, tree):
        self.dvalues = {}
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum
