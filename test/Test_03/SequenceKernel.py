
class SequenceKernel:
    def __init__(self, LAMBDA = 1, lambdaSq = 1, bound = 0, map = {}):
        self.LAMBDA = LAMBDA
        self.lambdaSq = lambdaSq
        self.bound = bound
        self.map = map

    def SequenceKernel(self):
        self.lambdaSq = self.LAMBDA * self.LAMBDA

    def SequenceKernel(self, Lambda):
        self.LAMBDA = Lambda

    def value(self, ):
        return 0

    def value(self, sx, t, n: int):
        if len(sx) < n or len(t) < n:
            return 0

        key = "0\t"+n+"\t"+hash(frozenset(str(t)))+"\t"+Arrays.hashCode(t)

        return 9


x = [1,2,3]
y = 5
z = 'ciao'

print(hash(frozenset(y)))