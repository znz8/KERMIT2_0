
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

    def value(self, a, b):
        self.map.clear()
        sum = 0
        bound = 0

        if self.bound == 0:
            bound = min(len(a), len(b))
        else:
            bound = self.bound

        for i in range(1, bound+1):
            sum += self.values(a,b,i)

        return sum


    def values(self, sx, t, n: int):
        if len(sx) < n or len(t) < n:
            return 0

        key = "0\t"+str(n)+"\t"+str(hash(frozenset(str(sx))))+"\t"+str(hash(frozenset(str(t))))

        if key in self.map:
            return self.map[key]

        s = sx[0:-1]
        sum = self.values(s,t,n)
        c = sx[-1]

        for j in range(0, len(t)):
            if t[j] == c:
                sum += self.lambdaSq * self.K1(s, t[0:j], n-1)

        self.map[key] = sum
        return sum


    def K1(self, sx, t, n:int):
        if n == 0:
            return 1

        if len(sx) < n or len(t) < n:
            return 0

        key = "1\t"+str(n)+"\t"+str(hash(frozenset(str(sx))))+"\t"+str(hash(frozenset(str(t))))

        if key in self.map:
            return self.map[key]

        s = sx[0:-1]
        sum = self.LAMBDA * self.K1(s,t,n) + self.K2(sx, t, n)

        self.map[key] = sum

        return sum

    def K2(self, sx, tu, n:int):
        if len(tu) < n or len(sx) < n:
            return 0

        key = "2\t"+str(n)+str(hash(frozenset(str(sx))))+"\t"+str(hash(frozenset(tu)))

        if key in self.map:
            return self.map[key]

        t = tu[0:-1]
        sum = self.LAMBDA * self.K2(sx, t, n)

        if sx[-1] == tu[-1]:
            s = sx[0:-1]
            sum = sum + self.lambdaSq * self.K1(s,t, n-1)

        self.map[key] = sum
        return sum

    def getLambda(self):
        return self.LAMBDA

    def setLambda(self, new_lambda):
        self.LAMBDA = new_lambda
        self.lambdaSq = new_lambda * new_lambda

    def getBound(self):
        return self.bound

    def setBound(self, new_bound):
        self.bound = new_bound

SK = SequenceKernel()

#q = SK.value([1,2,3,4,5],[0,0,1,2,0])

#print(q)