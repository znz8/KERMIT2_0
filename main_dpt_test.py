import os

from kerMIT.structenc.dpte import partialTreeKernel
import pandas as pd
import kerMIT.operation as op
from kerMIT.tree import Tree
from kerMIT.tree_encode import parse
import numpy as np
import json


class PT_Kernel:
    __MAX_CHILDREN = 50
    __MAX_RECURSION = 20

    def __init__(self, LAMBDA=1.0, mu=1.0):
        self.LAMBDA = LAMBDA
        self.mu = mu

    def kernel_similarity(self, t1: Tree, t2: Tree):
        """
        The evaluation of the common PTs rooted in nodes n1 and n2 requires the selection of the shared child subsets of the two nodes.
        For example (S (DT JJ N)) and (S (DT N N)) have (S [N)) (2 times) and (S [DT N)) in common.
        Let F = {f1, f2, .., f|F|} be a tree fragment space of type PTs and let the indicator function I_i(n) be equal to 1 if the target fi is rooted at node n and 0 otherwise.
        We define the PT kernel as:
        K(T1, T2) = \sum_{n1 \in N_{T1}}\sum_{n2 \in N_{T2}} \Delta(n1, n2)
        where N_{T1} and N_{T2} are the sets of nodes in T1 and T2 and \Delta(n1, n2) counts common fragments rooted at the n1 and n2 nodes:
        \Delta(n1, n2) counts common fragments rooted at the n1 and n2 nodes:
        \Delta(n1, n2) = \sum_{i = 1}^{|F|} I_i(n1)I_i(n2)
        """
        sim = 0

        for n1 in t1.allNodes():
            for n2 in t2.allNodes():
                sim += self.delta(n1, n2)

        return sim

    def delta_sk(self, t1_children, t2_children):
        n = len(t1_children)
        m = len(t2_children)

        DPS = np.empty(shape=(PT_Kernel.__MAX_CHILDREN, PT_Kernel.__MAX_CHILDREN), dtype=float)
        DP = np.empty(shape=(PT_Kernel.__MAX_CHILDREN, PT_Kernel.__MAX_CHILDREN), dtype=float)
        kernel_mat = np.empty(shape=PT_Kernel.__MAX_CHILDREN, dtype=float)

        p = min(n, m)
        for j in range(0, m + 1):
            for i in range(0, n + 1):
                DPS[i][j] = DP[i][j] = 0

        kernel_mat[0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if t1_children[i - 1].root == t2_children[j - 1].root:
                    DPS[i][j] = self.delta(t1_children[i - 1], t2_children[j - 1])
                    kernel_mat[0] += DPS[i][j]
                else:
                    DPS[i][j] = 0

        for l in range(1, p):
            kernel_mat[l] = 0
            for i in range(1, n + 1):
                DP[i][l - 1] = 0
            for j in range(1, m + 1):
                DP[l - 1][j] = 0

            for i in range(l, n + 1):
                for j in range(l, m + 1):
                    DP[i][j] = DPS[i][j] + self.LAMBDA * DP[i - 1][j] + self.LAMBDA * DP[i][j - 1] - (
                            self.LAMBDA ** 2) * DP[i - 1][j - 1]

                    if t1_children[i - 1].root == t2_children[j - 1].root:
                        DPS[i][j] = self.delta(t1_children[i - 1], t2_children[j - 1]) * DP[i - 1][j - 1]
                        kernel_mat[l] += DPS[i][j]

        return sum(kernel_mat[0:p])

    def delta(self, t1: Tree, t2: Tree):
        if t1.root != t2.root:
            return 0

        if t1.isTerminal() or t2.isTerminal():
            return self.mu * (self.LAMBDA ** 2)

        return self.mu * (self.LAMBDA ** 2 + self.delta_sk(t1.children, t2.children))


def test_in_original_space(input_file, LAMBDA: float = 1, DIMENSION: int = 8192, operation=op.fast_shuffled_convolution,
                           output_path="record_result_test_vs_original_fragments_space.csv"):
    input_sentences = pd.read_csv(input_file)
    ss1 = input_sentences["s1"]
    ss2 = input_sentences["s2"]

    kernel = partialTreeKernel(dimension=DIMENSION, LAMBDA=LAMBDA, operation=operation)
    records = []
    for i in range(0, len(ss1)):
        s1, s2 = ss1[i], ss2[i]
        record = {"s1": s1, "s2": s2}

        s1 = s1.replace(")", ") ").replace("(", " (")
        s2 = s2.replace(")", ") ").replace("(", " (")
        t1 = Tree(string=s1)
        t2 = Tree(string=s2)
        print(t1)
        print(t2)

        pdt1 = kernel.ds(t1)
        pdt2 = kernel.ds(t2)
        print("PDT computed:")
        print("\t", pdt1)
        print("\t", pdt2)

        count = np.dot(pdt1, pdt2)
        cosine = count / np.sqrt(np.dot(pdt1, pdt1) * np.dot(pdt2, pdt2))
        print("dpt count: ", count)
        print("dpt similarity: ", cosine)
        record["dpt_count"] = count
        record["dpt_similarity"] = cosine

        sub_t1 = kernel.substructures(t1)
        sub_t2 = kernel.substructures(t2)

        tot = list(set(sub_t1) | set(sub_t2))

        array_1_in_F = np.array([sub_t1.count(t) for t in tot])
        array_2_in_F = np.array([sub_t2.count(t) for t in tot])

        count = np.dot(array_1_in_F, array_2_in_F)
        cosine = count / np.sqrt(np.dot(array_1_in_F, array_1_in_F) * np.dot(array_2_in_F, array_2_in_F))
        print("original count: ", count)
        print("original space similarity: ", cosine)
        record["original_count"] = count
        record["original_similarity"] = cosine

        records.append(record)

    df = pd.DataFrame(records)
    df["dimension"] = DIMENSION * len(df)
    df["LAMBDA"] = LAMBDA * len(df)

    df.to_csv(output_path)


def test_with_kernel(input_file, LAMBDA: float = 1, DIMENSION: int = 8192, operation=op.fast_shuffled_convolution,
                     output_path="record_result_test_vs_original_kernel.csv"):
    input_sentences = pd.read_csv(input_file)
    ss1 = input_sentences["s1"]
    ss2 = input_sentences["s2"]

    pt_encoder = partialTreeKernel(dimension=DIMENSION, LAMBDA=LAMBDA, operation=operation)
    pt_kernel = PT_Kernel(LAMBDA=1)

    records = []
    for i in range(0, len(ss1)):
        s1, s2 = ss1[i], ss2[i]
        record = {"s1": s1, "s2": s2}

        s1 = s1.replace(")", ") ").replace("(", " (")
        s2 = s2.replace(")", ") ").replace("(", " (")
        t1 = Tree(string=s1)
        t2 = Tree(string=s2)

        ## Partial Tree kernel
        sim = pt_kernel.kernel_similarity(t1, t2)
        scaled = sim / np.sqrt(pt_kernel.kernel_similarity(t1, t1) * pt_kernel.kernel_similarity(t2, t2))
        record["original_count"] = sim
        record["original_scaled"] = scaled

        ## Distributed Partial Tree encoder
        dpt1 = pt_encoder.ds(t1)
        dpt2 = pt_encoder.ds(t2)

        count = np.dot(dpt1, dpt2)
        scaled = count / np.sqrt(np.dot(dpt1, dpt1) * np.dot(dpt2, dpt2))
        record["dpt_count"] = count
        record["dpt_scaled"] = scaled

        records.append(record)

    df = pd.DataFrame(records)
    df["dimension"] = DIMENSION
    df["LAMBDA"] = LAMBDA
    print(df.loc[0])
    df.to_csv(output_path)


def create_test(output):
    input_path = "coco_annotation_2014/captions_train2014.json"

    fd = open(input_path, "r")
    j = json.load(fd)
    fd.close()
    captions = pd.DataFrame(j["annotations"]).set_index("image_id", drop=True).sort_index().groupby(level=0).sample(2)["caption"]

    test = []
    for i in range(0, 20, 2):
        s1 = captions.values[i]
        s2 = captions.values[i + 1]

        t1 = parse(s1).replace('\r', '').replace('\t', ' ')
        t2 = parse(s2).replace('\r', '').replace('\t', ' ')

        test.append({"s1": t1, "s2": t2})

    pd.DataFrame(test).to_csv(output)


def test_kernel_and_explicit():
    s1, s2 = "(FRAG    (NP (NNP Bob))    (VP (VB read)      (NP (DT a) (NN message))))", "(FRAG    (NP (NNP Bob))    (VP (VB send)      (NP (DT a) (NN message))))"
    s1, s2 = s1.replace(")", ") ").replace("(", " ("), s2.replace(")", ") ").replace("(", " (")
    t1, t2 = Tree(string=s1), Tree(string=s2)

    pt_kernel = PT_Kernel()
    dpt_kernel = partialTreeKernel(dimension=8192, LAMBDA=1.0, operation=op.fast_shuffled_convolution)
    print("pt kernel: ", pt_kernel.kernel_similarity(t1, t2))
    sub_t1 = dpt_kernel.substructures(t1)
    sub_t2 = dpt_kernel.substructures(t2)

    res = 0
    tot = set(sub_t1) & set(sub_t2)
    for t in tot:
        c1 = sub_t1.count(t)
        c2 = sub_t2.count(t)
        res += c1 * c2
    print("exp: ", res)


if __name__ == "__main__":
    print("---------------------------------")
    print("KERNEL vs FRAGMENTS SPACE")
    test_kernel_and_explicit()
    print("---------------------------------")

    input_file = "test.csv"
    if not os.path.exists(input_file):
        create_test(input_file)

    print("---------------------------------")
    print("DISTRIBUTED KERNEL vs KERNEL")
    test_with_kernel(input_file, LAMBDA=1, DIMENSION=8192, operation=op.fast_shuffled_convolution,
                     output_path="record_result_test_vs_original_kernel.csv")
    print("---------------------------------")
    # TODO non funziona
    # print(kernel.compute(t1, t2))
    # print("penalizing values ", [(k, kernel.dtf_cache[k][1]) for k in kernel.dtf_cache])
    # print(kernel.kernel(t,frag))
