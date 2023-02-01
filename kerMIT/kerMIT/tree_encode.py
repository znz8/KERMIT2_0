import numpy as np
from kerMIT.tree import Tree
from stanfordcorenlp import StanfordCoreNLP
import ast
import time


def parse(text, nlp=None, **kwargs):
    if nlp is None:
        nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')

    #text = (text.encode('ascii', 'ignore')).decode("utf-8")
    implemented_annotators = {'parse', 'depparse'}
    if 'annotator' in kwargs and kwargs['annotator'] in implemented_annotators:
        annotator = kwargs['annotator']
    else:
        annotator = 'parse'
    try:
        try:
            props={'annotators': annotator,'outputFormat':'json'}
            output = nlp.annotate(text, properties=props)
        except Exception as e:
            print("Exception during parsing!!")
            print(e)
            if annotator == 'parse':
                return "(S)"
            elif annotator == 'depparse':
                return "(ROOT)"

        outputD = ast.literal_eval(output)
        sentences = outputD['sentences']

        if annotator == 'parse':
            if len(sentences) <= 1:
                root = sentences[0]['parse'].strip('\n')
                root = root.split(' ',1)[1]
                root = root[1:len(root)-1]
            else:
                s1 = sentences[0]['parse'].strip('\n')
                s1 = s1.split(' ', 1)[1]
                s1 = s1[1:len(s1)-1]
                root = "(S" + s1
                for sentence in sentences[1:]: # not sure if there can be multiple items here. If so, it just returns the first one currently.
                    s2 = sentence['parse'].strip('\n')
                    s2 = s2.split(' ', 1)[1]
                    s2 = s2[1:len(s2)-1]
                    root = root + s2
                root = root + ")"

            return root.replace("\n", "")

        if annotator == 'depparse':
            trees = []
            for i in range(len(sentences)):
                dependencies = sentences[i]['basicDependencies']
                tokens = sentences[i]['tokens']

                pos_tags = False
                if 'pos_tags' in kwargs:
                    pos_tags=kwargs['pos_tags']

                root = min([d['governor'] for d in dependencies])
                parsing = ParseDependencies(root, dependencies, tokens, pos_tags=pos_tags)

                tokens_as_leaves = False
                if 'tokens_as_leaves' in kwargs:
                    tokens_as_leaves = kwargs['tokens_as_leaves']
                tree = parsing.tree(tokens_as_leaves=tokens_as_leaves)
                trees.append(tree)

            if len(sentences) == 1:
                return str(trees[0])
            else:
                return str(Tree(root="ROOT", children=trees))

    except Exception as e:
        print("Exception occurred during parsing")
        print(e)
        if annotator == 'parse':
            return '(S)'
        elif annotator == 'depparse':
            return '(ROOT)'

class ParseDependencies:
    def __init__(self, root, dependencies, tokens, **kwargs):
        self.dependencies = dependencies
        self.tokens = {token["index"]: token for token in tokens}
        self.root = root

        self._pos_tags = False
        if "pos_tags" in kwargs:
            self._pos_tags = kwargs["pos_tags"]

        self.nodes = self._nodes()
        self.adj = self._adj()

    def _find_dependency(self, idx):
        for d in self.dependencies:
            if d["dependent"] == idx: # TODO one root per word
                return {"label": d["dep"], "token": d["dependentGloss"]}


    def _nodes(self):
        nodes = {}
        for idx in self.tokens:
            nodes[idx] = self._find_dependency(idx)
            if self._pos_tags:
                nodes[idx]["pos"] = self.tokens[idx]['pos']
        return nodes

    def _adj(self):
        children = sorted([dep for dep in self.dependencies if dep["governor"] == self.root], key=lambda dep: dep['dep'])
        adj = {self.root: [dep["dependent"] for dep in children]}
        for token in self.tokens:
            children = sorted([dep for dep in self.dependencies if dep["governor"] == token], key=lambda dep: dep['dep'])
            adj[token] = [dep["dependent"] for dep in children]
        return adj

    def to_str(self, tokens_as_leaves=True):
        return str(self.tree(tokens_as_leaves=tokens_as_leaves))

    def tree(self, tokens_as_leaves=True) -> Tree:
        if tokens_as_leaves:
            return self._rec_tree(self.root)
        else:
            return self._rec_tree_with_tokens(self.root)

    def _rec_tree(self, root):
        if root in self.nodes:
            if not self._pos_tags:
                tree = Tree(root=self.nodes[root]['label'], children=[Tree(root=self.nodes[root]['token'])])
            else:
                tree = Tree(root=self.nodes[root]['label'], children=[Tree(root=self.nodes[root]['pos'],
                                                                           children=[Tree(root=self.nodes[root]['token'])])
                                                                      ]
                            )
            for child in self.adj[root]:
                tree.children.append(self._rec_tree(child))

        else:
            if root == self.root:
                child = self.adj[root][0]
                tree = self._rec_tree(child)
            else:
                raise Exception(f"Unkown node {root}")
        return tree

    def _rec_tree_with_tokens(self, root):
        tree = None
        if root in self.nodes:
            if not self._pos_tags:
                tree = Tree(root=self.nodes[root]['label'], children=[Tree(root=self.nodes[root]['token'])])
            else:
                tree = Tree(root=self.nodes[root]['label'], children=[Tree(root=self.nodes[root]['pos'],
                                                                           children=[Tree(root=self.nodes[root]['token'])])
                                                                      ])
            if len(self.adj[root]) > 0:
                tree.children[0].children =[]
                for child in self.adj[root]:
                    tree.children[0].children.append(self._rec_tree_with_tokens(child))
        else:
            if root == self.root:
                child = self.adj[root][0]
                tree = self._rec_tree_with_tokens(child)
            else:
                raise Exception(f"Unkown node {root}")

        return tree


if __name__ == "__main__":
    nlp = StanfordCoreNLP('/stanford-corenlp-full-2018-10-05')

    text = "The cat is on the table"
    tree_str = parse(text, nlp=nlp, annotator='depparse')
    print(tree_str)


    text = "The cat sleeps on the table"
    tree_str = parse(text, nlp=nlp, annotator='depparse')
    print(tree_str)


    text = "This time around, they're moving even faster."
    tree_str = parse(text, nlp=nlp, annotator='depparse')
    print(tree_str)