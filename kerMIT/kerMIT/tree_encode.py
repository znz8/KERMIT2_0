import pandas as pd

import numpy as np

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

                inner_nodes = {d['governor'] for d in dependencies}
                adj_matrix={
                    node:sorted([d for d in dependencies if d['governor'] == node],
                                key=lambda d: d['dep']) for node in inner_nodes}
                tree_str = to_str_visit(adj_matrix, 0, set())
                trees.append(tree_str)

            if len(sentences) == 1:
                return trees[0]
            else:
                return "(ROOT1 " + " ".join(trees) + ")"


    except Exception as e:
        print(e)
        print("Except")
        if annotator == 'parse':
            return "(S)"
        elif annotator == 'depparse':
            return "(ROOT)"

def to_str_visit(adj, root, visited):
    if root in visited:
        return ""

    visited.add(root)
    children = adj[root]

    tree_str = ""
    for child in children:
        tree_str = tree_str + "(" + child['dep']
        if child['dependent'] in adj:
            tree_str = tree_str +" ( (gloss "+ child['dependentGloss'] + " ) "+ to_str_visit(adj, child['dependent'], visited) +" ))"
        else:
            tree_str = tree_str +" (gloss "+ child['dependentGloss'] +" ))"

        visited.add(child['dependent'])
    return tree_str


if __name__ == "__main__":
    nlp = StanfordCoreNLP('/stanford-corenlp-full-2018-10-05')

    text = "This time around, they're moving even faster. Who are they?"
    tree_str = parse(text, nlp=nlp, annotator='depparse')
    print(tree_str)