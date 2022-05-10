__author__ = 'Leonardo Ranaldi'

import pandas as pd
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import ast
import time


nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')

def parse(text):
    """

    :param text: free-format text
    :return: tree in parienthetic form
    """
    try:
        try:
            parsed=""
            props={'annotators': 'parse','outputFormat':'json'}
            output = nlp.annotate(text, properties=props)
        except Exception:
            print("Parser Exception")
            return "(S)"
        outputD = ast.literal_eval(output)
        senteces = outputD['sentences']
        if len(senteces) <= 1:
            root = senteces[0]['parse'].strip('\n')
            root = root.split(' ',1)[1]
            root = root[1:len(root)-1]
        else:
            s1 = senteces[0]['parse'].strip('\n')
            s1 = s1.split(' ', 1)[1]
            s1 = s1[1:len(s1)-1]
            root = "(S" + s1
            for sentence in senteces[1:]:
                s2 = sentence['parse'].strip('\n')
                s2 = s2.split(' ', 1)[1]
                s2 = s2[1:len(s2)-1]
                root = root + s2
            root = root + ")"
        return root.replace("\n", "")
    except Exception:
        print("Parsing Exception")
        return "(S)"

