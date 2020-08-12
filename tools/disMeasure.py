# -*- coding: utf-8 -*-
'''
@Time    : 2020/8/12 10:08
@Author  : daluzi
@File    : disMeasure.py
'''

import math


def calCosineSimilarity(list1, list2):
    '''

    :param list1: shape: List
    :param list2: shape likes List1
    :return: cosine similarity value
    '''
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))