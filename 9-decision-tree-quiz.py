import numpy as np

S = [1, 0, 1, 0, 1, 0]
A = [1, 1, 1]
B = [0, 0, 0]


def calc_gini(data):
    count = len(data)
    survived = sum(data) / count
    unsurvived = 1 - survived

    return 2 * survived * unsurvived


def calc_information_gain(data, A, B):
    HS = calc_gini(data)
    HA = calc_gini(A)
    HB = calc_gini(B)

    return HS - len(A) / len(S) * HA - len(B) / len(S) * HB


information_gain = calc_information_gain(S, A, B)

print(round(information_gain, 5))