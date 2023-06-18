# Sample input 0 1 2 1 2 => sample output 0.9820
w1, w2, b, x1, x2 = [float(x) for x in input().split()]


def sigmoid(a):
    return 1 / (1 + 2.71828**(-1 * a))


activation = b + w1 * x1 + w2 * x2
print(round(sigmoid(activation), 4))