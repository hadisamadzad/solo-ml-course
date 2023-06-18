tp, fp, fn, tn = 233, 65, 109, 480

all = tp + fp + fn + tn

accuracy = (tp + tn) / all
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(round(accuracy, 4))
print(round(precision, 4))
print(round(recall, 4))
print(round(f1, 4))
