def accuracy(true, pred):
    total, correct = len(true), 0
    for y, y_pred in zip(true, pred):
        correct += 1 if y == y_pred else 0
    acc = correct / (total or 1)
    return acc


def binary_f1(true, pred, selected_class=True):
    tp, fn, fp, tn = 0, 0, 0, 0
    for y, pred_y in zip(true, pred):
        if y and pred_y:
            tp += 1
        elif y and not pred_y:
            fn += 1
        elif not y and pred_y:
            fp += 1
        else:
            tn += 1
    if selected_class:
        precision = tp / ((tp + fp) or 1)
        recall = tp / ((tp + fn) or 1)
    else:
        precision = tn / ((tn + fn) or 1)
        recall = tn / ((tn + fp) or 1)
    f1 = 2 * precision * recall / ((precision + recall) or 1)
    return f1


def binary_macro_f1(true, pred):
    averaged_macro_f1 = (binary_f1(true, pred, selected_class=True) + binary_f1(true, pred, selected_class=False)) / 2
    return averaged_macro_f1
