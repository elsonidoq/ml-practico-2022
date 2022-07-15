from os import path
from .serialize import iter_jl


def date_split(X, y, cut):
    X_train, X_test = [], []
    y_train, y_test = [], []
    for x_i, y_i in zip(X, y):
        X_list = X_train if x_i['created_on'] <= cut else X_test
        y_list = y_train if x_i['created_on'] <= cut else y_test
        X_list.append(x_i)
        y_list.append(y_i)
    return X_train, X_test, y_train, y_test


def load_train_dev_test(data_path):
    X, y = map(list, map(iter_jl, [path.join(data_path, 'X_train.jl'), path.join(data_path, 'y_train.jl')]))

    X_train, X_test, y_train, y_test = date_split(X, y, '2021-03-01')
    X_dev, X_test, y_dev, y_test = date_split(X_test, y_test, '2021-04-15')
    print({
        'pct(train)': len(X_train) / len(X),
        'pct(dev)': len(X_dev) / len(X),
        'pct(test)': len(y_test) / len(X)
    })
    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)