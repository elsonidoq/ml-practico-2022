from sklearn.metrics import mean_squared_error
from os import path

from sklearn.model_selection import train_test_split

from .serialize import iter_jl


def date_split(X, y, cut):
    """
    Funcion util para partir los datos segun una fecha de corte
    """
    X_train, X_test = [], []
    y_train, y_test = [], []
    for x_i, y_i in zip(X, y):
        X_list = X_train if x_i['created_on'] <= cut else X_test
        y_list = y_train if x_i['created_on'] <= cut else y_test
        X_list.append(x_i)
        y_list.append(y_i)
    return X_train, X_test, y_train, y_test


def load_train_dev_test(data_path):
    """
    Permite levantar los datos
    """
    X, y = map(list, map(iter_jl, [path.join(data_path, 'X_train.jl'), path.join(data_path, 'y_train.jl')]))

    X_train, X_test, y_train, y_test = date_split(X, y, '2021-04-15')
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, train_size=0.9, random_state=42)
    train_test_split()
    print({
        'pct(train)': len(X_train) / len(X),
        'pct(dev)': len(X_dev) / len(X),
        'pct(test)': len(y_test) / len(X)
    })
    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


def load_submission_data(data_path):
    return list(iter_jl(path.join(data_path, 'X_test.jl')))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


class Evaluator:
    def __init__(self, X_train, y_train, X_dev, y_dev, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_test = X_test
        self.y_test = y_test
        self.evaluations = []

    def eval_pipe(self, model_name, pipe):
        res = self.eval_prediction(model_name, pipe.predict(self.X_train), pipe.predict(self.X_dev))
        if self.X_test is not None:
            res['test'] = rmse(self.y_test, pipe.predict(self.X_test))
        return res

    def eval_prediction(self, model_name, y_hat_train, y_hat_dev):
        res = dict(
            name=model_name,
            train=rmse(self.y_train, y_hat_train),
            dev=rmse(self.y_dev, y_hat_dev)
        )

        self.evaluations.append(res)
        return res
