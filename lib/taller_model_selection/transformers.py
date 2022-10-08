from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProjection(BaseEstimator, TransformerMixin):
    """
    Recibe una lista de campos a proyectar, y los proyecta como listas o como diccionarios
    Ver notebook 02
    """

    def __init__(self, fields, as_dict=False, convert_na=True):
        self.fields = fields
        self.as_dict = as_dict
        self.convert_na = convert_na

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            if self.as_dict:
                row = {field: doc[field] for field in self.fields}
            else:
                row = [doc[field] for field in self.fields]
            res.append(row)
        return res


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encodea una categorica como un vector de cuatro dimensiones
    [mean(y), std(y), percentile(y, 5), percentile(y, 95)]
    """

    def __init__(self, categorical_field, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.stats_ = None
        self.default_stats = None

    def fit(self, X, y):
        values = defaultdict(list)
        for i, x in enumerate(X):
            values[x[self.categorical_field]].append(y[i])

        self.stats_ = {}
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue
            tar_values = np.asarray(tar_values)
            self.stats_[cat_value] = [
                np.mean(tar_values), np.std(tar_values),
                np.percentile(tar_values, 90), np.percentile(tar_values, 10),
            ]

        self.default_stats_ = [
            np.mean(y), np.std(y),
            np.percentile(y, 90), np.percentile(y, 10)
        ]
        # Siempre hay que devolver self
        return self

    def transform(self, X):
        res = []
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res


class PretrainedFastTextTransformer(BaseEstimator, TransformerMixin):
    """
    Dado un nombre de archivo de un modelo de fasttext (ver notebook 4a y 4b) y un campo de texto
    Genera features del campo textual a traves del modelo de fasttext
    """

    def __init__(self, fname, field):
        self.fname = fname
        self.field = field
        self.model_ = None

    def sync_resources(self):
        if self.model_ is None:
            # Lazy import. Solo falla si lo usas.
            try:
                import fasttext
            except ImportError:
                raise ImportError('Falta instalar fasttext. \n \n pip install fasttext \n \n')
            self.model_ = fasttext.load_model(self.fname)

    def fit(self, X, y):
        self.sync_resources()
        return self

    def transform(self, X):
        self.sync_resources()
        res = []
        for doc in X:
            value = doc[self.field].replace('\n', '')
            res.append(self.model_.get_sentence_vector(value))
        return np.asarray(res)

    def __getstate__(self):
        state = vars(self).copy()
        state['model_'] = None
        return state

    def __setstate__(self, state):
        vars(self).update(state)
