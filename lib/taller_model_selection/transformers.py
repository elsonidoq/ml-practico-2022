import fasttext
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

