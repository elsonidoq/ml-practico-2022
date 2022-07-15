import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProjection(BaseEstimator, TransformerMixin):
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
    def __init__(self, fname, field):
        self.fname = fname
        self.field = field

    def fit(self, X, y):
        self.model_ = fasttext.load_model(self.fname)
        return self

    def transform(self, X):
        res = []
        for doc in X:
            value = doc[self.field]
            res.append(self.model_.get_sentence_vector(value))
        return np.asarray(res)
