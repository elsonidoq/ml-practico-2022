from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5
