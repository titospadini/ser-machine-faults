import numpy as np

import cuml
import cudf


def _naive_bayes_train(
        X_train,
        y_train,
        **params
    ):
    model = cuml.naive_bayes.MultinomialNB(**params)
    model.fit(X_train, y_train)

    return model


def naive_bayes_train(
        X_train,
        y_train,
        **params
    ):
    X_train = cudf.DataFrame(X_train)
    y_train = cudf.Series(y_train)

    model = _naive_bayes_train(
        X_train,
        y_train,
        **params
    )

    return model


def _naive_bayes_predict(
        model,
        X_test
    ):
    y_pred = model.predict(X_test)

    return y_pred


def naive_bayes_predict(
        model,
        X_test
    ):
    X_test = cudf.DataFrame(X_test)

    y_pred = _naive_bayes_predict(
        model,
        X_test
    )

    return y_pred