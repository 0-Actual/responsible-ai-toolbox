# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import random
from numbers import Integral

import numpy as np

try:
    import pyspark.pandas as ps
    spark_available = True
except ImportError:
    spark_available = False
    pass


def is_integer_array(values):
    """Identify integer arrays, including non-missing object integers."""
    return (np.issubdtype(values.dtype, np.integer) or
            (values.dtype == object and values.size > 0 and
             all(isinstance(value, Integral) and
                 not isinstance(value, (bool, np.bool_))
                 for value in values.flat)))


def normalize_regression_values(values):
    """Keep integer containers exact before NumPy can infer float64.

    Object integers are converted to Python ints so NumPy scalar arithmetic
    cannot wrap. Missing, floating and nonnumeric inputs retain their types.
    """
    if isinstance(values, (list, tuple)):
        candidate = np.asarray(values, dtype=object)
        array = candidate if is_integer_array(
            candidate) else np.asarray(values)
    else:
        array = np.asarray(values)
    if array.dtype == object and is_integer_array(array):
        array = np.array([int(value) for value in array.flat],
                         dtype=object).reshape(array.shape)
    return array


def compute_regression_residuals(pred_y, true_y):
    """Subtract integer inputs without wraparound or rounding."""
    predictions = normalize_regression_values(pred_y)
    labels = normalize_regression_values(true_y)
    if is_integer_array(predictions) and is_integer_array(labels):
        # Inspect operands before promotion: mixing int64 and uint64 would
        # otherwise round to float64. Python integers also keep abs(INT64_MIN)
        # and differences spanning the full 64-bit range representable.
        size = max(predictions.dtype.itemsize, labels.dtype.itemsize)
        signed_dtype = (object if size == 8 or predictions.dtype == object or
                        labels.dtype == object else np.int64)
        predictions = predictions.astype(signed_dtype)
        labels = labels.astype(signed_dtype)
    return predictions - labels


def generate_random_unique_indexes(count, nnz):
    """Utility to generate a list of random unique indexes in memory.

    Uses Knuth's n of k algorithm to generate the list in memory.

    :param count: The range of values to choose from.
    :type count: int
    :param nnz: The number of unique values to choose.
    :type nnz: int
    :return: The list of unique indexes of length nnz.
    :rtype: numpy.ndarray
    """
    if nnz > count:
        raise Exception("Knuth n-of-k algorithms requires 0 < n < k")
    indexes = np.zeros(nnz, dtype=int)
    current = 0
    i = 0
    while i < count and current < nnz:
        probability = (nnz - current) / (count - i)
        if random.random() < probability:
            indexes[current] = i
            current += 1
        i += 1
    return indexes


def is_spark(df):
    """Utility to check if a dataframe is a spark dataframe.

    :param df: The dataframe to check.
    :type df: pyspark.sql.dataframe.DataFrame or pandas.DataFrame
    :return: True if the dataframe is a spark dataframe, False otherwise.
    :rtype: bool
    """
    try:
        return spark_available and isinstance(df, ps.frame.DataFrame)
    except Exception:
        return False
