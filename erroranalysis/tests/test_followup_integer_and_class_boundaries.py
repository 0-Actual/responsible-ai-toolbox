# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from erroranalysis._internal import surrogate_error_tree
from erroranalysis._internal.constants import (ARG, COLUMN, METHOD, Metrics,
                                               ModelTask, RootKeys)
from erroranalysis._internal.metrics import metric_to_func
from erroranalysis._internal.utils import compute_regression_residuals
from erroranalysis.analyzer import ModelAnalyzer, PredictionsAnalyzer
from raiutils.cohort import CohortFilterMethods

INTEGER_CASES = [
    pytest.param(pd.Series([2**63 - 1, 2**63 - 2], dtype='Int64'),
                 pd.Series([2**63 - 2, 2**63 - 3], dtype='Int64'),
                 id='nullable-signed'),
    pytest.param(pd.Series([2**64 - 1, 2**64 - 2], dtype='UInt64'),
                 pd.Series([2**64 - 2, 2**64 - 3], dtype='UInt64'),
                 id='nullable-unsigned'),
    pytest.param([-1, 2**63, 2**63 + 3], [-2, 2**63 - 1, 2**63 + 1],
                 id='python-mixed-range'),
    pytest.param((2**80, -2**80), (2**80 - 1, -2**80 + 1),
                 id='python-beyond-uint64'),
    pytest.param(np.array([np.uint64(0), np.uint64(2**64 - 1)], object),
                 np.array([np.uint64(1), np.uint64(2**64 - 2)], object),
                 id='object-numpy-integers'),
]


class IndexedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, dataset):
        rows = np.asarray(dataset)[:, -1].astype(int)
        if isinstance(self.predictions, pd.Series):
            return self.predictions.iloc[rows].to_numpy()
        if isinstance(self.predictions, np.ndarray):
            return self.predictions[rows]
        return np.asarray(self.predictions, dtype=object)[rows]


def make_analyzer(kind, predictions, labels, metric, classes=None):
    dataset = pd.DataFrame({'group': np.arange(len(labels)) % 2,
                            'row': np.arange(len(labels))})
    analyzer_type = ModelAnalyzer if kind == 'model' else PredictionsAnalyzer
    source = IndexedModel(predictions) if kind == 'model' else predictions
    task = (ModelTask.REGRESSION if metric == Metrics.MEAN_ABSOLUTE_ERROR
            else ModelTask.CLASSIFICATION)
    return analyzer_type(source, dataset, labels, list(dataset.columns), [],
                         model_task=task, metric=metric, classes=classes)


@pytest.mark.parametrize('predictions,labels', INTEGER_CASES)
def test_integer_container_residuals_and_mae(predictions, labels):
    expected = [int(p) - int(y) for p, y in zip(predictions, labels)]
    assert compute_regression_residuals(
        predictions, labels).tolist() == expected
    actual = metric_to_func[Metrics.MEAN_ABSOLUTE_ERROR](labels, predictions)
    assert actual == sum(abs(value) for value in expected) / len(expected)


@pytest.mark.parametrize('predictions,labels', INTEGER_CASES)
@pytest.mark.parametrize('kind', ['predictions', 'model'])
def test_integer_containers_public_views(predictions, labels, kind):
    expected = [abs(int(p) - int(y)) for p, y in zip(predictions, labels)]
    analyzer = make_analyzer(kind, predictions, labels,
                             Metrics.MEAN_ABSOLUTE_ERROR)
    assert analyzer.compute_root_stats()[RootKeys.METRIC_VALUE] == (
        sum(expected) / len(expected))
    # One row per bin ensures small errors cannot disappear in an aggregate.
    heatmap = analyzer.compute_matrix(['row', None], [], [],
                                      num_bins=len(labels))
    cells = [cell for row in heatmap['matrix'] for cell in row]
    assert sum(cell['count'] for cell in cells) == len(labels)
    for row_id, cell in zip(heatmap['category1']['values'], cells):
        assert cell['metricValue'] == expected[row_id]
    tree = analyzer.compute_error_tree(['group', 'row'], [], [],
                                       min_child_samples=1)
    assert tree[0]['metricValue'] == sum(expected) / len(expected)


@pytest.mark.parametrize('predictions,labels', [
    (np.array([0, 255], np.uint8), np.array([1, 254], np.uint8)),
    (np.array([-2**63, 2**63 - 1], np.int64), np.array([1, -1], np.int64)),
    (np.array([2**63, 2**63 + 2], np.uint64),
     np.array([2**63 - 1, 2**63 - 1], np.int64)),
])
@pytest.mark.parametrize('kind', ['predictions', 'model'])
def test_surrogate_receives_exact_signed_residuals(
        monkeypatch, predictions, labels, kind):
    predictions = np.tile(predictions, 4)
    labels = np.tile(labels, 4)
    expected = [int(p) - int(y) for p, y in zip(predictions, labels)]
    analyzer = make_analyzer(kind, predictions, labels,
                             Metrics.MEAN_ABSOLUTE_ERROR)
    original = surrogate_error_tree.create_surrogate_model
    captured = []

    def capture(*args, **kwargs):
        captured.append(list(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(surrogate_error_tree,
                        'create_surrogate_model', capture)
    tree = analyzer.compute_error_tree(['group', 'row'], [], [],
                                       min_child_samples=1)
    assert captured == [expected]
    assert tree[0]['metricValue'] == sum(
        abs(x) for x in expected) / len(expected)


@pytest.mark.parametrize('kind', ['predictions', 'model'])
@pytest.mark.parametrize('view', ['heatmap', 'heatmap-two-features', 'tree'])
@pytest.mark.parametrize('classes', [
    ['negative', 'positive'], None, [0, 1], [1, 0]])
@pytest.mark.parametrize('metric,retained', [
    (Metrics.FALSE_NEGATIVE_RATE, 1),
    (Metrics.FALSE_POSITIVE_RATE, 0),
    (Metrics.SELECTION_RATE, 0),
    (Metrics.SELECTION_RATE, 1),
    (Metrics.PRECISION_SCORE, 1),
    (Metrics.RECALL_SCORE, 1),
    (Metrics.F1_SCORE, 1),
])
def test_filtered_views_keep_full_encoded_vocabulary(
        kind, view, classes, metric, retained):
    positive = 0 if classes == [1, 0] else 1
    if metric in (Metrics.FALSE_NEGATIVE_RATE, Metrics.PRECISION_SCORE,
                  Metrics.RECALL_SCORE, Metrics.F1_SCORE):
        retained = positive
    elif metric == Metrics.FALSE_POSITIVE_RATE:
        retained = 1 - positive
    labels = np.arange(12) % 2
    analyzer = make_analyzer(kind, labels.copy(), labels, metric, classes)
    filters = [{ARG: [retained], COLUMN: 'group',
                METHOD: CohortFilterMethods.METHOD_EQUAL}]
    expected = (float(retained == positive)
                if metric == Metrics.SELECTION_RATE else 0.0)
    if metric in (Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE,
                  Metrics.F1_SCORE):
        expected = 1.0
    if view.startswith('heatmap'):
        features = ['row', 'group' if view == 'heatmap-two-features' else None]
        result = analyzer.compute_matrix(features, filters, [], num_bins=2)
        cells = [cell for row in result['matrix'] for cell in row]
        assert sum(cell['count'] for cell in cells) == 6
        assert all(cell['metricValue'] == expected
                   for cell in cells if cell['count'])
    else:
        tree = analyzer.compute_error_tree(['row'], filters, [],
                                           min_child_samples=1)
        assert tree[0]['size'] == 6
        assert all(node['metricValue'] == expected for node in tree)


@pytest.mark.parametrize('labels,predictions', [
    (pd.Series([pd.NA], dtype='Int64'), pd.Series([1], dtype='Int64')),
    ([float('nan')], [0]), ([], []), ([1, 2], [1]),
])
def test_integer_normalization_preserves_invalid_mae_errors(
        labels, predictions):
    with pytest.raises((TypeError, ValueError)):
        metric_to_func[Metrics.MEAN_ABSOLUTE_ERROR](labels, predictions)


def test_float_and_multioutput_controls():
    metric = metric_to_func[Metrics.MEAN_ABSOLUTE_ERROR]
    assert metric([1.25, -2.5], [0.5, -1.0]) == 1.125
    assert metric([[1.5, 2.5]], [[0.5, 0.5]]) == 1.5
    np.testing.assert_array_equal(
        compute_regression_residuals([1.5, -2.5], [0.5, -1]), [1, -1.5])
