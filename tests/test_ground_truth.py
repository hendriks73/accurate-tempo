from accuratetempo.groundtruth import log_to_label, log_to_index, linear_to_index, linear_to_label
import pytest


def test_log_to_label():
    label = log_to_label(0, 256)
    assert 30. == pytest.approx(label)
    label = log_to_label(255, 256)
    assert 285. == pytest.approx(label)


def test_log_to_index():
    index = log_to_index(30, 256)
    assert 0. == index
    index = log_to_index(285, 256)
    assert 255. == index


def test_linear_to_label():
    label = linear_to_label(0, 256)
    assert 30. == pytest.approx(label)
    label = linear_to_label(255, 256)
    assert 285. == pytest.approx(label)


def test_linear_to_index():
    index = linear_to_index(30, 256)
    assert 0. == index
    index = linear_to_index(285, 256)
    assert 255. == index
