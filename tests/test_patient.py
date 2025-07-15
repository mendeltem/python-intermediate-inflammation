"""Tests for the Patient model."""
"""
from inflammation.models import Patient

def test_create_patient():

    name = 'Alice'
    p = Patient(name=name)

    assert p.name == name
"""
import pytest
import numpy as np
import numpy.testing as npt
from inflammation.models import patient_normalise

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


def test_patient_normalise_zeros():
    """Test normalisation works for an array of zeros."""
    test_input = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    expected_result = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    
    result = patient_normalise(test_input)
    npt.assert_array_equal(result, expected_result)     

# def test_create_patient():
#     """Test that a Patient can be created with a name."""

#     name = 'Alice'
#     p = Patient(name=name)    
#     assert p.name == name

# def test_patient_normalise_empty():
#     """Test that normalisation of an empty array returns an empty array."""

#     test_input = np.array([[]])
#     expected_result = np.array([[]])
