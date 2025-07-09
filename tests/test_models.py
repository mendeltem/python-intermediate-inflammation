"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean
from inflammation.models import daily_max
from inflammation.models import daily_min

from inflammation.functions import filter_list,find_single_file


# Example of a correct parametric test for daily_max
@pytest.mark.parametrize('test_input, expected', [
    (np.array([[0, 1],
               [2, 3]]),  
      np.array([2, 3])), # expected  
    (np.array([[1, 2],
               [3, 4]]),
      np.array([3, 4])), # expected
    (np.array([[-1, -2],
               [-3, -4]]),
      np.array([-1, -2])) #excpted
])
def test_daily_max_param(test_input, expected):
    npt.assert_array_equal(daily_max(test_input), expected)




def test_daily_max_zeros():
    """Test that max function works for an empty array."""
    
    test_input = np.array([[0,0,0,0], 
                           [0,0,0,0], 
                           [0,0,0,0]])
    test_result = np.array([0,0,0,0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_integers():
    """Test that max function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                            [3, 4],
                            [5, 6]])
    test_result = np.array([5, 6])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_negative():
    """Test that min function works for an array with negative integers"""

    test_input = np.array([[-1,2],
                           [3,-2],
                           [4,5]])
    test_result = np.array([4, 5])

    npt.assert_array_equal(daily_max(test_input), test_result)



def test_daily_min_integers(): 
    """Test that min function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([1, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_zeros():
    """Test that min function works for an empty array."""
    
    test_input = np.array([[0,0,0,0],
                            [0,0,0,0], 
                            [0,0,0,0]])
    test_result = np.array([0, 0, 0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)




def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize('data, expected_standard_deviation', [
    ([0, 0, 0], 0.0),
    ([1.0, 1.0, 1.0], 0),
    ([0.0, 2.0], 1.0)
])
def test_daily_standard_deviation(data, expected_standard_deviation):
    from inflammation.models import s_dev
    result_data = s_dev(data)['standard deviation']
    npt.assert_approx_equal(result_data, expected_standard_deviation)
