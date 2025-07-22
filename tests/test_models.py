"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean
from inflammation.models import daily_max
from inflammation.models import daily_min

from inflammation.functions import filter_list,find_single_file

# Test data for filter_list
@pytest.mark.parametrize('files, include, exclude, expected', [
    # Basic functionality tests
    (['file1.txt', 'file2.txt', 'data.csv'], ['txt'], None, ['file1.txt', 'file2.txt']),
    (['file1.txt', 'file2.txt', 'data.csv'], None, ['txt'], ['data.csv']),
    (['file1.txt', 'file2.txt', 'data.csv'], ['file'], ['2'], ['file1.txt']),
    
    # Multiple include criteria (ALL must be present)
    (['data_2023.csv', 'data_2024.csv', 'report_2023.txt'], ['data', '2023'], None, ['data_2023.csv']),
    (['user_data_final.csv', 'user_report.txt', 'final_data.csv'], ['user', 'data'], None, ['user_data_final.csv']),
    
    # Multiple exclude criteria (NONE should be present)
    (['file1.txt', 'file2.txt', 'temp.tmp', 'backup.bak'], None, ['tmp', 'bak'], ['file1.txt', 'file2.txt']),
    
    # Both include and exclude
    (['data_old.csv', 'data_new.csv', 'report_old.txt'], ['data'], ['old'], ['data_new.csv']),
    
    # Edge cases - empty inputs
    ([], ['txt'], None, []),
    (['file1.txt', 'file2.txt'], [], None, ['file1.txt', 'file2.txt']),
    (['file1.txt', 'file2.txt'], None, [], ['file1.txt', 'file2.txt']),
    
    # Edge cases - None inputs
    (['file1.txt', 'file2.txt'], None, None, ['file1.txt', 'file2.txt']),
    
    # No matches
    (['file1.txt', 'file2.txt'], ['csv'], None, []),
    (['file1.txt', 'file2.txt'], None, ['txt'], []),
    
    # Case sensitivity
    (['File1.TXT', 'file2.txt'], ['txt'], None, ['file2.txt']),
    
    # Full paths
    (['/path/to/data.csv', '/path/to/report.txt', '/other/data.txt'], ['data'], None, ['/path/to/data.csv', '/other/data.txt']),
    
    # Complex filtering - no matches because 'test_integration.py' doesn't contain 'unit'
    (['test_unit_final.py', 'test_integration.py', 'unit_helper.py'], ['test', 'unit'], ['final'], []),
])

def test_filter_list(files, include, exclude, expected):
    """Test filter_list function with various input combinations."""
    result = filter_list(files, include, exclude)
    assert result == expected






# Test cases that should raise ValueError (multiple matches)
@pytest.mark.parametrize('files, include, exclude', [
    # Multiple matches
    (['file1.txt', 'file2.txt'], ['txt'], None),
    (['data_2023.csv', 'data_2024.csv', 'report.txt'], ['data'], None),
    (['user_report.txt', 'admin_report.txt', 'data.csv'], ['report'], None),
    
    # Complex criteria with multiple matches
    (['test_unit_a.py', 'test_unit_b.py', 'test_integration.py'], ['test'], ['integration']),
])
def test_find_single_file_multiple_matches(files, include, exclude):
    """Test find_single_file when multiple matches are found."""
    with pytest.raises(ValueError, match="Expected exactly one match, but found"):
        find_single_file(files, include, exclude)


# Test cases that should raise FileNotFoundError
@pytest.mark.parametrize('files, include, exclude', [
    # No matches
    (['file1.txt', 'file2.txt'], ['csv'], None),
    (['file1.txt', 'file2.txt'], None, ['txt']),
    ([], ['txt'], None),
    (['data.csv', 'report.txt'], ['pdf'], None),
    
    # Complex criteria with no matches
    (['test_unit.py', 'test_integration.py'], ['test', 'performance'], None),
])
def test_find_single_file_not_found(files, include, exclude):
    """Test find_single_file when no matches are found."""
    with pytest.raises(FileNotFoundError, match="No file matches the given criteria"):
        find_single_file(files, include, exclude)


# Test data for find_single_file
@pytest.mark.parametrize('files, include, exclude, expected', [
    # Single match cases
    (['file1.txt', 'file2.txt', 'data.csv'], ['csv'], None, 'data.csv'),
    (['report_2023.txt', 'report_2024.txt', 'data.csv'], ['2023'], None, 'report_2023.txt'),
    (['user_data.csv', 'admin_data.csv', 'report.txt'], ['user'], None, 'user_data.csv'),
    
    # Complex filtering with single result
    (['test_unit_old.py', 'test_unit_new.py', 'test_integration.py'], ['test', 'unit'], ['old'], 'test_unit_new.py'),
])
def test_find_single_file_success(files, include, exclude, expected):
    """Test find_single_file when exactly one match is found."""
    result = find_single_file(files, include, exclude)
    assert result == expected




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


<<<<<<< HEAD



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflammation.compute_data import load_inflammation_data, analyse_data


def test_load_inflammation_data():   
    """Test that load_inflammation_data loads data correctly."""
    
    # Assuming the data directory is 'data' and contains valid CSV files    
    data_dir = 'data'
    data = load_inflammation_data(data_dir)

    # Check that data is a list of numpy arrays 
    assert isinstance(data, list)
    assert all(isinstance(d, np.ndarray) for d in data)
    

=======
@pytest.mark.parametrize('data, expected_standard_deviation', [
    ([0, 0, 0], 0.0),
    ([1.0, 1.0, 1.0], 0),
    ([0.0, 2.0], 1.0)
])
def test_daily_standard_deviation(data, expected_standard_deviation):
    from inflammation.models import s_dev
    result_data = s_dev(data)['standard deviation']
    npt.assert_approx_equal(result_data, expected_standard_deviation)
<<<<<<< HEAD
 

=======
>>>>>>> main
>>>>>>> 60f6114c7b4b059623b5562062c3e6829b0184ab
