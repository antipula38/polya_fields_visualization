import pytest
import numpy as np
from typing import Callable, Union, Tuple
from ..polya_fields_visualization.utils import conjugate_function, validate_input
from ..polya_fields_visualization.polya_fields_visualization import visualization, visualization_anim, visualization_sphere, animate_sphere

def test_conjugate_function_basic():
    def f(z):
        return z**2
    
    x = [1, 2]
    y = [1, 2]
    result = conjugate_function(f, x, y)
    
    assert isinstance(result, list)
    assert len(result) == len(y)
    assert all(len(row) == len(x) for row in result)
    
    assert np.isclose(result[0][0], complex(0, -2))
    assert np.isclose(result[1][1], complex(0, -8))

def test_conjugate_function_zero_division():
    def f(z):
        return 1/z
    
    x = [0, 1]
    y = [0, 1]
    result = conjugate_function(f, x, y)
    
    assert not np.isnan(result[0][0])
    assert not np.isinf(result[0][0])

def test_validate_input_correct():
    def f(z):
        return z
    
    x, y, config = validate_input(f, (-1, 1, 10), (-2, 2, 20), "2d", True)
    assert len(x) == 10
    assert len(y) == 20
    assert isinstance(config, dict)
    
    x_list = list(np.linspace(-1, 1, 5))
    y_list = list(np.linspace(-2, 2, 10))
    x, y, config = validate_input(f, x_list, y_list, "3d", False)
    assert len(x) == 5
    assert len(y) == 10

def test_validate_input_errors():
    def f(z):
        return z
    
    with pytest.raises(TypeError):
        validate_input("not a function", (-1, 1, 10), (-2, 2, 20), "2d", True)
    
    with pytest.raises(ValueError):
        validate_input(f, (-1, 1), (-2, 2, 20), "2d", True)  # Не хватает count
    
    with pytest.raises(ValueError):
        validate_input(f, (1, -1, 10), (-2, 2, 20), "2d", True)  # min > max
    
    with pytest.raises(TypeError):
        validate_input(f, "invalid", (-2, 2, 20), "2d", True)


def test_extreme_values():
    def f(z):
        return 1e100 * z

    x = [-1e100, 0, 1e100]
    y = [-1e100, 0, 1e100]

    result = conjugate_function(f, x, y)
    assert isinstance(result, list)
    x, y, config = validate_input(f, (-1e100, 1e100, 10), (-1e100, 1e100, 10), "2d", True)
    assert len(x) == 10
    assert len(y) == 10


def test_config_parameters():
    def f(z):
        return z
    x, y, config = validate_input(f, (-1, 1, 10), (-2, 2, 20), "2d", True,
                                  title_plot="Custom Title", color_vector="winter")

    assert config["title_plot"] == "Custom Title"
    assert config["color_vector"] == "winter"

    with pytest.raises(ValueError):
        validate_input(f, (-1, 1, 10), (-2, 2, 20), "2d", True, invalid_param=123)
