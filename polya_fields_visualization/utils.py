from typing import Callable
import numpy as np
from typing import Union, Tuple

def validate_input(f: Callable,
                   x_range: Union[tuple[float, float, int], list[float]],
                   y_range: Union[tuple[float, float, int], list[float]],
                   type: str,
                   static: bool,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Function for input validation and processing
    
    :param f: Complex function
    :param x_range: X-axis range (list or tuple (min, max, count))
    :param y_range: Y-axis range (list or tuple (min, max, count))
    :param kwargs: Visualization parameters
    :return: Tuple (x, y, config) with processed data and settings
    """
    if type == "2d":
        config = {
            "type_plot": "vector",
            "lable_x": "X",
            "lable_y": "Y",
            "title_plot": "Vizualization",
            "color_line_x": "black",
            "color_line_y": "black",
            "width_line_x": 0.5,
            "width_line_y": 0.5,
            "fig_size": (8, 8),
            "width_line": 2,
            "color_vector": "autumn",
            "vector_scale": 0.1,
            "contour_func": None,
            "contour_color": "green",
            "contour_linewidth": 2,
            "eps": 1e-100
        }
    else:
        config = {
            "lable_x": "X",
            "lable_y": "Y",
            "lable_z": "Z",
            "title_plot": "Vizualization",
            "color_line_x": "black",
            "color_line_y": "black",
            "color_line_z": "black",
            "width_line_x": 0.5,
            "width_line_y": 0.5,
            "width_line_z": 0.5,
            "fig_size": (8, 8),
            "width_line": 2,
            "color_vector": 'autumn',
            "vector_scale": 0.1,
            "vector_length": 0.2,
            "subsampling": 2,
            "contour_func": None,
            "contour_color": "green",
            "contour_linewidth": 2,
            "eps": 1e-100
        }
    
    if not static:
        config.update({"show_vectors": False, "num_particles": 20, "dt": 0.1, "trail_length": 5, "trail_width": 0.5,
                      "frames": 100, "interval": 50, "color_particles": 'darkgray'})
    if not callable(f):
        raise TypeError("f must be a callable function")
        
    if isinstance(x_range, list):
        if len(x_range) == 0:
            raise ValueError("x_range: List cannot be empty")
        x = np.array(x_range)
    elif isinstance(x_range, tuple):
        if len(x_range) != 3:
            raise ValueError("x_range: Expected tuple of 3 elements (min, max, count)")
        xmin, xmax, nx = x_range
        if not (isinstance(xmin, (int, float)) and 
                isinstance(xmax, (int, float)) and 
                isinstance(nx, int)):
            raise TypeError("x_range: Elements must be numbers, count must be integer")
        if xmin >= xmax:
            raise ValueError("x_range: min must be less than max")
        if nx <= 0:
            raise ValueError("x_range: count must be positive")
        x = np.linspace(xmin, xmax, nx)
    else:
        raise TypeError("x_range: Unsupported type, must be list or tuple")
    
    if isinstance(y_range, list):
        if len(y_range) == 0:
            raise ValueError("y_range: List cannot be empty")
        y = np.array(y_range)
    elif isinstance(y_range, tuple):
        if len(y_range) != 3:
            raise ValueError("y_range: Expected tuple of 3 elements (min, max, count)")
        ymin, ymax, ny = y_range
        if not (isinstance(ymin, (int, float)) and 
                isinstance(ymax, (int, float)) and 
                isinstance(ny, int)):
            raise TypeError("y_range: Elements must be numbers, count must be integer")
        if ymin >= ymax:
            raise ValueError("y_range: min must be less than max")
        if ny <= 0:
            raise ValueError("y_range: count must be positive")
        y = np.linspace(ymin, ymax, ny)
    else:
        raise TypeError("y_range: Unsupported type, must be list or tuple")
    
    valid_params = config.keys()
    for key, value in kwargs.items():
        if key not in valid_params:
            raise ValueError(f"Parameter '{key}' is not supported")
        elif key == "contour_func":
            if not callable(value):
                raise TypeError("contour_func must be a callable function")
            config[key] = value
        elif key in ["vector_scale", "width_line_x", "width_line_y", "width_line", 
                    "dt", "vector_length", "contour_linewidth", "eps"]:
            if not isinstance(value, (int, float)):
                raise TypeError(f"'{key}' must be a number")
            if value <= 0:
                raise ValueError(f"'{key}' must be positive")
                
        elif key in ["num_particles", "frames", "interval", "subsampling", "trail_length", "trail_width"]:
            if not isinstance(value, int):
                raise TypeError(f"'{key}' must be an integer")
            if value <= 0:
                raise ValueError(f"'{key}' must be positive")
                
        elif key == "fig_size":
            if not (isinstance(value, tuple) and len(value) == 2):
                raise TypeError("fig_size must be a tuple of two numbers")
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("fig_size elements must be numbers")
        
        elif key == "type_plot":
            if value not in ["vector", "stream"]:
                raise ValueError("type_plot must be 'vector' or 'stream'")
                
        elif key == "show_vectors":
            if not isinstance(value, bool):
                raise TypeError(f"'{key}' must be a boolean")

        else:
            if not isinstance(value, str):
                raise TypeError(f"'{key}' must be a string")
        
        config[key] = value

    return x, y, config

def conjugate_function(f, x, y, eps):
    lst = []
    for imag_elem in y:
        row = []
        for real_elem in x:
            try:
                res = f(complex(real_elem, imag_elem))
            except ZeroDivisionError:
                res = f(complex(real_elem + eps, imag_elem))
            row.append(complex(res.real, -res.imag))
        lst.append(row)
    return lst