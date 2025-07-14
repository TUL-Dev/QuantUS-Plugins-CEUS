from typing import List

def required_kwargs(*kwarg_names: List[str]) -> dict:
    """
    A decorator to specify the required keyword arguments for a function.

    Args:
        kwarg_names (list): List of required keyword argument names.

    Returns:
        function: The decorated function with the specified keyword arguments.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['kwarg_names'] = kwarg_names
            return out_dict
        func['kwarg_names'] = kwarg_names
        return func
    return decorator
