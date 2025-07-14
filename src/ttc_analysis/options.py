import importlib
from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

def analysis_args(parser: ArgumentParser):
    parser.add_argument('analysis_type', type=str, default='spectral_paramap',
                        help='Analysis type to complete. Available analysis types: ' + ', '.join(get_analysis_types()[0].keys()))
    parser.add_argument('--analysis_kwargs', type=str, default='{}',
                        help='Analysis kwargs in JSON format needed for analysis class.')
    
def get_required_kwargs(analysis_type: str, analysis_funcs: list) -> list:
    """Get required kwargs for a given list of analysis functions.

    Args:
        analysis_type (str): the type of analysis to perform.
        analysis_funcs (list): list of analysis functions to apply.

    Returns:
        list: List of required kwargs for the specified analysis functions.
    """
    
    all_analysis_funcs = get_analysis_types()[1]
    
    # Find all required kwargs
    required_kwargs = []
    for name in analysis_funcs:
        # Consider dependencies of analysis functions as well
        for dep in all_analysis_funcs[analysis_type][name].get('deps', []):
            if dep not in analysis_funcs:
                analysis_funcs.append(dep)

    for name in analysis_funcs:
        required_kwargs.extend(all_analysis_funcs[analysis_type][name].get('kwarg_names', []))
    required_kwargs = list(set(required_kwargs))  # Remove duplicates
    
    return required_kwargs
    
def get_analysis_types() -> Tuple[dict, dict]:
    """Get analysis types for the CLI.
    
    Returns:
        dict: Dictionary of analysis types.
        dict: Dictionary of analysis functions for each type.
    """
    types = {}
    analysis_func_handles = {}
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(
                    f"src.ttc_analysis.{folder.name}.framework"
                )
                entry_class_name = ''.join(word.capitalize() for word in folder.name.split('_')) + "Analysis"
                entry_class = getattr(module, entry_class_name, None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError as e:
                print(f"Module not found: {e}")
                # Handle the case where the module cannot be found
                pass
            
    functions = {}
    for type_name, type_class in types.items():
        try:
            module = importlib.import_module(f'src.ttc_analysis.{type_name}.functions')
            for name, obj in vars(module).items():
                try:
                    if type(obj) == dict and callable(obj['func']):
                        if not isinstance(obj, type):
                            functions[type_name] = functions.get(type_name, {})
                            functions[type_name][name] = obj
                except (TypeError, KeyError):
                    pass
        except ModuleNotFoundError:
            # Handle the case where the functions module cannot be found
            functions[type_name] = {}
            
    return types, functions
