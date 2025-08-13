import json
import yaml
import argparse
from pathlib import Path

from src.image_loading.options import get_scan_loaders, scan_loader_args
from src.seg_loading.options import get_seg_loaders, seg_loader_args
from src.time_series_analysis.options import get_analysis_types, analysis_args
from src.curve_loading.options import get_curves_loaders
from src.curve_quantification.options import get_quantification_funcs
from src.curve_quantification.framework import CurveQuantifications
from src.visualizations.options import get_visualization_types

DESCRIPTION = """
QuantUS | Custom US Analysis Workflows
"""
    
def main_cli() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    scan_loader_args(parser)
    seg_loader_args(parser)
    analysis_args(parser)
    args = parser.parse_args()
    args.scan_loader_kwargs = json.loads(args.scan_loader_kwargs)
    args.seg_loader_kwargs = json.loads(args.seg_loader_kwargs)
    args.analysis_kwargs = json.loads(args.analysis_kwargs)
    args.curves_loader_kwargs = json.loads(args.curves_loader_kwargs) if args.curves_loader_kwargs else {}
    args.curve_quant_kwargs = json.loads(args.curve_quant_kwargs) if args.curve_quant_kwargs else {}
    args.visualization_kwargs = json.loads(args.visualization_kwargs) if args.visualization_kwargs else {}

    if hasattr(args, 'curves_path') and args.curves_path is not None:
        return preloaded_pipeline(args)
    
    return core_pipeline(args)    

def main_yaml() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config, **vars(args))
    args.scan_loader_kwargs = {} if args.scan_loader_kwargs is None else args.scan_loader_kwargs
    args.seg_loader_kwargs = {} if args.seg_loader_kwargs is None else args.seg_loader_kwargs
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs
    args.curves_loader_kwargs = {} if args.curves_loader_kwargs is None else args.curves_loader_kwargs
    args.curve_quant_kwargs = {} if args.curve_quant_kwargs is None else args.curve_quant_kwargs
    args.visualization_kwargs = {} if args.visualization_kwargs is None else args.visualization_kwargs

    if hasattr(args, 'curves_path') and args.curves_path is not None:
        return preloaded_pipeline(args)
    
    return core_pipeline(args)

def main_dict(config: dict) -> int:
    """Runs the full QuantUS workflow from a config dictionary.
    
    Args:
        config (dict): Configuration dictionary with all necessary parameters.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = argparse.Namespace(**config)
    args.curve_quant_kwargs = {} if args.curve_quant_kwargs is None else args.curve_quant_kwargs

    if hasattr(args, 'curves_path') and args.curves_path is not None:
        args.curves_loader_kwargs = {} if args.curves_loader_kwargs is None else args.curves_loader_kwargs
        return preloaded_pipeline(args)

    args.scan_loader_kwargs = {} if args.scan_loader_kwargs is None else args.scan_loader_kwargs
    args.seg_loader_kwargs = {} if args.seg_loader_kwargs is None else args.seg_loader_kwargs
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs    
    return core_pipeline(args)

def preloaded_pipeline(args) -> int:
    """Runs the full QuantUS CEUS workflow with preloaded curves.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    curves_loaders = get_curves_loaders()
    quantification_funcs = get_quantification_funcs()
    
    # Get applicable plugins
    try:
        curves_loader = curves_loaders[args.curves_loader]
    except KeyError:
        print(f'Curves loader "{args.curves_loader}" is not available!')
        print(f"Available curves loaders: {', '.join(curves_loaders.keys())}")
        return 1
    
    # Check curve quantification setup
    if not len(args.curve_quant_funcs):
        args.curve_quant_funcs = list(quantification_funcs.keys())
    for func_name in args.curve_quant_funcs:
        if func_name not in quantification_funcs:
            raise ValueError(f"Function '{func_name}' not found in curve quantification functions.\nAvailable functions: {', '.join(quantification_funcs.keys())}")
        required_kwargs = getattr(quantification_funcs[func_name], 'kwarg_names', [])
        for kwarg in required_kwargs:
            if kwarg not in args.curve_quant_kwargs:
                raise ValueError(f"curve_quant_kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")
    
    analysis_obj = curves_loader(args.curves_path, **args.curves_loader_kwargs)
    curve_quant_obj = CurveQuantifications(analysis_obj, args.curve_quant_funcs, args.curve_quant_output_path, **args.curve_quant_kwargs)
    curve_quant_obj.compute_quantifications()
    
    return 0

def core_pipeline(args) -> int:
    """Runs the full QuantUS workflow. Different from entrypoints in that all requirements are checked at the start rather than dynamically.
    """
    scan_loaders = get_scan_loaders()
    seg_loaders = get_seg_loaders()
    analysis_types, analysis_funcs = get_analysis_types()
    quantification_funcs = get_quantification_funcs()
    all_visualization_types, all_visualization_funcs = get_visualization_types()
    
    # Get applicable plugins
    try:
        scan_loader = scan_loaders[args.scan_loader]['cls']
        assert max([args.scan_path.endswith(ext) for ext in scan_loaders[args.scan_loader]['file_exts']]), f"File must end with {scan_loaders[args.scan_loader]['file_exts']}"
    except KeyError:
        print(f'Parser "{args.scan_loader}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    try:
        seg_loader = seg_loaders[args.seg_loader]
    except KeyError:
        print(f'Segmentation loader "{args.seg_loader}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    try:
        analysis_class = analysis_types[args.analysis_type]
    except KeyError:
        print(f'Analysis type "{args.analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(analysis_types.keys())}")
        return 1
    try:
        visualization_class = all_visualization_types[args.visualization_type]
    except KeyError:
        print(f'Visualization type "{args.visualization_type}" is not available!')
        print(f"Available visualization types: {', '.join(all_visualization_types.keys())}")
        return 1
    
    # Check scan paths
    try:
        assertions = [args.scan_path.endswith(ext) for ext in scan_loaders[args.scan_loader]['file_exts']]
        assert max(assertions), f"Scan file must end with {', '.join(scan_loaders[args.scan_loader]['file_exts'])}"
    except KeyError:
        print(f"Scan loader '{args.scan_loader}' does not have defined file extensions.")
        print(f"Available scan loaders: {', '.join(scan_loaders.keys())}")
        return 1

    # Check analysis setup
    if args.analysis_funcs is None:
        args.analysis_funcs = list(analysis_funcs.keys())
    for name in args.analysis_funcs:
        assert analysis_funcs.get(name) is not None, f"Function '{name}' not found in {args.analysis_type} analysis type.\nAvailable functions: {', '.join(analysis_funcs.keys())}"
        analysis_kwargs = getattr(analysis_funcs[name], 'kwarg_names', [])
        for kwarg in analysis_kwargs:
            if kwarg not in args.analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {args.analysis_type} analysis type.")
    
    # Check curve quantification setup
    if not len(args.curve_quant_funcs):
        args.curve_quant_funcs = list(quantification_funcs.keys())
    for func_name in args.curve_quant_funcs:
        if func_name not in quantification_funcs:
            raise ValueError(f"Function '{func_name}' not found in curve quantification functions.\nAvailable functions: {', '.join(quantification_funcs.keys())}")
        required_kwargs = getattr(quantification_funcs[func_name], 'kwarg_names', [])
        for kwarg in required_kwargs:
            if kwarg not in args.curve_quant_kwargs:
                raise ValueError(f"curve_quant_kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")

    # Check visualization inputs
    assert args.visualization_type in all_visualization_types.keys(), f"Visualization type '{args.visualization_type}' not found. Available types: {', '.join(all_visualization_types.keys())}"
    for func_name in args.custom_visualization_funcs:
        if func_name not in all_visualization_funcs.keys():
            raise ValueError(f"Function '{func_name}' not found in visualization functions.\nAvailable functions: {', '.join(all_visualization_funcs.keys())}")

    # Parsing / data loading
    image_data = scan_loader(args.scan_path, **args.scan_loader_kwargs) # Load signal data
    seg_data = seg_loader(image_data, args.seg_path, scan_path=args.scan_path, **args.seg_loader_kwargs) # Load seg data
    analysis_obj = analysis_class(image_data, seg_data, args.analysis_funcs, **args.analysis_kwargs)
    analysis_obj.compute_curves()
    curve_quant_obj = CurveQuantifications(analysis_obj, args.curve_quant_funcs, args.curve_quant_output_path, **args.curve_quant_kwargs)
    curve_quant_obj.compute_quantifications()
    if 'paramap' in args.analysis_type.lower():
        assert 'paramap_folder_path' in args.visualization_kwargs, "paramap_folder_path must be specified in visualization_kwargs for paramap visualizations"
        visualization_obj = visualization_class(curve_quant_obj, args.visualization_params, args.custom_visualization_funcs, **args.visualization_kwargs)
        visualization_obj.generate_visualizations()

    return 0

if __name__ == '__main__':
    exit(main_yaml())
