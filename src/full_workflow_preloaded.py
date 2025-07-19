import json
import yaml
import argparse
from pathlib import Path

from src.image_loading.options import get_scan_loaders, scan_loader_args
from src.seg_loading.options import get_seg_loaders, seg_loader_args
from src.ttc_analysis.options import get_analysis_types, analysis_args
from src.curve_loading.options import get_curves_loaders
from src.curve_quantification.options import get_quantification_funcs
from src.entrypoints import load_curves_step, curve_quantification_step

DESCRIPTION = """
QuantUS | Custom US Analysis Workflows
"""
    
def main_dict(config: dict) -> int:
    """Runs the full QuantUS workflow from a config dictionary.
    
    Args:
        config (dict): Configuration dictionary with all necessary parameters.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = argparse.Namespace(**config)
    args.curves_loader_kwargs = {} if args.curves_loader_kwargs is None else args.curves_loader_kwargs
    args.curve_quant_kwargs = {} if args.curve_quant_kwargs is None else args.curve_quant_kwargs
    
    # Determine workflow type based on configuration
    if hasattr(args, 'curves_path') and args.curves_path is not None:
        return curve_loading_workflow(args)
    else:
        return core_pipeline(args)

def curve_loading_workflow(args) -> int:
    """Runs the curve-loading workflow for pre-computed curves.
    
    Args:
        args: Namespace containing workflow configuration
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    curves_loaders = get_curves_loaders()
    quantification_funcs = get_quantification_funcs()
    
    # Validate curve loader
    try:
        if not hasattr(args, 'curves_loader_type') or args.curves_loader_type is None:
            args.curves_loader_type = 'load_ttc_curves'  # Default
            
        if args.curves_loader_type not in curves_loaders:
            print(f'Curve loader "{args.curves_loader_type}" is not available!')
            print(f"Available curve loaders: {', '.join(curves_loaders.keys())}")
            return 1
    except Exception as e:
        print(f"Error validating curve loader: {e}")
        return 1
    
    # Load curves
    try:
        analysis_obj = load_curves_step(args.curves_path, args.curves_loader_type, **args.curves_loader_kwargs)
        if analysis_obj == 1:  # Error code from load_curves_step
            return 1
    except Exception as e:
        print(f"Error loading curves: {e}")
        return 1
    
    # Perform curve quantification if requested
    if hasattr(args, 'output_path') and args.output_path is not None:
        try:
            # Set default function names if not provided
            if not hasattr(args, 'quantification_functions') or args.quantification_functions is None:
                args.quantification_functions = []  # Empty list will use all functions
                
            curve_quant = curve_quantification_step(
                analysis_obj, 
                args.quantification_functions, 
                args.output_path, 
                **args.curve_quant_kwargs
            )
            
            if curve_quant == 1:  # Error code
                return 1
                
            print(f"Curve quantification completed. Results saved to: {args.output_path}")
            
        except Exception as e:
            print(f"Error during curve quantification: {e}")
            return 1
    else:
        print("Curves loaded successfully. No output path specified for quantification.")
    
    return 0
    
def core_pipeline(args) -> int:
    """Runs the full QuantUS workflow. Different from entrypoints in that all requirements are checked at the start rather than dynamically.
    """
    scan_loaders = get_scan_loaders()
    seg_loaders = get_seg_loaders()
    analysis_types, analysis_funcs = get_analysis_types()
    quantification_funcs = get_quantification_funcs()
    
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
        args.analysis_funcs = list(analysis_funcs[args.analysis_type].keys())
    for name in args.analysis_funcs:
        assert analysis_funcs[args.analysis_type].get(name) is not None, f"Function '{name}' not found in {args.analysis_type} analysis type.\nAvailable functions: {', '.join(analysis_funcs[args.analysis_type].keys())}"
        analysis_kwargs = analysis_funcs[args.analysis_type][name].get('kwarg_names', [])
        for kwarg in analysis_kwargs:
            if kwarg not in args.analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {args.analysis_type} analysis type.")
    
    # Parsing / data loading
    image_data = scan_loader(args.scan_path, **args.scan_loader_kwargs) # Load signal data
    seg_data = seg_loader(image_data, args.seg_path, scan_path=args.scan_path, **args.seg_loader_kwargs) # Load seg data
    analysis_obj = analysis_class(image_data, seg_data, args.analysis_funcs, **args.analysis_kwargs)
    analysis_obj.compute_curves()
    
    # Perform curve quantification if requested
    if hasattr(args, 'output_path') and args.output_path is not None:
        try:
            # Set default function names if not provided
            if not hasattr(args, 'quantification_functions') or args.quantification_functions is None:
                args.quantification_functions = []  # Empty list will use all functions
                
            curve_quant = curve_quantification_step(
                analysis_obj, 
                args.quantification_functions, 
                args.output_path, 
                **args.curve_quant_kwargs
            )
            
            if curve_quant == 1:  # Error code
                return 1
                
            print(f"Analysis and curve quantification completed. Results saved to: {args.output_path}")
            
        except Exception as e:
            print(f"Error during curve quantification: {e}")
            return 1
    else:
        print("Analysis completed successfully. No output path specified for quantification.")
    
    return 0

def main_yaml(yaml_path: str = None) -> int:
    """Main function to run workflow from YAML configuration file.
    
    Args:
        yaml_path (str): Path to YAML configuration file.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    if yaml_path is None:
        # Could be provided via command line argument or default
        parser = argparse.ArgumentParser(description=DESCRIPTION)
        parser.add_argument('config_path', help='Path to YAML configuration file')
        args = parser.parse_args()
        yaml_path = args.config_path
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return main_dict(config)
    except FileNotFoundError:
        print(f"Configuration file not found: {yaml_path}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return 1
    except Exception as e:
        print(f"Error running workflow: {e}")
        return 1

def main_json(json_path: str = None) -> int:
    """Main function to run workflow from JSON configuration file.
    
    Args:
        json_path (str): Path to JSON configuration file.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    if json_path is None:
        # Could be provided via command line argument or default
        parser = argparse.ArgumentParser(description=DESCRIPTION)
        parser.add_argument('config_path', help='Path to JSON configuration file')
        args = parser.parse_args()
        json_path = args.config_path
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        return main_dict(config)
    except FileNotFoundError:
        print(f"Configuration file not found: {json_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return 1
    except Exception as e:
        print(f"Error running workflow: {e}")
        return 1

def create_curve_loading_config(curves_path: str, output_path: str = "quantification_results.csv",
                               curves_loader_type: str = "load_ttc_curves",
                               curves_to_fit: list = None, n_frames_to_analyze: int = 100) -> dict:
    """Create a configuration dictionary for curve loading workflow.
    
    Args:
        curves_path (str): Path to the CSV file containing pre-computed curves
        output_path (str): Path where quantification results will be saved
        curves_loader_type (str): Type of curve loader to use
        curves_to_fit (list): List of curve names to fit (default: ['moderate_diagnostics_Image-original_Mean'])
        n_frames_to_analyze (int): Number of frames to analyze
        
    Returns:
        dict: Configuration dictionary ready for main_dict()
    """
    if curves_to_fit is None:
        curves_to_fit = ['moderate_diagnostics_Image-original_Mean']
    
    return {
        'curves_path': curves_path,
        'curves_loader_type': curves_loader_type,
        'curves_loader_kwargs': {},
        'output_path': output_path,
        'quantification_functions': [],
        'curve_quant_kwargs': {
            'curves_to_fit': curves_to_fit,
            'n_frames_to_analyze': n_frames_to_analyze
        }
    }

def create_full_analysis_config(scan_path: str, scan_loader: str, seg_path: str, seg_loader: str,
                               analysis_type: str = "ttc_curves", output_path: str = None) -> dict:
    """Create a configuration dictionary for full analysis workflow.
    
    Args:
        scan_path (str): Path to the scan file
        scan_loader (str): Type of scan loader to use
        seg_path (str): Path to the segmentation file
        seg_loader (str): Type of segmentation loader to use
        analysis_type (str): Type of analysis to perform
        output_path (str): Path where results will be saved (optional)
        
    Returns:
        dict: Configuration dictionary ready for main_dict()
    """
    config = {
        'scan_path': scan_path,
        'scan_loader': scan_loader,
        'scan_loader_kwargs': {},
        'seg_path': seg_path,
        'seg_loader': seg_loader,
        'seg_loader_kwargs': {},
        'analysis_type': analysis_type,
        'analysis_funcs': None,
        'analysis_kwargs': {},
        'curves_loader_kwargs': {},
        'curve_quant_kwargs': {}
    }
    
    if output_path is not None:
        config.update({
            'output_path': output_path,
            'quantification_functions': [],
            'curve_quant_kwargs': {
                'curves_to_fit': ['moderate_diagnostics_Image-original_Mean'],
                'n_frames_to_analyze': 100
            }
        })
    
    return config

if __name__ == '__main__':
    # Try to determine config file type from command line
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            exit(main_yaml(config_path))
        elif config_path.endswith('.json'):
            exit(main_json(config_path))
        else:
            print("Unsupported config file format. Use .yaml, .yml, or .json")
            exit(1)
    else:
        print("Usage: python full_workflow_preloaded.py <config_file>")
        exit(1)
