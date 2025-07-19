import copy

from src.data_objs import UltrasoundImage, CeusSeg
from src.image_loading.options import get_scan_loaders
from src.seg_loading.options import get_seg_loaders
from src.ttc_analysis.options import get_analysis_types
from src.ttc_analysis.ttc_curves.framework import TtcCurvesAnalysis
from src.curve_loading.options import get_curves_loaders
from src.curve_quantification.framework import CurveQuantifications
from src.curve_quantification.options import get_quantification_funcs

def scan_loading_step(scan_type: str, scan_path: str, **scan_loader_kwargs) -> UltrasoundImage:
    """Load the scan data using the specified scan loader.

    Args:
        scan_type (str): The type of scan loader to use.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **scan_loader_kwargs: Additional keyword arguments for the scan loader.

    Returns:
        UltrasoundRfImage: Loaded ultrasound RF image data.
    """
    scan_loaders = get_scan_loaders()
    
    # Find the scan loader
    try:
        scan_loader = scan_loaders[scan_type]['cls']
        assertions = [scan_path.endswith(ext) for ext in scan_loaders[scan_type]['file_exts']]
        assert max(assertions), f"Scan file must end with {', '.join(scan_loaders[scan_type]['file_exts'])}"
    except KeyError:
        print(f'Parser "{scan_type}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    
    image_data: UltrasoundImage = scan_loader(scan_path, **scan_loader_kwargs)
    return image_data

def seg_loading_step(seg_type: str, image_data: UltrasoundImage, seg_path: str,
                     scan_path: str, **seg_loader_kwargs) -> CeusSeg:
    """Load the segmentation data using the specified segmentation loader.

    Args:
        seg_type (str): The type of segmentation loader to use.
        image_data (UltrasoundImage): Loaded ultrasound image data.
        seg_path (str): Path to the segmentation file.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **seg_loader_kwargs: Additional keyword arguments for the segmentation loader.

    Returns:
        CeusSeg: Loaded segmentation data.
    """
    seg_loaders = get_seg_loaders()
    
    # Find the segmentation loader
    try:
        seg_loader = seg_loaders[seg_type]
    except KeyError:
        print(f'Segmentation loader "{seg_type}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    
    return seg_loader(image_data, seg_path, scan_path=scan_path, **seg_loader_kwargs)

def analysis_step(analysis_type: str, image_data: UltrasoundImage, seg_data: CeusSeg, 
                  analysis_funcs: list, **analysis_kwargs) -> TtcCurvesAnalysis:
    """Perform analysis using the specified analysis type.
    
    Args:
        analysis_type (str): The type of analysis to perform.
        image_data (UltrasoundImage): Loaded ultrasound image data.
        config (RfAnalysisConfig): Loaded analysis configuration.
        seg_data (CeusSeg): Loaded segmentation data.
        analysis_funcs (list): List of analysis functions to apply.
        **analysis_kwargs: Additional keyword arguments for the analysis.
    Returns:
        TtcCurvesAnalysis: Analysis object containing the results.
    """
    all_analysis_types, all_analysis_funcs = get_analysis_types()
    
    # Find the analysis class
    try:
        analysis_class = all_analysis_types[analysis_type]
    except KeyError:
        print(f'Analysis type "{analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(all_analysis_types.keys())}")
        return 1
    
    # Check analysis setup
    for name in analysis_funcs:   
        if name not in all_analysis_funcs[analysis_type]:
            raise ValueError(f"Function '{name}' not found in {analysis_type} analysis type.\nAvailable functions: {', '.join(all_analysis_funcs[analysis_type].keys())}")
        required_analysis_kwargs = all_analysis_funcs[analysis_type][name].get('kwarg_names', [])
        for kwarg in required_analysis_kwargs:
            if kwarg not in analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {analysis_type} analysis type.")
            
    # Perform analysis
    analyzed_image_data = copy.deepcopy(image_data)
    
    analysis_obj = analysis_class(analyzed_image_data, seg_data, analysis_funcs, **analysis_kwargs)
    analysis_obj.compute_curves()
    
    return analysis_obj

def load_curves_step(curves_path: str, curves_loader_type: str,
                     **kwargs) -> TtcCurvesAnalysis:
    """Load TTC curves from a specified path.
    
    Args:
        curves_path (str): Path to the CSV file containing TTC curves.
        **kwargs: Additional keyword arguments for loading curves.
    
    Returns:
        TtcCurvesAnalysis: Loaded TTC curves analysis object.
    """
    curves_loaders = get_curves_loaders()

    try:
        curve_loader = curves_loaders[curves_loader_type]
    except KeyError:
        print(f'Curve loader "{curves_loader_type}" is not available!')
        print(f"Available curve loaders: {', '.join(curves_loaders.keys())}")
        return 1

    return curve_loader(curves_path, **kwargs)

def curve_quantification_step(analysis_obj: TtcCurvesAnalysis, function_names: list[str],
                          output_path: str, **kwargs) -> CurveQuantifications:
    """
    Perform curve quantifications using the specified analysis objects and function names.

    Args:
        analysis_obj (TtcCurvesAnalysis): The analysis object containing the curves.
        function_names (List[str]): List of function names to apply for quantification.
        output_path (str): The path to save the output CSV file.
        **kwargs: Additional keyword arguments for the quantification functions.

    Returns:
        Dict[str, float]: A dictionary containing the computed quantifications.
    """
    quant_funcs = get_quantification_funcs()

    if not len(function_names):
        function_names = list(quant_funcs.keys())

    for func_name in function_names:
        if func_name not in quant_funcs.keys():
            raise ValueError(f"Function '{func_name}' not found in quantification functions.\nAvailable functions: {', '.join(quant_funcs.keys())}")
        required_kwargs = getattr(quant_funcs[func_name], 'kwarg_names', [])
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")

    curve_quant = CurveQuantifications(analysis_obj, function_names, output_path, **kwargs)
    curve_quant.compute_quantifications()

    return curve_quant