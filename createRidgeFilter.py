import math
import os
from typing import List
# In this file we will be created a ridgefilter, through a text file that topas will read and produce.

def ridgepin(hlx_values: List[float], hlz_max: float = 3, material: str = "G4_WATER", file_name: str = "ridgepin.txt", header_text: str = "", units: str = "mm") -> List[str]:
    """
    Creates a layered structure with varying HLX values, each layer stacked on top of the previous one.
    Optionally adds a piece of text at the beginning of the file.

    Args:
        hlx_values (List[float]): A list of half-length X values (HLX) for each layer.
        hlz_max (float): The total HLZ height for all layers combined.
        material (str): The material of each layer. Default is "G4_WATER".
        file_name (str): The name of the file to write the configuration to.
        header_text (str): Optional text to be added at the beginning of the file.
    
    Returns:
        List[str]: A list of strings defining each layer in the desired format.
    """
    # Define the directory where the file will be saved
    directory = "ridgepins"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full file path within the subdirectory
    file_path = os.path.join(directory, file_name)

    hlz_value = hlz_max / len(hlx_values)  # Divide the total HLZ by the number of layers to get the HLZ for each layer
    structure = []

    with open(file_path, 'w') as file:
        # Write the optional header text if provided
        if header_text:
            file.write(f"{header_text}\n\n")
        beginning_text = [
            "# Ridge Filter Pin Configuration",
            f"# Total HLZ: {hlz_max} {units}",
            f"# Material: {material}",
            f"# HLX values: {hlx_values}",
            f"# HLZ value for each layer: {hlz_value} {units}",
            "",
            "#Define a world of 1m^3",
            "s:Ge/World_RF/Material  = \"Air\"",
            f"d:Ge/World_RF/HLX       = {max(hlx_values)} {units}",
            f"d:Ge/World_RF/HLY       = {max(hlx_values)} {units}",
            f"d:Ge/World_RF/HLZ       = {hlz_max} {units}",
            "",
            "# Define the graphics",
            "s:Gr/ViewA/Type              = \"OpenGL\"",
            "b:Gr/ViewA/IncludeAxes = \"True\"",
            f"d:Gr/ViewA/AxesSize = {2*max(hlx_values)} {units}",
            "Ts/UseQt = \"True\"",
            "",

        ]
        file.write("\n".join(beginning_text) + "\n")    
        for i, hlx in enumerate(hlx_values):
            layer_num = i + 1        
            layer_definitions = [
                f"# Layer {layer_num}",
                f"s:Ge/ridgefilterpinlayer{layer_num}/parent = \"World_RF\"",
                f"s:Ge/ridgefilterpinlayer{layer_num}/type = \"G4Trd\"",
                f"d:Ge/ridgefilterpinlayer{layer_num}/hlx1 = {hlx} {units}",
                f"d:Ge/ridgefilterpinlayer{layer_num}/hlx2 = {hlx} {units}",  # Assuming hlx2 is half of hlx1 for each layer
                f"d:Ge/ridgefilterpinlayer{layer_num}/hly1 = Ge/ridgefilterpinlayer{layer_num}/hlx1 {units}",
                f"d:Ge/ridgefilterpinlayer{layer_num}/hly2 = Ge/ridgefilterpinlayer{layer_num}/hlx2 {units}",
                f"d:Ge/ridgefilterpinlayer{layer_num}/hlz = {hlz_value} {units}",
                f"s:Ge/ridgefilterpinlayer{layer_num}/material = \"{material}\"",
                f"d:Ge/ridgefilterpinlayer{layer_num}/transz = {round(i*hlz_value,2)} {units}",
                ""
            ]
            
            structure.extend(layer_definitions)
            structure.append("\n")  # Add a newline between layer definitions for readability
            # Write each layer's definitions to the file
            file.write("\n".join(layer_definitions) + "\n")
        return structure
def is_monotone_decreasing(lst: List[float]) -> bool:
    """
    Checks if the given list of floats is monotone decreasing.

    Args:
        lst (List[float]): A list of floats.

    Returns:
        bool: True if the list is monotone decreasing, False otherwise.
    """
    return all(lst[i] >= lst[i+1] for i in range(len(lst) - 1))

def place_ridgepins_xy(hlx_values: List[float], n_x: int = 16, 
                       n_y: int= 16, spacing_x: float = 1 , spacing_y: float = 1, 
                       hlz_max: float = 30, material: str = "G4_WATER", 
                       file_name: str = "ridgepins_xy.txt", units: str = "mm", 
                       header_text: str = "", view_graphics: bool = False, 
                       width_x: float = 33, width_y: float = 33):
    """
    Places 'n_x' ridgepins along the X-axis and 'n_y' ridgepins along the Y-axis in a grid pattern.
    Adjusts the transx and transy positions to center the grid around the origin.

    Args:
        hlx_values (List[float]): A list of half-length X values (HLX) for each layer.
        n_x (int): Number of ridgepins to place along the X-axis.
        n_y (int): Number of ridgepins to place along the Y-axis.
        spacing_x (float): Horizontal spacing between ridgepins along the X-axis.
        spacing_y (float): Vertical spacing between ridgepins along the Y-axis.
        hlz_max (float): The total HLZ height for all layers combined.
        material (str): The material of each layer. Default is "G4_WATER".
        file_name (str): The name of the file to write the configuration to.
        units (str): Units for the dimensions. Default is "mm".
        header_text (str): Optional text to be added at the beginning of the file.

    Returns:
        None
    """
    # Define the directory where the file will be saved
    directory = "ridgepins"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full file path within the subdirectory
    file_path = os.path.join(directory, file_name)

    hlz_value = hlz_max / len(hlx_values)  # Divide the total HLZ by the number of layers to get the HLZ for each layer
    max_hlx = max(hlx_values)  # The largest HLX value determines the half-width of each ridgepin
    total_width_x = n_x * (2 * max_hlx) + (n_x - 1) * spacing_x  # Total width of all ridgepins plus spacing in X direction
    total_width_y = n_y * (2 * max_hlx) + (n_y - 1) * spacing_y  # Total width of all ridgepins plus spacing in Y direction
   
    # Making sure the total width is always atleast width of width_x so no escaping the box
    if total_width_x < width_x:
        spacing_x = (width_x - n_x * 2 * max_hlx) / (n_x - 1)
        total_width_x = width_x
    if total_width_y < width_y:
        spacing_y = (width_y - n_y * 2 * max_hlx) / (n_y - 1)
        total_width_y = width_y
    center_offset_x = total_width_x / 2  # Offset to center the structure around the X-axis
    center_offset_y = total_width_y / 2  # Offset to center the structure around the Y-axis

    with open(file_path, 'w') as file:
        # Write the optional header text if provided
        if header_text:
            file.write(f"{header_text}\n\n")
        beginning_text = [
            "# Ridge Filter Pin Configuration",
            f"# Total HLZ: {hlz_max} {units}",
            f"# Material: {material}",
            f"# HLX values: {hlx_values}",
            f"# HLZ value for each layer: {hlz_value} {units}",
            f"# Number of ridgepins in the x direction: {n_x}",
            f"# Number of ridgepins in the y direction: {n_y}",
            "",
            "# Define a world",
            "s:Ge/World_RF/Material  = \"Air\"",
            f"d:Ge/World_RF/HLX       = {total_width_x} {units}",
            f"d:Ge/World_RF/HLY       = {total_width_y} {units}",
            f"d:Ge/World_RF/HLZ       = {hlz_max} {units}",
            "s:ge/World_RF/Type = \"TsBox\"",
            "",
            "# Define box under the ridgepins with width 5 mm",
            "s:Ge/box/Material = \"G4_WATER\"",
            "s:Ge/box/Type = \"TsBox\"",
            f"d:Ge/box/HLX = {0.5*total_width_x+1} {units}",
            f"d:Ge/box/HLY = {0.5*total_width_y+1} {units}",
            f"d:Ge/box/HLZ = 2.5 mm",
            f"d:Ge/box/TransZ = -10 mm",
            "s:ge/bpx/color = \"blue\"",
            "s:Ge/box/drawingstyle = \"Solid\"",
            "s:Ge/box/Parent = \"World_RF\"",
            "",
        ]
        if view_graphics:
            beginning_text.extend([
                "# Define the graphics",
                "s:Gr/ViewA/Type              = \"OpenGL\"",
                "b:Gr/ViewA/IncludeAxes = \"True\"",
                f"d:Gr/ViewA/AxesSize = {2*max_hlx} {units}",
                "Ts/UseQt = \"True\"",
                ""
            ])
        file.write("\n".join(beginning_text) + "\n")

        for pin_x in range(n_x):  # Loop for creating 'n_x' ridgepins along the X-axis
            transx = pin_x * (max_hlx * 2 + spacing_x) - center_offset_x + max_hlx  # Compute the X translation for each ridgepin along X

            for pin_y in range(n_y):  # Loop for creating 'n_y' ridgepins along the Y-axis
                transy = pin_y * (max_hlx * 2 + spacing_y) - center_offset_y + max_hlx  # Compute the Y translation for each ridgepin along Y

                for i, hlx in enumerate(hlx_values):  # For each layer
                    layer_num = i + 1
                    layer_definitions = [
                        f"# Ridgepin ({pin_x + 1}, {pin_y + 1}) - Layer {layer_num}",
                        f"s:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/parent = \"World_RF\"",
                        f"s:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/type = \"G4Trd\"",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hlx1 = {hlx} {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hlx2 = {hlx} {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hly1 = Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hlx1 {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hly2 = Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hlx2 {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/hlz = {hlz_value} {units}",
                        f"s:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/material = \"{material}\"",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/transz = {(i * hlz_value)} {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/transx = {(transx)} {units}",
                        f"d:Ge/ridgefilterpin{pin_x+1}_{pin_y+1}layer{layer_num}/transy = {(transy)} {units}",
                        
                        ""
                    ]

                    # Write each layer's definitions to the file
                    file.write("\n".join(layer_definitions) + "\n")

    print(f"Ridgepins configuration saved to {file_path}")

def create_ridgefilter(hlx_values: List[float], file_name: str = "ridgepins_main.txt", spacing_x: float = 1 , spacing_y: float = 1, 
                       hlz_max: float = 30, material: str = "G4_WATER", 
                       units: str = "mm", 
                       header_text: str = "", view_graphics: bool = False, 
                       width_x: float = 33, width_y: float = 33):
    #if not is_monotone_decreasing(ridge_pin_coordinates):
    #    raise ValueError('The ridge pin coordinates are not in decreasing order')
    create_ridgefilter_geometry_coordinates(hlx_values, hlz_max= hlz_max,material = material, units = units, header_text=header_text, file_name= file_name, spacing_x=spacing_x, spacing_y=spacing_y, view_graphics=False )

def create_ridgefilter_geometry_coordinates(hlx_values: List[float], n_x: int = 33, 
                       n_y: int= 33, spacing_x: float = 1,  spacing_y: float = 1, 
                       hlz_max: float = 30, material: str = "G4_WATER", 
                       file_name: str = "ridgepins_main.txt", units: str = "mm", 
                       header_text: str = "", view_graphics: bool = False, 
                       filter_base_height: float = 5):
    """
     G4double fRidgepinMaxHeight;
    G4double fRidgepinSpacing;
    G4double fRidgeFilterBaseHeight, 

     world_HZ, fRidgeFilterWidth_X, fRidgeFilterWidth_Y, fRidgePinMaxWidth, fRidgePinHeightStep;
    Places 'n_x' ridgepins along the X-axis and 'n_y' ridgepins along the Y-axis in a grid pattern.
    Adjusts the transx and transy positions to center the grid around the origin.

    Args:
        hlx_values (List[float]): A list of half-length X values (HLX) for each layer.
        n_x (int): Number of ridgepins to place along the X-axis.
        n_y (int): Number of ridgepins to place along the Y-axis.
        spacing_x (float): Horizontal spacing between ridgepins along the X-axis.
        spacing_y (float): Vertical spacing between ridgepins along the Y-axis.
        hlz_max (float): The total HLZ height for all layers combined.
        material (str): The material of each layer. Default is "G4_WATER".
        file_name (str): The name of the file to write the configuration to.
        units (str): Units for the dimensions. Default is "mm".
        header_text (str): Optional text to be added at the beginning of the file.

    Returns:
        None
    """
    # Define the directory where the file will be saved
    directory = "ridgepins"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full file path within the subdirectory
    file_path = os.path.join(directory, file_name)

    amount_of_element = len(hlx_values)
    hlz_value = hlz_max / len(hlx_values)  # Divide the total HLZ by the number of layers to get the HLZ for each layer
    max_hlx = max(hlx_values)  # The largest HLX value determines the half-width of each ridgepin
    total_width_x = n_x * (2 * max_hlx) + (n_x - 1) * spacing_x  # Total width of all ridgepins plus spacing in X direction
    total_width_y = n_y * (2 * max_hlx) + (n_y - 1) * spacing_y  # Total width of all ridgepins plus spacing in Y direction
   
    hlx_coordinate_conversion = str(amount_of_element)
    for hlx in hlx_values:
        hlx_coordinate_conversion += " " + str(hlx)

    with open(file_path, 'w') as file:
        # Write the optional header text if provided
        if header_text:
            file.write(f"{header_text}\n\n")
        beginning_text = [
            "# Ridge Filter Pin Configuration",
            f"# Total HLZ: {hlz_max + filter_base_height} {units}",
            f"# Material: {material}",
            f"# HLX values: {hlx_values}",
            f"# HLZ value for each layer: {hlz_value} {units}",
            f"# Number of ridgepins in the x direction: {n_x}",
            f"# Number of ridgepins in the y direction: {n_y}",
            "",
            "# Define a world",
            "s:Ge/World_RF/Material  = \"Air\"",
            f"d:Ge/World_RF/HLX       = {2*total_width_x} {units}",
            f"d:Ge/World_RF/HLY       = {2*total_width_y} {units}",
            f"d:Ge/World_RF/HLZ       = {2*hlz_max} {units}",
            "s:ge/World_RF/Type = \"TsBox\"",
            "",
            "# Define the RF using geometry component",
            "s:Ge/RidgePin/Parent = \"World_RF\"",
            f"s:Ge/RidgePin/Material= \"{material}\"",
            f"s:Ge/RidgePin/Type = \"TsRidgepin\"",
            "s:Ge/RidgePin/DrawingStyle = \"Solid\"",
            f"d:Ge/Ridgepin/RidgepinMaxHeight = {hlz_max} {units}",
            f"d:Ge/Ridgepin/RidgepinSpacing = {spacing_x} {units}",
            f"dv:Ge/Ridgepin/RidgepinWidths = {hlx_coordinate_conversion} {units}",
            f"i:ge/Ridgepin/RidgepinElements = {amount_of_element}",	
            f"i:Ge/Ridgepin/RidgeFilterNumberOfPinsX = {n_x}",
            f"i:Ge/Ridgepin/RidgeFilterNumberOfPinsY = {n_y}",
            f"d:Ge/Ridgepin/RidgeFilterBaseHeight = {filter_base_height} {units}",
            f"d:Ge/RidgePin/RidgepinBaseWidth = {max_hlx} {units}",
            "s:Ge/RidgePin/WorldLog/Material= \"Air\"",
            f"s:Ge/RidgePin/baseLayerLog/Material = \"G4_WATER\"",
            f"s:Ge/RidgePin/ridgepinLog/Material = \"G4_WATER\"",
            "",
        ]
        if view_graphics:
            beginning_text.extend([
                "# Define the graphics",
                "s:Gr/ViewA/Type              = \"OpenGL\"",
                "b:Gr/ViewA/IncludeAxes = \"True\"",
                f"d:Gr/ViewA/AxesSize = {2*max_hlx} {units}",
                "Ts/UseQt = \"True\"",
                ""
            ])
        file.write("\n".join(beginning_text) + "\n")

       

    print(f"Ridgepins configuration saved to {file_path}")
hlx_values = [2.74881284, 2.44901661, 2.38651827, 2.33233216, 2.10345351, 2.08674063,
 1.84900484, 1.64783538, 1.57007964, 1.4994695,  1.44250786, 1.30511209,
 1.3018969,  1.1880049,  0.89106094, 0.89106094, 0.2142218,  0.18530209,
 0.07451673, 0.03029631]  # Example HLX values for 4 layers
print(hlx_values)
n_x = n_y = 16  # Number of ridgepins to place
spacing_x = spacing_y = 1  # Spacing between ridgepins in mm
hlz_max = 30  # Total HLZ for all layers
material = "G4_WATER"
print(is_monotone_decreasing(hlx_values))
# place_ridgepins_xy(hlx_values, n_x,n_y, spacing_x, spacing_y, hlz_max, material, "ridgepins.txt")
create_ridgefilter(hlx_values, hlz_max= hlz_max,material = material, file_name= "ridgepins_main.txt", spacing_x=spacing_x, spacing_y=spacing_y, view_graphics=False )