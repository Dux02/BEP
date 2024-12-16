from topas2numpy import BinnedResult
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple
import json
from collections import Counter,defaultdict
import matplotlib.cm as cm
from itertools import islice
import datetime as dt
def dose_info(dose: BinnedResult):
    print(dose.data.keys())
    print('{0} [{1}]'.format(dose.quantity, dose.unit))
    print('Statistics: {0}'.format(dose.statistics))
    for dim in dose.dimensions:
        print('{0} [{1}]: {2} bins'.format(dim.name, dim.unit, dim.n_bins))

def normalize_data(dose: BinnedResult) -> np.ndarray:
    max_dose = np.max(np.squeeze(dose.data['Sum']))
    normalized_data = np.squeeze(dose.data['Sum']) / max_dose
    return normalized_data

def import_data_from_file(file_path: str) -> tuple:
    dose = BinnedResult(file_path)
    x_data = dose.dimensions[2].get_bin_centers()[::-1]
    y_data = normalize_data(dose)
    return dose, x_data, y_data

def normalized_data_from_dose(dose: BinnedResult) -> tuple:
    x_data = dose.dimensions[2].get_bin_centers()[::-1]
    y_data = normalize_data(dose)
    return x_data, y_data

def matplot_plotting(
        data_x, data_y, save_path: str, figure_title: str, 
        figure_filename: str, x_label: str = "Depth [cm]", 
        y_label: str =  "Normalized Dose", x_max: float = 17, 
        y_max: float = 1.1, save_fig: bool = True,
        show_fig: bool = False, plot_edges: bool = True, distal_edge: float = 14.9, proximal_edge: float = 10.9):
    #Makes the save folder if it does not exist.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the file already exists and modify the filename if necessary
    full_path = os.path.join(save_path, figure_filename)
    base, ext = os.path.splitext(full_path)
    counter = 1
    while os.path.exists(full_path):
        full_path = f"{base}_{counter}{ext}"
        counter += 1

    if '.png' not in full_path:
        full_path = full_path + '.png'

    plt.figure()
    
    if plot_edges:
        plt.axvline(x=distal_edge, color='r', linestyle='--')
        plt.axvline(x=proximal_edge, color='r', linestyle='--')
    ax = plt.subplot(111)
    plt.plot(data_x, data_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xlim(xmax=x_max)
    ax.set_ylim(ymax=y_max)
    plt.title(figure_title)
    if save_fig:
        plt.savefig(full_path, format='png', dpi = 300)
    if show_fig:
        plt.show()
    plt.close()

def simple_data_analysis(dose) -> Tuple[float, float, float, float, float]:
    x_data = dose.dimensions[2].get_bin_centers()[::-1]
    y_data = np.squeeze(dose.data['Sum'])
    max_dose = np.max(np.squeeze(dose.data['Sum']))
    min_dose = np.min(np.squeeze(dose.data['Sum']))
    mean_dose = np.mean(np.squeeze(dose.data['Sum']))
    std_dose = np.std(np.squeeze(dose.data['Sum']))
    std_error_of_the_mean = std_dose/np.sqrt(len(dose.data['Sum']))
    
    x_y_pairs = zip(x_data, y_data)
    for x, y in x_y_pairs:
        if y == max_dose:
            print(f"The max dose is at {x} cm")
            x_location_max = x
    print(f"The max dose: {max_dose} {dose.unit}")
    print(f"The min dose: {min_dose} {dose.unit}")
    print(f"The mean dose: {mean_dose} {dose.unit}")
    print(f"The std dose: {std_dose} {dose.unit}")
    print(f"The std error of the mean: {std_error_of_the_mean} {dose.unit}")
    return max_dose, min_dose, mean_dose, std_dose, std_error_of_the_mean, x_location_max

def pointwise_difference(dose1, dose2) -> np.ndarray:
    x_data_1, y_data_1 = normalized_data_from_dose(dose1)
    x_data_2, y_data_2 = normalized_data_from_dose(dose2)
    if x_data_1 != x_data_2:
        raise ValueError('The x_data of the two doses are not the same')
    else:
        x_data = x_data_1
    pointwise_dose_difference = y_data_2 - y_data_1
    return x_data, pointwise_dose_difference

def relative_difference(dose1, dose2) -> np.ndarray:
    x_data_1, y_data_1 = normalized_data_from_dose(dose1)
    x_data_2, y_data_2 = normalized_data_from_dose(dose2)
    if x_data_1 != x_data_2:
        raise ValueError('The x_data of the two doses are not the same')
    else:
        x_data = x_data_1
    pointwise_dose_difference = y_data_2 - y_data_1
    relative_dose_difference = pointwise_dose_difference/y_data_1
    return x_data, relative_dose_difference

def plot_bragpeak_pointwise_difference(dose1, dose2, title: str =""):
    x_data, pointwise_dose_difference = pointwise_difference(dose1, dose2)
    matplot_plotting(x_data, pointwise_dose_difference, 'images/pointwise_difference/', title, title)
    simple_data_analysis(dose1)
    simple_data_analysis(dose2)
    
def plot_bragpeak_relative_difference(dose1, dose2, title= ""):
    x_data, relative_dose_difference = relative_difference(dose1, dose2)
    matplot_plotting(x_data, relative_dose_difference, 'images/relative_difference/', title, title)
    simple_data_analysis(dose1)
    simple_data_analysis(dose2)
    
def plot_bragpeak(dose, title=''):
    x_data, y_data = normalized_data_from_dose(dose)
    matplot_plotting(x_data, y_data, 'images/', title, title)

def figure_title_maker(file_path: str) -> str:
    figure_title = file_path.split('\\')[-1].split('.')[0].replace('_', ' ').capitalize()
    return figure_title

def filter_data_by_range(x_data, y_data, x_min, x_max):
    filtered_x_data = [x for x in x_data if x_min <= x <= x_max]
    filtered_y_data = [y for x, y in zip(x_data, y_data) if x_min <= x <= x_max]
    return filtered_x_data, filtered_y_data

def mean_square_error(y_data_1, y_data_2) -> float:
    abs_diff = np.abs(np.subtract(y_data_1, y_data_2))
    mse_unsummed = np.square(abs_diff)/len(y_data_1)
    mse = np.sum(mse_unsummed)
    return mse, abs_diff, mse_unsummed

def maximum_error(y_data_1, y_data_2) -> float:
    abs_diff = np.abs(np.subtract(y_data_1, y_data_2))
    max_diff = np.max(abs_diff)
    return max_diff, abs_diff

def fit_function_new(file_name, proximal_edge: float = 10.9, distal_edge: float = 14.9, weight: float = 1, debug: bool = False, max_error: bool = False, automatic_edges: bool = False, sqrt_fit_val: bool = False) -> float:
    # New function to calculate the fit value, includes options for debugging.
    dose, x_data, y_data = import_data_from_file(file_name) # Importing the data from the file
    normalized_y_data = normalize_data(dose) # Normalizing the data
    
    if debug:
        print(f"The proximal edge is: '{proximal_edge}', and the distal edge is: '{distal_edge}'.")
        print(f"The x_data is: '{x_data[::-1]}', and the y_data is: '{y_data}'.")
        simple_data_analysis(dose)
    
    if automatic_edges: # DO NOT USE!!!!!!, this is not a 
        max_dose, min_dose, mean_dose, std_dose, std_error, x_location_max = simple_data_analysis(dose)
        
        proximal_edge = x_location_max - 3
        distal_edge = x_location_max + 3
        print(f"The proximal edge is: '{proximal_edge}', and the distal edge is: '{distal_edge}'")
         
    # Splitting the data into the SOBP and the fall off
    x_data_plateau, y_data_plateau = filter_data_by_range(x_data, normalized_y_data, proximal_edge, distal_edge) #Filters for the SOBP Plateau
    x_data_fall_off, y_data_fall_off = filter_data_by_range(x_data, normalized_y_data, distal_edge, np.max(x_data)) #Filters for the fall off
    
    # Describe desired SOBP and optionally fall off
    y_desired_plateau = np.ones(len(y_data_plateau))
    x_desired_plateau = x_data_plateau
    
    y_desired_fall_of = np.zeros(len(y_data_fall_off))
    x_desired_fall_of = x_data_fall_off
    
    # Calculate the mean square error for the SOBP and the fall off
    mse_plateau, abs_diff_plateau, mse_unsummed_plateau = mean_square_error(y_data_plateau, y_desired_plateau)
    mse_fall_off, abs_diff_fall_off, mse_unsummed_fall_off = mean_square_error(y_data_fall_off, y_desired_fall_of)
    
    # Taking the square root of the mean square error for a more intuitive value
    mse_plateau_sqrt = np.sqrt(mse_plateau)
    mse_fall_off_sqrt = np.sqrt(mse_fall_off)
    
    # Calculate the max error for the SOBP and the fall of
    max_error_plateau, abs_diff_plateau = maximum_error(y_data_plateau, y_desired_plateau)
    max_error_fall_off, abs_diff_fall_off = maximum_error(y_data_fall_off, y_desired_fall_of)
    
    # Taking the square root of the max error for a more intuitive value
    max_error_plateau_sqrt = np.sqrt(max_error_plateau)
    max_error_fall_off_sqrt = np.sqrt(max_error_fall_off)
    
    if debug:
        print(f"The mean square error for the SOBP is: '{mse_plateau}', and the mean square error for the fall off is: '{mse_fall_off}'.")
        print(f"The square root of the mean square error for the SOBP is: '{mse_plateau_sqrt}', and the square root of the mean square error for the fall off is: '{mse_fall_off_sqrt}'.")
        print(f"The max error for the SOBP is: '{max_error_plateau}', and the max error for the fall off is: '{max_error_fall_off}'.")
        print(f"The square root of the max error for the SOBP is: '{max_error_plateau_sqrt}', and the square root of the max error for the fall off is: '{max_error_fall_off_sqrt}'.")
        print(f"The abs diff for the SOBP is: '{abs_diff_plateau}', and the abs diff for the fall off is: '{abs_diff_fall_off}'.")   

    
    if max_error:
        if sqrt_fit_val:
            fit_value = max_error_plateau_sqrt
        else:
            fit_value = max_error_plateau
    else:
        if sqrt_fit_val:
            fit_value = mse_plateau_sqrt
        else:
            fit_value = mse_plateau
    
    return fit_value


def fit_function_new_bone(file_name, proximal_edge: float = 10.9, distal_edge: float = 14.9, weight: float = 1, debug: bool = False, max_error: bool = False, automatic_edges: bool = False, sqrt_fit_val: bool = False) -> float:
    # New function to calculate the fit value, includes options for debugging.
    dose, x_data, y_data = import_data_from_file(file_name) # Importing the data from the file
    y_data_sum = y_data[0] + y_data[1]

    normalized_y_data = y_data_sum / np.max(y_data_sum)
    if debug:
        print(f"The proximal edge is: '{proximal_edge}', and the distal edge is: '{distal_edge}'.")
        print(f"The x_data is: '{x_data[::-1]}', and the y_data is: '{y_data}'.")
        simple_data_analysis(dose)
    
    if automatic_edges: # DO NOT USE!!!!!!, this is not a 
        max_dose, min_dose, mean_dose, std_dose, std_error, x_location_max = simple_data_analysis(dose)
        
        proximal_edge = x_location_max - 3
        distal_edge = x_location_max + 3
        print(f"The proximal edge is: '{proximal_edge}', and the distal edge is: '{distal_edge}'")
         
    # Splitting the data into the SOBP and the fall off
    x_data_plateau, y_data_plateau = filter_data_by_range(x_data, normalized_y_data, proximal_edge, distal_edge) #Filters for the SOBP Plateau
    x_data_fall_off, y_data_fall_off = filter_data_by_range(x_data, normalized_y_data, distal_edge, np.max(x_data)) #Filters for the fall off
    
    # Describe desired SOBP and optionally fall off
    y_desired_plateau = np.ones(len(y_data_plateau))
    x_desired_plateau = x_data_plateau
    
    y_desired_fall_of = np.zeros(len(y_data_fall_off))
    x_desired_fall_of = x_data_fall_off
    
    # Calculate the mean square error for the SOBP and the fall off
    mse_plateau, abs_diff_plateau, mse_unsummed_plateau = mean_square_error(y_data_plateau, y_desired_plateau)
    mse_fall_off, abs_diff_fall_off, mse_unsummed_fall_off = mean_square_error(y_data_fall_off, y_desired_fall_of)
    
    # Taking the square root of the mean square error for a more intuitive value
    mse_plateau_sqrt = np.sqrt(mse_plateau)
    mse_fall_off_sqrt = np.sqrt(mse_fall_off)
    
    # Calculate the max error for the SOBP and the fall of
    max_error_plateau, abs_diff_plateau = maximum_error(y_data_plateau, y_desired_plateau)
    max_error_fall_off, abs_diff_fall_off = maximum_error(y_data_fall_off, y_desired_fall_of)
    
    # Taking the square root of the max error for a more intuitive value
    max_error_plateau_sqrt = np.sqrt(max_error_plateau)
    max_error_fall_off_sqrt = np.sqrt(max_error_fall_off)
    
    if debug:
        print(f"The mean square error for the SOBP is: '{mse_plateau}', and the mean square error for the fall off is: '{mse_fall_off}'.")
        print(f"The square root of the mean square error for the SOBP is: '{mse_plateau_sqrt}', and the square root of the mean square error for the fall off is: '{mse_fall_off_sqrt}'.")
        print(f"The max error for the SOBP is: '{max_error_plateau}', and the max error for the fall off is: '{max_error_fall_off}'.")
        print(f"The square root of the max error for the SOBP is: '{max_error_plateau_sqrt}', and the square root of the max error for the fall off is: '{max_error_fall_off_sqrt}'.")
        print(f"The abs diff for the SOBP is: '{abs_diff_plateau}', and the abs diff for the fall off is: '{abs_diff_fall_off}'.")   

    
    if max_error:
        if sqrt_fit_val:
            fit_value = max_error_plateau_sqrt
        else:
            fit_value = max_error_plateau
    else:
        if sqrt_fit_val:
            fit_value = mse_plateau_sqrt
        else:
            fit_value = mse_plateau
    
    return fit_value

def fit_functions(x_data, y_data, distal_edge, proximal_edge, weight, debug: bool = False) -> float:
    x_max = np.max(x_data)

    # Filtering the data for the SOBP and the fall off
    filtered_x_data_sobp, filtered_y_data_sobp = filter_data_by_range(x_data, y_data, proximal_edge, distal_edge) #Filters for the SOBP
    filtered_x_data_fall_off, filtered_y_data_fall_off = filter_data_by_range(x_data, y_data, distal_edge, x_max) #Filters for the fall off
    
    #print(x_data)
    
    
    fit_value_1, abs_diff_1, mse_unsummed_1 = mean_square_error(filtered_y_data_sobp, np.ones(len(filtered_y_data_sobp)))
    fit_value_2, abs_diff_2, mse_unsummed_2 = mean_square_error(filtered_y_data_fall_off, np.zeros(len(filtered_y_data_fall_off)))
    
    scaled_fit_value_1_mean  = (fit_value_1 - np.mean(mse_unsummed_1))/np.std(mse_unsummed_1)
    scaled_fit_value_2_mean  = (fit_value_2 - np.mean(mse_unsummed_2))/np.std(mse_unsummed_2)
   
    
    scaled_fit_value_1_minmax = (mse_unsummed_1-np.min(mse_unsummed_1)/(np.max(mse_unsummed_1)-np.min(mse_unsummed_1)))
    scaled_fit_value_2_minmax = (mse_unsummed_2 -np.min(mse_unsummed_2)/(np.max(mse_unsummed_2)-np.min(mse_unsummed_2)))
    
    scaled_fit_value_1 = np.mean(scaled_fit_value_1_mean)
    scaled_fit_value_2 = np.mean(scaled_fit_value_2_mean)
    if debug:
        print(f"The min max scaled fit value 1 array is {scaled_fit_value_1_minmax}, and the scaled fit value 2 array is {scaled_fit_value_2_minmax}")
        print(f"The mean scaled fit value 1 array is {scaled_fit_value_1_mean}, and the mean scaled fit value 2 array is {scaled_fit_value_2_mean}")
        print(f"The min max scaled fit value 1 is {np.mean(scaled_fit_value_1_minmax)}, and the scaled fit value 2 is {np.mean(scaled_fit_value_2_minmax)}")
        print(f"The mean scaled fit value 1 is {np.mean(scaled_fit_value_1_mean)}, and the scaled fit value 2 is {np.mean(scaled_fit_value_2_mean)}")    
        
    weighted_fit_value_1 = weight*scaled_fit_value_1
    weighted_fit_value_2 = (1-weight)*scaled_fit_value_2
    
    fit_value = weighted_fit_value_1 + weighted_fit_value_2
    return fit_value, weighted_fit_value_1, weighted_fit_value_2, fit_value_1, fit_value_2

def weighted_fit_functions(x_data, y_data, distal_edge, proximal_edge, weight) -> float:
    filtered_x_data_sobp, filtered_y_data_sobp = filter_data_by_range(x_data, y_data, proximal_edge, distal_edge) #Filters for the SOBP
    x_max = np.max(x_data)
    filtered_x_data_fall_off, filtered_y_data_fall_off = filter_data_by_range(x_data, y_data, distal_edge, x_max) #Filters for the fall off
    
    fit_value_1, abs_diff_1 = mean_square_error(filtered_y_data_sobp, np.ones(len(filtered_y_data_sobp)))
    #matplot_plotting(filtered_x_data_sobp, abs_diff_1, 'images/', 'SOBP abs diff', 'SOBP abs diff', save_fig=False)
    fit_value_2, abs_diff_2 = mean_square_error(filtered_y_data_fall_off, np.zeros(len(filtered_y_data_fall_off)))
    weighted_fit_value_1 = weight*fit_value_1
    weighted_fit_value_2 = (1-weight)*fit_value_2
    fit_value = weight*fit_value_1 + (1-weight)*fit_value_2
    return fit_value, weighted_fit_value_1, weighted_fit_value_2

def total_fit_functions(final_filename, weight = 0.5, data_analysis: bool = False, determined_proximal_edge: float = 10.9, determined_distal_edge: float = 14.9, debug: bool = False) -> float:
    # Importing the data from the file
    dose, x_data, y_data = import_data_from_file(final_filename)
    normalized_y_data = normalize_data(dose)

    if data_analysis:
        max_dose, min_dose, mean_dose, std_dose, std_error, x_location_max = simple_data_analysis(dose)
    
    # Defining proximal and distal edge from earlier experiments
    proximal_edge =  determined_proximal_edge
    distal_edge =  determined_distal_edge
    print(f"The proximal edge is {proximal_edge}, and the distal edge is {distal_edge}")
    
    # Calculating the fit value
    fit_value, weighted_fit_value_1, weighted_fit_value_2, fit_value_1, fit_value_2 = fit_functions(x_data, normalized_y_data, distal_edge, proximal_edge, weight, debug)
    print(f"The fit value, for the file: {final_filename}, is: {fit_value}, it consist of fit value 1: {fit_value_1} and fit value 2: {fit_value_2} total fit value is: \"{weight}*fitval1+ {1-weight}*fitval2\" ")
    
    return fit_value, fit_value_1, fit_value_2

def plot_ridgepin_shape_old(
        save_path: str, figure_title: str, figure_filename: str,
        solution_values: list[float] = [1,1.5,3],
        max_height: float = 30, # in mm
        y_label: str = "Width of the ridgepin (mm)", 
        x_label: str =  "Height of the ridgepin (mm)",
        save_fig: bool = True,
        show_fig: bool = False):
    #Makes the save folder if it does not exist.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the file already exists and modify the filename if necessary
    full_path = os.path.join(save_path, figure_filename)
    base, ext = os.path.splitext(full_path)
    counter = 1
    while os.path.exists(full_path):
        full_path = f"{base}_{counter}{ext}"
        counter += 1

    if '.png' not in full_path:
        full_path = full_path + '.png'

    # Calculating the data needed for the plot
    total_height_steps = max_height/len(solution_values)
    heights = [i*total_height_steps for i in range(len(solution_values))]
    negative_heights = [-1*i for i in heights]
    negative_solution_values = [-1*i for i in solution_values]
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(solution_values, heights, color='blue')
    plt.plot(negative_solution_values, heights, color='blue')
    plt.plot([np.max(negative_solution_values), np.min(solution_values)], [np.max(heights), np.max(heights)], color='blue')
    #plt.plot(solution_values[::-1], negative_heights, color='blue')
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(figure_title)
    if save_fig:
        plt.savefig(full_path, format='png', dpi = 300)
    if show_fig:
        plt.show()
    plt.close()

def plot_ridgepin_shape(
        save_path: str, figure_title: str, figure_filename: str,
        solution_values: list[float] = [1, 1.5, 3],
        max_height: float = 30,  # in mm
        x_label: str = "Width [mm]", 
        y_label: str = "Height [mm]",
        save_fig: bool = True,
        show_fig: bool = False,
        save_fig_eps: bool = False):	
    # Makes the save folder if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the file already exists and modify the filename if necessary
    full_path = os.path.join(save_path, figure_filename)
    base, ext = os.path.splitext(full_path)
    counter = 1
    while os.path.exists(full_path):
        full_path = f"{base}_{counter}{ext}"
        counter += 1

    if '.png' not in full_path:
        full_path = full_path + '.png'

    # Calculating the data needed for the plot
    total_height_steps = max_height / len(solution_values)
    heights = [i * total_height_steps for i in range(len(solution_values))]
    widths = solution_values
    negative_widths = [-1 * val for val in solution_values]

    # print(f"heights: {heights}, widths: {widths}, negative_widths: {negative_widths}, length of heights: {len(heights)}, length of widths: {len(widths)}, length of negative_widths: {len(negative_widths)}")
    plt.figure()
    ax = plt.subplot(111)

    # Plot the positive ridgepin shape
    for i in range(len(heights) - 1):
        # Horizontal line
        plt.plot([widths[i], widths[i + 1]], [heights[i], heights[i]], color='gray', linewidth=1.5)
        # Vertical line
        if i < len(heights) - 2:
            plt.plot([widths[i + 1], widths[i + 1]], [heights[i], heights[i + 1]], color='gray', linewidth=1.5)

    # Plot the negative ridgepin shape (mirrored)
    for i in range(len(heights) - 1):
        # Horizontal line
        plt.plot([negative_widths[i], negative_widths[i + 1]], [heights[i], heights[i]], color='gray', linewidth=1.5)
        # Vertical line
        if i < len(heights) - 2:
            plt.plot([negative_widths[i + 1], negative_widths[i + 1]], [heights[i], heights[i + 1]], color='gray', linewidth=1.5)

    # Connect the topmost points
    plt.plot([negative_widths[-1], widths[-1]], [heights[-1], heights[-1]], color='gray', linewidth=1.5)
    plt.plot([negative_widths[-1], negative_widths[-1]], [heights[-2], heights[-1]], color='gray', linewidth=1.5)  # Leftmost vertical
    plt.plot([widths[-1], widths[-1]], [heights[-2], heights[-1]], color='gray', linewidth=1.5)  # Rightmost vertical
    # Add labels and title
    plt.ylabel(y_label, fontdict={'fontsize': 16})
    plt.xlabel(x_label, fontdict={'fontsize': 16})
    plt.title(figure_title)

    # Save or show the figure
    if save_fig:
        image_format = 'png'
        image_name = full_path.replace('.png', f'.{image_format}')
        plt.savefig(image_name, format=image_format, dpi=1200)
    if save_fig_eps:
        image_format = 'eps'
        image_name = full_path.replace('.png', f'.{image_format}')
        plt.savefig(image_name, format=image_format, dpi=1200)
    if show_fig:
        plt.show()
    plt.close()
def calculate_average_fitness(file_1: str, file_2: str = None):
    """
    Calculate the average fitness for each generation from the input dictionary.

    Args:
        data (dict): A dictionary where values are dictionaries containing
                     generation and fitness data.

    Returns:
        dict: A dictionary with generation numbers as keys and their 
              average fitness values as values.
    """
    data = json.load(open(file_1))

    # Dictionary to store fitness values grouped by generation
    fitness_by_generation = defaultdict(list)

    # Iterate over the dictionary
    for value in data.values():
        generation_number = value["generation"]
        fitness_by_generation[generation_number].append(value["fitness"])

    # Calculate average fitness for each generation
    average_fitness_by_generation = {
        generation: sum(fitnesses) / len(fitnesses)
        for generation, fitnesses in fitness_by_generation.items()
    }

    return average_fitness_by_generation

def data_vis_results(filename: str) -> list[float]:
    # Import the data from the file
    data = json.load(open(filename, 'r'))
    
    # DATA PROCESSING
    total_data = []
    height_data = []
    width_data = []
    step_heigt = 1 #mm
    best_sol = []
    best_fit_val = 1
    best_location = ""
    time_fitness_data = []
    start_time = (data[list(data.keys())[0]]['timestamp'])
    date_time_start_time = dt.datetime.strptime(start_time, '%H:%M:%S')
    for key, value in data.items():
        #print(f"The key is: {key}, and the value is: {value}")
        for i, val in enumerate(value['solution values']):
            # Some weirdness in the data, so we need to correct it
            if i == 0:

                if val < 2.3:
                    val = 1.05*value['solution values'][i+1]
            if i == 29:
                if val > 0.5:
                    val = value['solution values'][i-1]
                #print((round(val,4), i*step_heigt))
            total_data.append((round(val,4), i*step_heigt))
            height_data.append(i*step_heigt)
            width_data.append(val)
        date_time_time = dt.datetime.strptime(value['timestamp'], '%H:%M:%S')
        time_difference = date_time_time - date_time_start_time
        if time_difference.days < 0:
            time_difference = dt.timedelta(days=0, seconds=time_difference.seconds, microseconds=time_difference.microseconds)
        base_date = dt.datetime(1900, 1, 1)  # Arbitrary date
        new_datetime = base_date + time_difference
        time_fitness_data.append((value['timestamp'], value['fitness']))
        
        if value["fitness"] < best_fit_val:
            best_gen = value["generation"]
            best_fit_val = value["fitness"]
            best_sol = value['solution values']
            best_location = value['data_file']
    print(f"The best solution is: {best_sol}, with a fit value of: {best_fit_val}, and the location is: {best_location}, it was set in generation {best_gen}")
    
    
    
    #plotting convergence over time
    plt.figure()
    time, fitness = zip(*time_fitness_data)
    plt.scatter(time, fitness, linestyle=":", marker="x", label="Fitness value")
    plt.ylabel("Fitness value", fontdict={'fontsize': 14})
    plt.xlabel("Time", fontdict={'fontsize': 14})
    plt.savefig("/home/dabouwer/tests/codetorun/images/convergence_plot_time_bone.png", format='png')
    plt.close()
    #plotting fitness value convergence over generations
    fitness_per_generation_mse = (calculate_average_fitness(filename))
    plt.figure()
    #plt.plot(list(max_error_first_10.keys()), list(max_error_first_10.values()), linestyle=":", marker="x", label="Root maximum absolute error", )
    plt.plot(list(fitness_per_generation_mse.keys()), list(fitness_per_generation_mse.values()), linestyle=":", marker="x",  label="Average fitness value")
    plt.scatter(best_gen, best_fit_val, marker="*", color="red", label=f"Best fitness value: {round(best_fit_val,5)}")
    plt.legend()
    plt.xlabel("Generation number", fontdict={'fontsize': 14})
    plt.ylabel("Average fitness value", fontdict={'fontsize': 14})
    plt.savefig("/home/dabouwer/tests/codetorun/images/convergence_plot_mse_best_bone.eps", format='eps')
    plt.savefig("/home/dabouwer/tests/codetorun/images/convergence_plot_mse_best_bone.png", format='png')
    plt.close()
    
    #Plotting IDD of the SOBP
    dose, x_data, y_data = import_data_from_file(best_location)
    y_data_sum = y_data[0] + y_data[1]
    normalized_y_data = y_data_sum / np.max(y_data_sum)
    #normalized_y_data = normalize_data(dose)
    filtered_x, filtered_y_data = filter_data_by_range(x_data, normalized_y_data, 10.9, 14.9)
    print(filtered_y_data)
    max_val = np.max(filtered_y_data)
    min_val = np.min(filtered_y_data)
    mean_val = np.mean(filtered_y_data)
    std_val = np.std(filtered_y_data)
    std_error = std_val/np.sqrt(len(filtered_y_data))
    dose_difference = max_val - min_val
    print(f"The max dose is: {round(max_val*100,6)} \%, the min dose is: {round(min_val*100,6)} \%, the mean dose is: {round(mean_val*100,6)}\%, the std dose is: {std_val}, the std error is: {std_error}, the dose difference is: {round(dose_difference*100,6)}\%")
    x_label = "Depth [cm]"
    y_label = "Normalized Dose"
    x_max = 18
    x_min = 7
    y_max = 1.1
    plt.figure()
    
    distal_edge = 10.9
    proximal_edge = 14.9
    plt.axvline(x=distal_edge, color='r', linestyle='--')
    plt.axvline(x=proximal_edge, color='r', linestyle='--')
    ax = plt.subplot(111)
    plt.plot(x_data, normalized_y_data)
    plt.xlabel(x_label, fontdict={'fontsize': 14})
    plt.ylabel(y_label, fontdict={'fontsize': 14})
    ax.set_xlim(xmin= x_min, xmax=x_max)
    ax.set_ylim(ymax=y_max)
    plt.savefig('images/IDD_SOBP_Best_bone.png', format='png', dpi = 1200)
    plt.savefig('images/IDD_SOBP_Best_bone.eps', format='eps', dpi = 1200)
    plt.close()

    #Making heat map
    #print(total_data)
    flattened_data = np.array(total_data).flatten()
    data_tuples = [tuple(pair) for pair in total_data]    # Convert to tuples for counting

    
    # Count the frequency of each (x, y) pair
    freq = Counter(data_tuples)

    # Extract x, y coordinates and their corresponding frequencies
    x_coords, y_coords = zip(*freq.keys())
    frequencies = list(freq.values())

    # Sort the data by frequency (ascending order so most common on top)
    sorted_data = sorted(zip(x_coords, y_coords, frequencies), key=lambda x: x[2])

    # Unzip sorted data
    x_coords, y_coords, frequencies = zip(*sorted_data)

    # Normalize frequencies to [0, 1] for colormap
    norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))
    # Generate a colormap, CIVIDIS OF PLASMA
    cmap = cm.cividis

    # Assign colors based on normalized frequencies
    colors = [cmap(norm(freq[(x, y)])) for x, y in zip(x_coords, y_coords)]

    # Scatter plot using x, y coordinates
    scatter = plt.scatter(x_coords, y_coords, c=frequencies, cmap=cmap)
    scatter = plt.scatter(-1*np.array(x_coords), y_coords, c=frequencies, cmap=cmap)

    # Add a colorbar, linking it directly to the scatter plot
    plt.colorbar(scatter, label='Frequency')

    # Plot formatting
    plt.xlabel("Width [mm]", fontdict={'fontsize': 14})
    plt.ylabel("Height [mm]", fontdict={'fontsize': 14})
    #plotting the ridgepin shape
    # Calculating the data needed for the plot
    max_height = 30  # in mm
    solution_values = best_sol
    total_height_steps = max_height / len(solution_values)
    heights = [i * total_height_steps for i in range(len(solution_values))] 
    widths = solution_values
    negative_widths = [-1 * val for val in solution_values]
   
    # Plot the positive ridgepin shape
    for i in range(len(heights) - 1):
        # Horizontal line
        plt.plot([widths[i], widths[i + 1]], [heights[i], heights[i]], color='red', linewidth=1.5)
        # Vertical line
        if i < len(heights) - 2:
            plt.plot([widths[i + 1], widths[i + 1]], [heights[i], heights[i + 1]], color='red', linewidth=1.5)

    # Plot the negative ridgepin shape (mirrored)
    for i in range(len(heights) - 1):
        # Horizontal line
        plt.plot([negative_widths[i], negative_widths[i + 1]], [heights[i], heights[i]], color='red', linewidth=1.5)
        # Vertical line
        if i < len(heights) - 2:
            plt.plot([negative_widths[i + 1], negative_widths[i + 1]], [heights[i], heights[i + 1]], color='red', linewidth=1.5)

    # Connect the topmost points
    plt.plot([negative_widths[-1], widths[-1]], [heights[-1], heights[-1]], color='red', linewidth=1.5)
    plt.plot([negative_widths[-1], negative_widths[-1]], [heights[-2], heights[-1]], color='red', linewidth=1.5)  # Leftmost vertical
    plt.plot([widths[-1], widths[-1]], [heights[-2], heights[-1]], color='red', linewidth=1.5)  # Rightmost vertical    
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color='red', lw=1.5, label='Optimized ridge pin shape'),
        ],
        loc='lower center',  # Place legend below the bbox anchor point
        bbox_to_anchor=(0.5, 1.02),  # Center the legend above the plot
    )
    plt.savefig('images/unique_points_ridgepin_cividis_red_best_bone.png', format='png', dpi=1200)
    plt.savefig('images/unique_points_ridgepin_cividis_red_best_bone.eps', format='eps', dpi=1200)
    plt.close()
    return 

def data_vis_method():
    # Import the data from the file
    csv_file = '/home/dabouwer/tests/codetorun/tests/DataRuns/ProximalEdge+VaryingWidthsRF/DoseAtWaterbox_proximaledge_w40mm.csv'
    filename = csv_file.split('/')[-1].split('.')[0]
    print(f"The filename is: {filename}")
    directory = 'images/method/'
    final_filepath = directory + filename
    #final_filepath += '_NotNormalized' # Remove when normalizing data
    #Plotting IDD of the SOBP
    dose, x_data, y_data = import_data_from_file(csv_file)
    y_data_not_normalized = np.squeeze(dose.data['Sum'])
    normalized_y_data = normalize_data(dose)
    x_label = "Depth [cm]"
    y_label = "Normalized Dose"
    x_max = 20
    x_min = 6
    y_max = 1.1
    plt.figure()
    
    proximal_edge = 10.9 #10.9 for own sim, 13.1 from benchmark
    distal_edge = 14.9 #14.9 for own sim, 15.1 from benchmark
    #Only include when generating SOBP
    #plt.axvline(x=distal_edge, color='r', linestyle='--', label = f"x = {distal_edge} cm")
    #plt.axvline(x=proximal_edge, color='r', linestyle='--', label= f"x = {proximal_edge} cm")
    

    ax = plt.subplot(111)
    plt.plot(x_data, y_data)
    plt.xlabel(x_label, fontdict={'fontsize': 14})
    plt.ylabel(y_label, fontdict={'fontsize': 14})
    ax.set_xlim(xmin= x_min, xmax=x_max)
    #ax.set_ylim(ymax=y_max) #Disable when doing not normalized runs
    #plt.legend(loc="upper left")
    plt.savefig(final_filepath + '.png', format='png', dpi = 1200)
    plt.savefig(final_filepath + '.eps', format='eps', dpi = 1200)

    return

def main():
    file_path_1 = '/home/dabouwer/tests/codetorun/tests/DataRuns/Unknown/DoseAtWaterbox.csv'
    file_path_2 = '/home/dabouwer/tests/codetorun/tests/DataRuns/Previous study/DoseAtWaterbox_benchmark.csv'
    dose_1, x_data_1, y_data_1 = import_data_from_file(file_path_1)
    dose_2, x_data_2, y_data_2 = import_data_from_file(file_path_2)
    print(f"For file 1, with filepath {file_path_1}:")
    max_1, min_1, mean_1, std_1, std_error_1, x_location_max_1 = simple_data_analysis(dose_1)
    print(f"For file 2, with filepath {file_path_2}:")
    max_2, min_2, mean_2, std_2, std_error_2, x_location_max_2 = simple_data_analysis(dose_2)
    #From the files DoseAtWaterbox_proximaledge_w40mm.csv and DoseAtWaterbox_distaledge_w33mm.csv
    #The maximum dose for distal edge is at 14.9 cm, and the maximum dose for proximal edge is at 10.9 cm
    proximal_edge = 13.1
    distal_edge = 15.1
    print(f"the distal edge is {distal_edge}")
    print(f"the proximal edge is {proximal_edge}")
    normalized_y_data_1 = normalize_data(dose_1)
    normalized_y_data_2 = normalize_data(dose_2)

    fit_value_1 = fit_function_new(file_name=file_path_1,debug=True, proximal_edge = proximal_edge, distal_edge = distal_edge,max_error=False, sqrt_fit_val=True)
    fit_value_2 = fit_function_new(file_name=file_path_2,debug=False, max_error=False, sqrt_fit_val=True)
    print(f"The fit value for the first file is {round(fit_value_1,6)}")
    print(f"The fit value for the second file is {fit_value_2}")
    filtered_x_data_1, filtered_y_data_1 = filter_data_by_range(x_data_1, normalized_y_data_1, x_min=10, x_max=18)
    filtered_x_data_2, filtered_y_data_2 = filter_data_by_range(x_data_2, normalized_y_data_2, x_min=distal_edge, x_max=18)

    file_title_1 = file_path_1.split('\\')[-1].split('.')[0]
    figure_title_1 = figure_title_maker(file_path_1)
    
    file_title_2 = file_path_2.split('\\')[-1].split('.')[0]
    figure_title_2= figure_title_maker(file_path_2)
    plt.figure()
    plt.plot(filtered_x_data_1, filtered_y_data_1)
    # plt.plot(filtered_x_data_1, (np.ones(len(filtered_y_data_1)) - filtered_y_data_1), color = 'blue')
    # plt.plot(filtered_x_data_2, filtered_y_data_2, color = 'blue')
    plt.axvline(x=distal_edge, color='r', linestyle='--')
    plt.axvline(x=proximal_edge, color='r', linestyle='--')
    plt.xlabel("Depth [cm]")
    plt.ylabel("Normalized Dose")
    filepath = "/home/dabouwer/tests/codetorun/images/"
    filename = "UnknownRun"
    format = 'png'
    
    plt.savefig(filepath + filename + '.' + format, format=format, dpi = 300)
    plt.show()
    filtered_x_data = (filtered_x_data_1 + filtered_x_data_2)
    filtered_y_data = ((np.ones(len(filtered_y_data_1)) - filtered_y_data_1) + (filtered_y_data_1))
    #matplot_plotting(filtered_x_data_2, filtered_y_data_2, 'images/', figure_title="Difference", distal_edge=distal_edge, proximal_edge=proximal_edge,figure_filename=file_title_1, save_fig=True, show_fig=True)
    #Plotting data for the first file
    #matplot_plotting(x_data_1, normalized_y_data_1, 'images/', figure_title_1, distal_edge=distal_edge, proximal_edge=proximal_edge,figure_filename=file_title_1, save_fig=True, show_fig=True)
    #matplot_plotting(filtered_x_data_1, filtered_y_data_1, 'images/', figure_title_1,distal_edge=distal_edge, proximal_edge=proximal_edge, figure_filename=file_title_1, save_fig=False)

    #Plotting data for the second file
    #matplot_plotting(x_data_2, normalized_y_data_2, 'images/', figure_title_2, distal_edge=distal_edge, proximal_edge=proximal_edge, figure_filename=file_title_2, save_fig=False)
    #matplot_plotting(filtered_x_data_2, filtered_y_data_2, 'images/', figure_title_2, distal_edge=distal_edge, proximal_edge=proximal_edge,figure_filename=file_title_2, save_fig=True)
   
def main_test():
    max_error_bool = True
    debug_bool = True
    fit_value_goed = fit_function_new(r"/home/dabouwer/tests/codetorun/DoseAtWaterbox_goed.csv", proximal_edge=13, distal_edge=15, max_error=max_error_bool, debug=debug_bool)
    fit_value_test_1 = fit_function_new(r"/home/dabouwer/tests/codetorun/runs/19-11-2024_10:00/generation_67_attempt_11_at_10:43:33/DoseAtWaterbox_generation_67_11.csv", max_error=max_error_bool, debug=debug_bool)
    fit_value_test_2 = fit_function_new(r"/home/dabouwer/tests/codetorun/runs/19-11-2024_10:00/generation_2_attempt_27_at_11:53:21/DoseAtWaterbox_generation_2_27.csv", max_error=max_error_bool, debug=debug_bool)
    print(f"The benchmark fit waarde is: '{fit_value_goed}', een test case 1 is: '{fit_value_test_1}', test case 2 is: '{fit_value_test_2}' .")
    
def bone():
    file = '/home/dabouwer/tests/codetorun/DoseAtWaterbox.csv'
    dose, x_data, y_data = import_data_from_file(file)
    print(x_data, y_data)
    y_data_sum = y_data[0] + y_data[1]
    print(type(y_data[1]), y_data_sum)
    
    # for i in range(len(y_data[1])):
    #     print(y_data[0])
    #     y_data = y_data[0][i] + y_data[1][i]
    #     y_data_sum.append(y_data)
    y_data_norm = y_data_sum / np.max(y_data_sum)
    #y_data_norm = normalize_data(y_data_sum)
    
    plt.figure()
    plt.plot(x_data, y_data_norm)
    plt.savefig('images/lead.png', format='png', dpi = 1200)
    dose_info(dose)

if __name__ == '__main__':
    #data_vis_method()
    #data_vis_results('/home/dabouwer/tests/codetorun/runs/12-12-2024_12:00/tried_solutions.json')
    #data_vis_results('/home/dabouwer/tests/codetorun/runs/14-12-2024_17:00/tried_solutions.json')#LE BONe
    plot_ridgepin_shape('/home/dabouwer/tests/codetorun/images/', '', 'ridgepin_shape_schematic', solution_values=[3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], max_height=30, save_fig=True, show_fig=False, save_fig_eps=True)
    #bone()
    #Plotting IDD of the SOBP
    # dose, x_data, y_data = import_data_from_file('/home/dabouwer/tests/codetorun/DoseAtWaterbox.csv')
    # normalized_y_data = normalize_data(dose)
    # filtered_x, filtered_y_data = filter_data_by_range(x_data, normalized_y_data, 13.1, 15.1)
    # print(filtered_y_data)
    # max_val = np.max(filtered_y_data)
    # min_val = np.min(filtered_y_data)
    # mean_val = np.mean(filtered_y_data)
    # std_val = np.std(filtered_y_data)
    # std_error = std_val/np.sqrt(len(filtered_y_data))
    # dose_difference = max_val - min_val
    # print(f"The max dose is: {max_val}, the min dose is: {min_val}, the mean dose is: {mean_val}, the std dose is: {std_val}, the std error is: {std_error}, the dose difference is: {dose_difference}")
