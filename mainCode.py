from createRidgeFilter import place_ridgepins_xy, is_monotone_decreasing
from dataAnalysis import fit_functions, import_data_from_file, normalize_data, simple_data_analysis
from randomFunctions import rename_file
import os

i = j = 1
old_filename = "DoseAtWaterbox.csv"
new_filename = f"DoseAtWaterbox_generation_{i}_population_{j}.csv"

def main():
    #Create ridge filter and checks if the inital ridge filter is in decreasing order
    ridge_pin_coordinates = [1.5,1.0,0.5,0.1]
    if not is_monotone_decreasing(ridge_pin_coordinates):
        raise ValueError('The ridge pin coordinates are not in decreasing order')
    file_name = 'ridgepins_main.txt'
    place_ridgepins_xy(hlx_values=ridge_pin_coordinates, file_name=file_name)

    #Run the monte carlo simulation using a bash file using topas and the file generated from the ridge filter
    os.system('/home/dabouwer/topas/bin/topas config.txt')
    #Try renaming the file, so it doesn't get overwritten
    final_filename = rename_file(old_filename, new_filename)

    dose, x_data, y_data = import_data_from_file(final_filename)
    print(f"For file 1, with filepath {final_filename}:")
    max_dose, min_dose, mean_dose, std_dose, std_error, x_location_max = simple_data_analysis(dose)
    proximal_edge = 0.975*10.9
    distal_edge = 1.025*14.9
    print(f"the distal edge is {distal_edge}")
    print(f"the proximal edge is {proximal_edge}")
    normalized_y_data = normalize_data(dose)

    fit_value = fit_functions(x_data, normalized_y_data, distal_edge, proximal_edge)
    print(f"The fit value for the file based up the following ridge pin coordinates: {ridge_pin_coordinates} is {fit_value}")

if __name__ == '__main__':
    main()