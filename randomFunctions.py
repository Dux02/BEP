import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import islice

# Function to check if the file is still being written to
def is_file_still_writing(file_path, wait_time=5):
    if not os.path.exists(file_path):
        return True  # File does not exist yet
    
    # Get the file size initially
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)
    
    # Get the file size after waiting
    new_size = os.path.getsize(file_path)
    
    # If the file size hasn't changed, the writing is complete
    return initial_size != new_size

def generate_new_filename(filename):
    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)
    
    # Initialize the count
    count = 1
    
    # While the filename exists, increment the count and modify the filename
   
    while os.path.exists(filename):
        filename = f"{name}_{count}{extension}"
        count += 1
    
    return filename

def rename_file(old_filename, new_filename):
    # Wait until the file stops being written to
    while is_file_still_writing(old_filename):
        print(f"Waiting for '{old_filename}' to finish writing...")
        time.sleep(5)  # Wait for 5 seconds before checking again
    final_filename = generate_new_filename(new_filename)
    # Rename the file once writing is complete
    try:
        os.rename(old_filename, final_filename)
        print(f"File has been renamed to {final_filename}")
    except FileNotFoundError:
        print(f"The file '{old_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return final_filename

def save_tried_solutions(solutions, fitness_value: float, filename: str, file_name_data: str, fitness_value_1: float = 0, fitness_value_2: float = 0, save_fitness_values_per_generation: bool = False, generation_number: int = 0, solution_number: int = 0):
    current_time = time.localtime()
    timestamp = f"{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}"
    #Making the file name for the data in json
    if save_fitness_values_per_generation:
        filename_generations = filename.replace('.txt', '.json')
        if os.path.exists(filename_generations):
            with open(filename_generations, 'r') as file:
                data = json.load(file)
        else:
            data = {}
        
        data_key = f"Generation_{generation_number}_solution_{solution_number}"
        data[data_key] = {
            "generation": generation_number,
            "solution": solution_number,
            "solution values": solutions, 
            "fitness": fitness_value, 
            "fitness_1": fitness_value_1, 
            "fitness_2": fitness_value_2, 
            "timestamp": timestamp,
            "data_file": file_name_data
            }
        # Write the updated data back to the JSON file
        with open(filename_generations, 'w') as file:
            json.dump(data, file, indent=4)
        
    
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write("")

    with open(filename, 'a') as file:
        if fitness_value_1 == 0 and fitness_value_2 == 0:
            file.write(f"{timestamp} - Solution: {solutions}, Fitness: {fitness_value}, Data can be found at: {file_name_data}\n")
        else:
            file.write(f"{timestamp} - Solution: {solutions}, Fitness: {fitness_value}, fit1: {fitness_value_1}, fit2: {fitness_value_2}, Data can be found at: {file_name_data}\n")

def create_run_directory(path: str = "", time_bool: bool = True):
    # Get the current date and hour
    current_time = time.localtime()
    if time_bool:
        directory_name = f"{current_time.tm_mday}-{current_time.tm_mon}-{current_time.tm_year}_{current_time.tm_hour}:00"
        final_directory = os.path.join(path, directory_name)
    else:
        final_directory = os.path.join(path)
    # Check if the directory already exists
    if not os.path.exists(final_directory):
        # Create the directory
        os.makedirs(final_directory)
        print(f"Directory {final_directory} created.")
    else:
        print(f"Directory {final_directory} already exists.")
    return final_directory

def move_files_to_directory(directory_name, files):
    for file in files:
        try:
            new_filename = os.path.join(directory_name, os.path.basename(file))
            os.rename(file, new_filename)
            print(f"File {file} moved to {directory_name}.")
        except FileNotFoundError:
            print(f"The file '{file}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    return new_filename

def move_files_to_directory2(
    save_path: str, 
    csv_filename: str):

    #Makes the save folder if it does not exist.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the file already exists and modify the filename if necessary
    full_path = os.path.join(save_path, csv_filename)
    base, ext = os.path.splitext(full_path)
    counter = 1
    while os.path.exists(full_path):
        full_path = f"{base}_{counter}{ext}"
        counter += 1

    if '.csv' not in full_path:
        full_path = full_path + '.csv'


    try:
        new_filename = os.path.join(directory_name, os.path.basename(file))
        os.rename(file, new_filename)
        print(f"File {file} moved to {directory_name}.")
    except FileNotFoundError:
        print(f"The file '{file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return new_filename

def initial_population(num_solutions, num_genes):
    initial_population = []
    for i in range(num_solutions):
        solution = np.random.uniform(low=0, high=3, size=num_genes)
        solution = np.sort(solution)[::-1]
        initial_population.append(solution)
        print(solution)
    print(initial_population)
    return initial_population

def convergence_plotting(file_1: str, file_2: str = None):
    data_file_1 = json.load(open(file_1))
    fitness_generation = {}
    for key in data_file_1.keys():
        datapoint = data_file_1[key]
        generation = datapoint['generation']
        solution = datapoint['solution']
        fitness = datapoint['fitness']
        fitness_generation[generation] = fitness
    
    for key in fitness_generation.keys():
        print(f"Generation: {key}, Fitness: {fitness_generation[key]}")
    print(fitness_generation)
    if file_2:
        data_file_2 = json.load(open(file_2))
        for key in data_file_2.keys():
            datapoint = data_file_2[key]
            generation = datapoint['generation']
            solution = datapoint['solution']
            fitness = datapoint['fitness']
            fitness_generation[generation] = fitness
        
    
    return None
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

def scale_dictionaries(dict1, dict2):
    """
    Scale two dictionaries to have their values in the same range [0, 1].
    
    Parameters:
        dict1 (dict): The first dictionary to scale.
        dict2 (dict): The second dictionary to scale.
        
    Returns:
        tuple: Scaled versions of dict1 and dict2.
    """
    def scale_values(d):
        min_val = 0 #min(d.values())
        max_val = max(d.values())
        return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}
    
    scaled_dict1 = scale_values(dict1)
    scaled_dict2 = scale_values(dict2)
    
    return scaled_dict1, scaled_dict2
# Function to get the first n key-item pairs
def get_first_n_key_item_pairs(d, n=10):
    return dict(islice(d.items(), n))

if __name__ == '__main__':
    # file_1_max = r"/home/dabouwer/tests/codetorun/runs/4-12-2024_12:00_MAX/tried_solutions.json"
    # file_2_mse = r"/home/dabouwer/tests/codetorun/runs/12-12-2024_12:00/tried_solutions.json"
    # #convergence_plotting(file_1, file_2)
    # fitness_per_generation_max = (calculate_average_fitness(file_1_max))
    # fitness_per_generation_mse = (calculate_average_fitness(file_2_mse))
    # print(fitness_per_generation_max, fitness_per_generation_mse)
    # print(np.sqrt(30))
    # # for value, key in fitness_per_generation_max.items():
    # #     print(f"Generation: {value}, Fitness max: {key}, fitness mse: {fitness_per_generation_mse[value]}, div {key/fitness_per_generation_mse[value]}")
    # #     print(f"Fitness_max {key}, Fitness_mse {fitness_per_generation_mse[value]} , Fitness_max*sqrt(30) {key*np.sqrt(30)}")
    # #     if key <= fitness_per_generation_mse[value] and fitness_per_generation_mse[value] <= key*np.sqrt(30):
    # #         print("True")
    
    # max_error_first_10 = get_first_n_key_item_pairs(fitness_per_generation_max, 10)
    # mse_first_10 = get_first_n_key_item_pairs(fitness_per_generation_mse, 10)
    
    # plt.figure()
    # #plt.plot(list(max_error_first_10.keys()), list(max_error_first_10.values()), linestyle=":", marker="x", label="Root maximum absolute error", )
    # plt.plot(list(fitness_per_generation_mse.keys()), list(fitness_per_generation_mse.values()), linestyle=":", marker="x",  label="Root mean squared error")
    # plt.xlabel("Generation number")
    # plt.ylabel("Average fitness value")
    # plt.legend()
    # plt.show()
    # plt.savefig("/home/dabouwer/tests/codetorun/images/convergence_plot_mse_final.png", format='png')
    tekst = ""
    for i in range(200):
        #tekst += '\"G4_WATER\" '
        if i == 50 or i == 51 or i == 150 or i == 151:
             tekst += '\"Bone\" '
        else:
            tekst += '\"G4_WATER\" '
    
    print(tekst)
    rp = []
    for i in range(30):
        rp.append(round((30-i)*0.1,3))
    print(rp)