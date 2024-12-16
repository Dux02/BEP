import pygad
import numpy as np
import os
import datetime as dt
import time

from randomFunctions import rename_file, save_tried_solutions, move_files_to_directory, create_run_directory
from createRidgeFilter import create_ridgefilter
from dataAnalysis import plot_ridgepin_shape, fit_function_new, import_data_from_file, normalize_data, matplot_plotting, fit_function_new_bone

# Example: Assuming your vector has 5 coordinates
dimension = 30 # Amount of points in the ridgepin
lowerbound = 0 # Lowerbound of the ridgepin in mm ( thinnest part)
upperbound = 3 # Upperbound of the ridgepin in mm ( thickest part)
fitvalue_weight = 1 # Weight of the fitness value 1 means only first fitness function 0 means
num_of_generations =  100# How many generations the genetic algorithm will run
num_of_initial_population = 80 # Number of initial solutions
num_of_parents_mating = 10 # Number of parents that will mate and continue to the next generation
run_file_name = create_run_directory(path="runs")


def initial_population(num_solutions, 
                       num_genes):
    # Generate initial population
    initial_population = []
    for i in range(num_solutions):
        solution = np.random.uniform(low=0, high=3, size=num_genes)
        solution = np.sort(solution)[::-1]
        initial_population.append(solution)
    # print(f" The initial population is the following array: {initial_population}")
    return initial_population

def fitness_func(ga_instance, 
                 solution, 
                 solution_idx):
    print(f"Running simulation for solution {solution_idx}: with ridgepin coordinate: {solution}")
    current_time = time.localtime()
    generation_number = ga_instance.generations_completed

    # Defining directory names and file names
    base_directory_name = run_file_name
    total_directory_name = os.path.join(base_directory_name, f"generation_{generation_number}_attempt_{solution_idx}_at_{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}")
    created_directory = create_run_directory(total_directory_name, time_bool=False)
    new_filename = f"DoseAtWaterbox_generation_{generation_number}_{solution_idx}.csv"

    # Run the Monte Carlo simulation and returns the location of the data output
    final_filename = run_monte_carlo_simulation(coordinate_vector=solution, 
                                                old_filename="DoseAtWaterbox.csv", 
                                                new_filename=new_filename, 
                                                directory_name=created_directory)
    
    # Calculate the fitness value

    fitness_value = fit_function_new_bone(file_name=final_filename,debug=False, max_error=False, sqrt_fit_val=True)
    
    filename_save_tried_solutions = os.path.join(base_directory_name, 'tried_solutions.txt')
    
    save_tried_solutions(solutions=list(solution), 
                         fitness_value=fitness_value,  
                         filename=filename_save_tried_solutions, 
                         file_name_data=final_filename,
                         generation_number=generation_number,
                         solution_number=solution_idx,
                         save_fitness_values_per_generation=True)
    
    print(f"Simulation complete for solution: '{solution_idx}'.")
    print(f"The final file name for this generation is: '{final_filename}'")
    
    # Plotting the ridgepin shape in the folder
    ridgepin_image_title = f"Ridgepin_shape_generation_{solution_idx}"
    ridgepin_output_file_name = f"{ridgepin_image_title}.png"
    
    plot_ridgepin_shape(solution_values=solution, 
                        save_path=created_directory, 
                        figure_filename=ridgepin_output_file_name, 
                        figure_title=ridgepin_image_title, 
                        save_fig=True, 
                        show_fig=False)

    # Plot the normalized SOBP at waterbox in the directory
    normalized_image_title = f"Normalized Dose, Generation: '{generation_number}', Sol. #: '{solution_idx}', fitness: {round(fitness_value,6)}"
    output_file_norm_name =  f"Normalized_dose_at_waterbox_generation_{generation_number}_sol_{solution_idx}.png"
    
    dose, x_data, y_data = import_data_from_file(final_filename)
    y_data_sum = y_data[0] + y_data[1]

    normalized_y_data = y_data_sum / np.max(y_data_sum)
    
    matplot_plotting(x_data, 
                     normalized_y_data, 
                     save_path=created_directory, 
                     figure_filename=output_file_norm_name, 
                     figure_title=normalized_image_title, 
                     save_fig=True, 
                     show_fig=False)
    
    return -fitness_value

# Function to run your Monte Carlo simulation, returning a result.
def run_monte_carlo_simulation(coordinate_vector: list[float ],  
                               new_filename: str, 
                               old_filename: str = "DoseAtWaterbox.csv",
                               directory_name: str =""):

    #Create the ridge filter based on the coordinate vector
    create_ridgefilter(coordinate_vector)

    # Simulate the geometry based on the coordinate vector
    os.system('/home/dabouwer/topas/bin/topas config_obstruction.txt') 
    
    # Try renaming the file, so it doesn't get overwritten
    final_filename = rename_file(old_filename, new_filename)
    new_final_filename = move_files_to_directory(directory_name, [final_filename])
    return new_final_filename  # Returns the file location where data is stored

def custom_crossover(parents, offspring_size, ga_instance):
    print(f"the parents are: {parents}")
    print(offspring_size, offspring_size[0])
    offspring = []

    # Generate offspring from parent pairs
    for k in range(offspring_size[0]):
        # Randomly select two parents
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        
        # Perform crossover (blend two parents)
        crossover_point = np.random.randint(1, parents.shape[1] - 1)
        offspring_k = np.concatenate((parents[parent1_idx, :crossover_point], 
                                      parents[parent2_idx, crossover_point:]))

        # Ensure offspring genes are sorted in increasing order
        offspring_k = np.sort(offspring_k)[::-1]
        
        offspring.append(offspring_k)
    # print(f"The new ofspring is {offspring}") Debug statement
    return np.array(offspring)

# Custom mutation function to ensure increasing order after mutation
def custom_mutation(offspring, ga_instance):
    # print(f"The not mutated offspring is {offspring}") Debug statement
    for idx in range(offspring.shape[0]):
        mutation_gene_idx = np.random.randint(0, offspring.shape[1])
        # Mutate the gene by selecting a random value from the gene space
        random_value = np.random.uniform(low=lowerbound, 
                                         high=upperbound)
        # Ensure the mutation keeps the genes in increasing order
        if mutation_gene_idx == 0:
            # For the first gene, ensure it's less than or equal to the second gene
            random_value = min(random_value, offspring[idx, mutation_gene_idx + 1])
        elif mutation_gene_idx == offspring.shape[1] - 1:
            # For the last gene, ensure it's greater than or equal to the previous gene
            random_value = max(random_value, offspring[idx, mutation_gene_idx - 1])
        else:
            # For middle genes, ensure it's between the previous and next gene
            random_value = max(min(random_value, offspring[idx, mutation_gene_idx + 1]), 
                               offspring[idx, mutation_gene_idx - 1])
        # Apply the mutation
        offspring[idx, mutation_gene_idx] = random_value
    # print(f"The mutated offspring is {offspring}") Debug statement
    return offspring

def main():
    # Define genetic algorithm parameters.
    ga_instance = pygad.GA(
        num_generations=num_of_generations,            # Number of generations
        initial_population=initial_population(num_of_initial_population, dimension), # Initial population
        num_parents_mating=num_of_parents_mating,          # Number of parents for mating
        fitness_func=fitness_func,      # Fitness function
        sol_per_pop=20,                # Population size
        num_genes=dimension,           # Number of coordinates in your vector
        mutation_percent_genes=10,     # Percentage of genes to mutate
        parent_selection_type="sss",    # Type of parent selection
        keep_elitism=10,                 # Number of parents to keep in the next generation
        mutation_type=custom_mutation,        # Custom mutation method
        crossover_type=custom_crossover,   # Custom crossover function
        gene_type=float,               # Type of input parameters
        gene_space=range(lowerbound,upperbound),         # Gene space
        stop_criteria="saturate_7", # Stop criteria if the fitness value doesn't change for 7 generations
    )
    # Create a directory for the run
    #run the genetic algorithm
    ga_instance.run()
    return ga_instance

if __name__ == '__main__':
    ga_instance = main()

    # After the run, you can inspect the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    filename_save_tried_solutions = os.path.join(run_file_name, 'tried_solutions.txt')
    
    print(f"Best solution: {solution}")
    print(f"Best fitness: {solution_fitness}")
    # save_tried_solutions(solutions=solution, 
    #                      fitness_value=solution_fitness,  
    #                      filename=filename_save_tried_solutions, 
    #                      file_name_data="N/A",
    #                      generation_number=ga_instance.generations_completed+1,
    #                      solution_number=solution_idx,
    #                      save_fitness_values_per_generation=True)