import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from statistics import NormalDist
import matplotlib.pyplot as plt


@dataclass
class SimulationInputs:
        
    # Diameter
    diameter_mean: int = 120
    diameter_cov: int = 5
    diameter_std: float = (diameter_mean * diameter_cov) /100

    # Thickness
    thickness_mean: int  = 4
    thickness_cov: int = 5
    thickness_std: float = (thickness_mean * thickness_cov) /100

    # Yield Strength
    yield_mean: int = 200
    yield_cov: int = 10
    yield_std: float = (yield_mean * yield_cov) /100

    # Internal Pressure
    internal_pressure: int = 10
    
    # Iterations
    iter_start: int = 1000
    iter_end: int = 10000
    iter_step: int = 1000

print("---------------------------------------Data CLass Object created ---------------------------------------")
simulation_data = SimulationInputs()

        
     
def initiate_simulation(iter_start = simulation_data.iter_start, iter_end = simulation_data.iter_end, iter_step = simulation_data.iter_step):
    """summary

    Args:
        iter_start (int, optional): define starting point for generating simulation values. Defaults to simulation_data.iter_start.
        iter_end (int, optional): define end point for generating simulation values. Defaults to simulation_data.iter_end.
        iter_step (int, optional): define the increment. Defaults to simulation_data.iter_step.
    """    
    try:
        print("---------------------------------------Initiating Simulations ---------------------------------------")
        runs = list(range(iter_start, iter_end, iter_step))
        sim_res = []
        for run in runs:
            sim_results = run_simulation(run)
            sim_res.append(sim_results)
            print(f'Simulation run for {run}: {sim_results}')  
        # print(f'sim_res: {sim_res}')
        plot_linechart(runs, sim_res)
        plot_histogram(runs, sim_res)
    
    except Exception as e:
        print(f'An exception occurred while trying to initiate the simulation: {e}')

def run_simulation(iterations):    
    """summary

    Returns:
        float: The function returns the number of negative values found in the simulation - negative value indicates that pipe failed

    Hoop Stress:
        int: Hoop stress =(Internal Pressure * Diameter)/(2 * Thickness)
    
    Objective:
        int: Failure = Yield - Hoop stress
    """        
    try:
              
        lst_diameter = simulate_values(simulation_data.diameter_mean, simulation_data.diameter_std,iterations, False)

        lst_thickness = simulate_values(simulation_data.thickness_mean, simulation_data.thickness_std,iterations, False)

        lst_yield = simulate_values(simulation_data.yield_mean, simulation_data.yield_std, iterations, False)

        df_final = pd.DataFrame(list(zip(lst_diameter, lst_thickness, lst_yield)), columns = ['Diameter', 'Thickness', 'Yield'])

        df_final['Hoop_Stress'] = simulation_data.internal_pressure * df_final['Diameter'] / (2 * df_final['Thickness'])

        df_final['Objective'] = df_final['Yield'] - df_final["Hoop_Stress"]
        
        min_stress = df_final['Hoop_Stress'].min()
        # print(f'The minimum stress value: {min_stress}')
        
        max_stress = df_final['Hoop_Stress'].max()
        # print(f'The maximum stress value: {max_stress}')
        # plot_fd(df_final, min_stress, max_stress)

        # print(styled_df((df_final)))
            
        return  df_final[df_final['Objective'] < 0].shape[0] / iterations   
    
    except Exception as e:
      print(f'An exception occurred with running the simulation: {e}')
        
def simulate_values(mu, sigma, iterations, print_output = True):
    """summary

    Args:
        mu (int): Define the mean of the diameter, thickness and yield
        sigma (int): Define the standard deviation of the diameter, thickness and yield
        iterations (int): Define number of iterations the simulations to be run
        print_output (bool, optional): Set value to True to view the print statement at various stages and set False to skip the prints. Defaults to True.

    Returns:
        List: The function returns the list which has the simulation values for diameter, thickess and yield
    """  
    try:
        result = []        
        for i in range(iterations):
            prob_value = round(random.uniform(.01, 0.99),3)
            sim_value = round(NormalDist(mu=mu, sigma= sigma).inv_cdf(prob_value),3)
            if print_output:
                print(f"The prob value is {prob_value} and the simulated value is {sim_value}")   
            result.append(sim_value)
        return result
    except Exception as e:
        print(f'An exception occurred while generating simulation values: {e}')

    
def plot_linechart(runs, sim_res):    
    try:
        print("---------------------------------------Initiating Simulation plot ---------------------------------------")
                
        y_mean = [np.mean(sim_res)]*len(runs)
        fig,ax = plt.subplots()
        ax.set_ylabel('Probability of Pipe Failure')
        ax.set_xlabel('Number of Iterations')
        ax.set_title('Montecarlo Simulation for Pipe Failure')
        # plt.plot(runs, sim_res, marker = '*')
        plt.plot(runs, sim_res, 'bo-', label = "Simulation Probabilities")
        for x,y in zip(runs, sim_res):            
            label = "{:.6f}".format(y)            
            plt.annotate(label, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 
        ax.plot(runs, y_mean, color='gray', lw=0.5, ls='dashdot', label=f'Average Probability: {round(y_mean[0],6)}')
        plt.legend(loc=0)
        plt.show()    
    except Exception as e:
      print(f'An error  occurred while trying to generate line chart {e}')

def plot_histogram(runs, sim_res):
    try:
        plt.hist(sim_res)
        plt.show()
    except Exception as e:
      print(f'An exception occurred while trying to generate the histogram: {e}')
    
# def plot_fd(df_final, min_stress, max_stress):
#     plt.hist(df_final['Hoop_Stress'], bins=np.arange(min_stress, min_stress+1))
#     plt.show()
    

def styled_df(df):
    """
    Styles DataFrames containing the inputs and years to retirement.
    """
    return df.style.format({
        'Diameter': '{:,.0f}', 
        'Thickness': '{:.1f}', 
        'Yield': '{:,.0f}', 
        'Hoop_Stress': '{:.1f}', 
        'Objective': '{:.1f}', 
        
    }).background_gradient(cmap='RdYlGn_r', subset='Objective')
        



  
