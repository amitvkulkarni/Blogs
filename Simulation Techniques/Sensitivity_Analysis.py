from pulp import *
import pandas as pd
import numpy as np

# Initialize Class
model = LpProblem("Product_Profits",LpMaximize)

# Define variables
A = LpVariable('A', lowBound=0)
B = LpVariable('B', lowBound=0)

# Define objective function: Profit on Product A and B
model += 30 * A + 45 * B 

# Constraint 1
model += 3 * A + 12 * B <= 150 


# Constraint 2
model += 4 * A + 3 * B <= 47

# Constraint 3
model += 5 * A + 2 * B <= 60

# Solve Model
model.solve()

print("Model Status:{}".format(LpStatus[model.status]))
print("Objective = ", round(value(model.objective),3))

for var in model.variables():
    print(var.name,"=", var.varValue)
    
   
res = [{'Name':name,'Constraint':const,'Price':const.pi,'Slack': const.slack} for name, const in model.constraints.items()]
print(pd.DataFrame(res))

####################################################################################################################################################
# Generating simulation #
####################################################################################################################################################
lst_constraint1 = list(range(140,151,1))
lst_constraint2 = list(range(45,48,1))
lst_constraint3 = list(range(35,40,1))

def sensitivity_table(lst_constraint1, lst_constraint2, lst_constraint3):
 
    """_summary_
    Args:
        L1 (List): Range of values of constraint 1
        L2 (List): Range of values of constraint 2
        L3 (List): Range of values of constraint 3

    Returns:
        Int: Returns the objective function value i.e profit
    """
    
    try:
        # Initialize Class, Define Vars., and Objective
        model = LpProblem("Product_Profits",LpMaximize)

        # Define variables
        A = LpVariable('A', lowBound=0)
        B = LpVariable('B', lowBound=0)

        # Define Objetive Function: Profit on Product A and B
        model += 30 * A + 45 * B 

        # Constraint 1
        model += 3 * A + 12 * B <= lst_constraint1

        # Constraint 2
        model += 4 * A + 3 * B <= lst_constraint2

        # Constraint 3
        model += 5 * A + 2 * B <= lst_constraint3

        # Solve Model
        model.solve()

        print("Model Status:{}".format(LpStatus[model.status]))
        print("Objective = ", round(value(model.objective),3))
        
        for var in model.variables():
            print(var.name,"=", var.varValue)
            print(f'"lst_constraint1" = {lst_constraint1}, "lst_constraint2" = {lst_constraint2}, "lst_constraint3" = {lst_constraint1}')
        res = [{'Name':name,'Constraint':const,'Price':const.pi,'Slack': const.slack} for name, const in model.constraints.items()]
        print(pd.DataFrame(res))
        return round(value(model.objective),2)
        
    except Exception as e:
        print(f'An exception occurred while trying to initiate the simulation: {e}')

res = [(p3, p2, p1, sensitivity_table(p1, p2, p3)) for p3 in lst_constraint3 for p2 in lst_constraint2 for p1 in lst_constraint1]

df = pd.DataFrame(res, columns= ['Constraint 3','Constraint 2', 'Constraint 1', 'Objective'])
df_pivot = df.pivot(index = 'Constraint 1', columns = ['Constraint 2', 'Constraint 3'], values = 'Objective')
df_pivot
# df_pivot.style.background_gradient(cmap='ocean_r')



####################################################################################################################################################
# Simulation with itertools
####################################################################################################################################################

import itertools

res_list = [(var[0], var[1], var[2], sensitivity_table(var[0], var[1], var[2])) for var in itertools.product(lst_constraint1, lst_constraint2, lst_constraint3)]
res_list
df_res_list = pd.DataFrame(res_list, columns= ['Constraint 3','Constraint 2', 'Constraint 1', 'Objective'])
df_res_list_pivot = df.pivot(index = 'Constraint 1', columns = ['Constraint 2', 'Constraint 3'], values = 'Objective')
df_res_list_pivot


####################################################################################################################################################
# Simulation with SensitivityAnalyzer
####################################################################################################################################################

from sensitivity import SensitivityAnalyzer

# creating a dictionary of constraints
sensitivity_dict = {
    'lst_constraint1' : lst_constraint1,
    'lst_constraint2' : lst_constraint2,
    'lst_constraint3' : lst_constraint3
}

sa = SensitivityAnalyzer(sensitivity_dict, sensitivity_table)
ddf = sa.df
ddf.style.hide_index()

plot = sa.plot()







