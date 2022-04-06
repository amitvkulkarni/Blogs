import numpy as np
import pandas as pd
import random
import os
import yaml
from dataclasses import dataclass
from statistics import NormalDist
import matplotlib.pyplot as plt

x = []
y = []
for i in range(50000):
    prob_value = round(random.uniform(.01, 0.99),3)
    sim_value = round(NormalDist(mu=150, sigma= 10).inv_cdf(prob_value),3)
    x.append(prob_value)
    y.append(sim_value)

    # print(f"The prob value is {prob_value} and the simulated value is {sim_value}") 

# plt.plot(x, y)

# plt.hist(y, bins=50)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency')


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(x, bins=10)
axs[1].hist(y, bins=10)
