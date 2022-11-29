import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from  matplotlib.widgets import Cursor
import random
import math
from matplotlib.artist import Artist
import itertools
import datetime
from PIL import Image, ImageDraw


start_time = datetime.datetime.now()
print(f'start_time : {start_time}')

iterations = 50
counter = 0
np.random.seed(123)
arr = np.random.uniform(0,50, size = (iterations,2))

point = random.choice(arr)
print(f'The selected point - {point}')

def distance2(p1,p2):
    return round(math.dist(p1, p2),2)


fig, ax = plt.subplots()
ax.scatter(arr[:,0], arr[:, 1], alpha = 0.5)

cursor = Cursor(ax, horizOn=True, vertOn=True, linewidth = 2.0, color = 'Red')

# fig.canvas.mpl_connect('button_press_event', onPress)

# min_idx = np.argmin([(distance2(arr[i], point)) for i in range(len(arr))])
min_idx = np.argmin([(math.dist(arr[i], point)) for i in range(len(arr))])
nearest_point = arr[min_idx]     
other_pts = np.delete(arr, min_idx, axis = 0)     

x_final = []
y_final = []

x_final.append(point[0])
y_final.append(point[1])

x_final.append(nearest_point[0])
y_final.append(nearest_point[1])

for i in range(iterations-1):
    counter += 1
    ax.plot(nearest_point[0], nearest_point[1], marker = 'o', markersize = 8, color = 'r')
    min_idx = np.argmin([(math.dist(other_pts[i], point)) for i in range(len(other_pts))])
    nearest_point = other_pts[min_idx]            
       
    x_final.append(nearest_point[0])
    y_final.append(nearest_point[1])
        
    other_pts = np.delete(other_pts, min_idx, axis = 0)     
    print(f'New point - {point}')
    plt.plot(x_final, y_final, linestyle="-", color = 'g', marker = 'o',markersize = 10, alpha = 0.3)
    
    for i_x, i_y in zip(x_final, y_final):
        plt.text(i_x + 0.5, i_y+ 1, '({}, {})'.format(round(i_x,1), round(i_y,1)), color = 'b', fontsize = 10)
    
    mag_near = round(distance2(x_final, y_final),2)
    x_midpoint = (point[0]+ nearest_point[0])/2
    y_midpoint = (point[1]+ nearest_point[1])/2
    ax.text(x_midpoint,y_midpoint, mag_near)
    
    point = nearest_point       
    
    title_text = 'CONNECTING THE NEAREST POINT - ' + str(counter+1) + ' @'+ str((round(i_x,1), round(i_y,1))) + ' with Coverage of: ' + str(round((counter+1)/iterations*100,0)) + ' %'
                   
    plt.title(title_text, fontweight='bold')      
    
    ax.plot(point[0], point[1], marker = 'o', markersize = 10, color = 'r')  
    
    plt.pause(0.0000001)

plt.show()
# end_time = datetime.datetime.now()       
print(f'End time : {end_time}')
print(f'Execution time: {end_time - start_time}')