# ================================================================
# Created by: Jack Roy
# Created On: March 2022
# Updated On: March 2022
#
# Description: DHBA implementation for SWARMS to run with Airsim
# ================================================================

#%%
from cgitb import small
from hashlib import new
from turtle import position
from xml.dom import minidom
from matplotlib.collections import LineCollection
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import random
import time
import sys

from numpy.core.fromnumeric import shape

from utils.data_classes import PosVec3, Quaternion
from utils.distance_utils import ned_position_difference
from utils.test_cases import test_cases

np.set_printoptions(precision=2)
#%%

global test_num  #here for generating figures all at once

def calculate_costs(position: PosVec3, target: PosVec3) -> float:
    '''
    Returns cost, in this case that is the distance from a drone to a target
    Using 2D distance formula
    '''
    return np.sqrt((target.X - position.X)**2 + (target.Y - position.Y)**2)

def calc_cost_matrix(drones, targets):
    '''
    Calculates cost matrix according to calculate_costs function
    '''
    ans = []
    for drone_name in drones.keys():
        row = list()
        for t in targets:
            row.append(calculate_costs(drones[drone_name]['Position'], t))
        ans.append(row)
    return ans

def assign_task(drone_name: str, drone_index: int, drone_info: dict, assignments: list, targets: list) -> float:
    '''
    Assigns one task to one drone with name and index
    format for assignments is (drone index #, task number)
    also calculates total cost of assigning those tasks and returns it as a float
    '''
    for a in assignments:
        #print("a is "+str(a))
        if(a[0] == drone_index): 
            drone_info["TaskList"][a[1]] = 1
            cost = calculate_costs(drone_info["Position"], targets[a[1]])
            return cost
    #should no longer reach this point since assignment works properyl
    print("ERROR: assignment not found!!")
    return 0

def zero_marker(mat):
        '''
        Takes in a matrix and return the zeros, rows, and columns
        '''
        bool_matrix = (np.array(mat) == 0) #true if 0 false if nonzero
        bmc = copy.deepcopy(bool_matrix)
        if(debug):
            print("cost bool matrix is")
            print(np.array(bool_matrix))
            print()
        #find a row with the fewest zeros
        marked_zeros = [] #stores coordinates of zeros to use
        #find the first row with >= 1 zero and mark it, then repeat until everything is marked
        while(np.sum(bool_matrix == True) > 0):
            min_row = [float('inf'), -1]#number of zeros, index
            for row_num in range(bool_matrix.shape[0]):#numpy function to obtain rows
                zeros = np.sum(bool_matrix[row_num] == True)
                if((zeros > 0) and (min_row[0] > zeros)):
                    min_row = [zeros, row_num]
            #mark all of that row, and the zero's location's columns as false
            zero_cols = np.where(bool_matrix[min_row[1]] == True)[0][0]
            marked_zeros.append((min_row[1], zero_cols))
            bool_matrix[min_row[1],:] = False
            bool_matrix[:,zero_cols] = False
        if(debug): 
            print("marked zeros are")
            print(marked_zeros)
            print()
        #find non marked rows and marked columns
        marked_zero_rows = np.sort([mz[0] for mz in marked_zeros])
        marked_zero_cols = np.sort([mz[1] for mz in marked_zeros])
        if(debug):
            print("mzr is "+str(marked_zero_rows))
            print("mzc is "+str(marked_zero_cols))
        #find non marked rows, and correspoiding columns
        non_marked_rows = list(set(range(len(bool_matrix))) - set(marked_zero_rows))
        final_marked_cols = []
        updated = True
        while(updated):
            updated = False
            for i in range(len(non_marked_rows)):
                row_columns = bmc[non_marked_rows[i],:] #bad name but it makes sense in my head
                for j in range(len(row_columns)):
                    if(row_columns[j] and j not in final_marked_cols):
                        final_marked_cols.append(j)
                        updated = True
            for row_i, col_i in marked_zeros:
                if(row_i not in non_marked_rows and col_i in final_marked_cols):
                    non_marked_rows.append(row_i)
                    updated = True
        final_marked_rows = list(set(range(len(mat))) - set(non_marked_rows))
        return(marked_zeros, final_marked_rows, final_marked_cols)

def adjust_cost_matrix(mat, rows, cols):
    """
    Adjust the matrix for another round of DHBA
    This is necessary when no solution is found
    """
    #find the smallest non-zero element not in the marked area
    smallest_element = float('inf')
    for r in range(len(mat)):
        if r not in rows:
            for c in range(len(mat[r])):
                if c not in cols:
                    if mat[r][c] < smallest_element:
                        smallest_element = mat[r][c]
    if(debug):
        print("smallest unmarked value is "+str(smallest_element))
    #subtract the smallest from all values not in the marked area
    #add the smallest to all areas overlapped
    for r in range(len(mat)):
        for c in range(len(mat[r])):
            if r not in rows and c not in cols:
                mat[r][c] = mat[r][c] - smallest_element
            elif r in rows and c in cols:
                mat[r][c] = mat[r][c] + smallest_element
    
    #return the adjusted matrix
    return mat

def DHBA(targets, drones):
    '''
    Implementation of DHBA for SWARM Drones
    Utilized information from the following
    https://drive.google.com/file/d/1iBqWmiTlzEEPhY41vaB3PQhOsgQDI8uU/view
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    Various websites for numpy functions
    '''
    start = time.time()
    cost_matrix = calc_cost_matrix(drones, targets)
    rounds = 0
    
    if(debug):
        print("initial cost matrix")
        print(np.array(cost_matrix))
        print()
    
    #cost_matrix is now calculated, begin implementation of HA
    #Subtract the minimum value from each row and column
    for row_num in range(len(cost_matrix)):
        cost_matrix[row_num] -= min(cost_matrix[row_num]) #row
    for col_num in range(len(cost_matrix)):
        col = [row[col_num] for row in cost_matrix]
        col_min = min(col) #it updates in the loop so I have to call it outside
        for row_num in range(len(cost_matrix)):
            cost_matrix[row_num][col_num] -= col_min #column
    
    if(debug):
        print("cost matrix after subtracting minimums")
        print(np.array(cost_matrix))
        print()

    #find 0 values in the cost matrix and use that to assign tasks
    num_assignments = 0
    while(num_assignments < len(drones)):
        rounds += 1
        marked_zeros, marked_rows, marked_cols = zero_marker(cost_matrix)
        
        if(debug):
            print("marked zeros are "+str(marked_zeros))
            print("marked rows (drones) are "+str(marked_rows))
            print("marked columns (targets) are " +str(marked_cols))
        
        #this is the first point where there could be a solution
        #if we did we are done, otherwise you have to adjust the matrix and do another marking
        num_assignments = len(marked_rows)+len(marked_cols)
        if(num_assignments < len(drones)):
            cost_matrix = adjust_cost_matrix(cost_matrix, marked_rows, marked_cols)
            
            if(debug):
                print("Did not find a solution, adjusted matrix to")
                print(np.array(cost_matrix))

    #assign tasks and calculate total distance covered by drones
    total_dist = 0
    for i, drone_name in enumerate(drones.keys()):
        total_dist += assign_task(drone_name, i, drones[drone_name], marked_zeros, targets)
    
    plot_targets(targets, drones, True, True)


def plot_targets(targets, drones, show_labels=True, show_lines=True):
    '''
    Plots drones and targets, added booleans to show labels and show lines
    '''
    target_list_x = list()
    target_list_y = list()
    labels = list()
    for i, target in enumerate(targets):
        labels.append("Target {}".format(i + 1))
        target_list_x.append(target.X)
        target_list_y.append(target.Y)
    
    plt.scatter(target_list_x, target_list_y)
    if show_labels:
        for (label, x, y) in zip(labels, target_list_x, target_list_y):
            plt.text(x=x, y=y, s=label)
    
    drone_pos_x = list()
    drone_pos_y = list()
    labels = list()
    for i, (drone_name, drone_info) in enumerate(drones.items()):
        labels.append(drone_name)
        drone_pos_y.append(drone_info["Position"].Y)
        drone_pos_x.append(drone_info["Position"].X)
    plt.scatter(drone_pos_x, drone_pos_y)
    
    if show_labels:
        for (label, x, y) in zip(labels, drone_pos_x, drone_pos_y):
            plt.text(x=x, y=y, s=label)

    
    for name, info in drones.items():
            for i in range(len(info["TaskList"])):
                if(info["TaskList"][i] == 1):
                    #print("{} goes to target {}".format(name, i+1))
                    xs = [target_list_x[i], info["Position"].X]
                    ys = [target_list_y[i], info["Position"].Y]
                    if show_lines:
                        plt.plot(xs, ys, 'g--')
    #plt.xlim(-400, 400)
    #plt.ylim(-400, 400)
    plt.axis('square')
    plt.xlabel("X-Axis (meters)")
    plt.xlabel("Y-Axis (meters)")
    plt.show()
    #filename = 'output_images2/DHBA_test_{}_figure.png'.format(test_num)
    #plt.savefig(filename)
    #plt.clf()

#%%
#--------------------TEST------------------
if __name__ == "__main__":
    ##add in how many bits of communication it takes
    ##expand to read from json files
    debug = False
    read_json = True ###DONT TOUCH
    map_boundary = {"+X": 800,
                    "-X": -800,
                    "+Y": 800,
                    "-Y": -800} # Meters
    min_separation_distance = 25 # meters
    numb_drones = numb_targets =  6 #test data is in groups of 6
    #random.seed(1)
    if(read_json):
        print("reading json")
        with open("settings2.json", 'r') as f:
            settings = json.load(f) #drones
        with open("targets2.json", 'r') as f:
            targets = json.load(f)
        new_targets = list()
        for t in targets["Targets"]:
            new_targets.append(PosVec3(X=t["X"],Y=t["Y"], Z=t["Z"]))
        print("found " + str(len(new_targets)) +" drones")
        drones = dict()
        for i in range(numb_drones):
            drone_name = "Drone{}".format(i + 1)
            drones[drone_name] = {
                "Position": PosVec3(
                X=settings["Vehicles"][drone_name]["X"],
                Y=settings["Vehicles"][drone_name]["Y"],
                Z=settings["Vehicles"][drone_name]["Z"],
                frame="global"
            ),
            "TaskAvailability": [0 for _ in range(len(new_targets))],
            "WinningBids": [0 for _ in range(len(new_targets))],
            "Winners": [None for _ in range(len(new_targets))],
            "TaskList": [0 for _ in range(len(new_targets))],
            }
        print("finished reading from json")
        DHBA(new_targets, drones)

    else:
        for j in range(100): #run 50 tests to get full suite
            print('\nBeginning Test ' +str(j))
            test_num = j
            #if(i > 9): #random_testing
                #numb_drones = numb_targets = i*2
            new_targets, new_drones = test_cases(11) #see utils/test_cases.py for usage
            if(j == 97):
                print(new_targets)
                print(new_drones)
            drones = dict()
            for i in range(numb_drones):
                drone_name = "Drone{}".format(i + 1)
                drones[drone_name] = {
                    "Position": new_drones[i],
                    "TaskAvailability": [0 for _ in range(len(new_targets))],
                    "WinningBids": [0 for _ in range(len(new_targets))],
                    "Winners": [None for _ in range(len(new_targets))],
                    "TaskList": [0 for _ in range(len(new_targets))],
                    "RawBids": [0 for _ in range(len(new_targets))],
                    "SwarmBids": {
                        "1": None,
                        "2": None,
                        "3": None,
                        "4": None,
                        "5": None
                    }
                }

            DHBA(new_targets, drones)
    print("\nDone Testing\n")
# %%
