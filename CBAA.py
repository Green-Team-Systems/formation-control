#%%
from hashlib import new
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

global test_num  #here for generating figures all at once
np.set_printoptions(precision=2)


##edit output to go to a file instead of the terminal
f = open("CBAA_output.txt", 'w')
sys.stdout = f
w_cost = 0
t_cost = 0
w_time = 0
t_time = 0
w_rounds = 0
t_rounds = 0
dist_list = list()


#%%
def nbr(adj: np.array, i: int):

    return np.nonzero(adj[:,i])


def arun(q: np.array, p: np.array):
    '''
    Assume q and p are d x n (d: 2D or 3D)

    Minimizes the ||q - (Rp + t)
    '''
    d = len(q)
    e = len(p)
    print(q)
    print(p)
    #shift point clouds by centroid
    mu_q = np.mean(q)
    mu_p = np.mean(p)

    Q = q - mu_q
    P = p - mu_p

    #Construct H matrix
    H = Q * np.transpose(P)
    print(H)

    return

def calculate_costs(position: PosVec3, target) -> float:
    '''
    Returns cost, in this case that is the distance from a drone to a target
    Using 2D distance formula
    '''
    return np.sqrt((target[0] - position.X)**2 + (target[1] - position.Y)**2)


def CBAA_swarm(adj: np.array, targets: np.array, drones: dict ) -> np.array :
    '''
    CBAA implementation for swarm assignment
    Source: MIT ACL Lab / aclswarm

    Inputs:
    -adj: n x n adjacency matrix
    -pm : 3 x n desired formation points
    -qm : 3 x n matrix of current vehicle positions

    assign 1 x n permutation vector
    Maps idx of qm to idx of pm (formpt)
    '''

    #begin timer
    start = time.time()
    n = len(list(drones.keys())) #number of agents
    drone_ids = list()
    for name in drones.keys():
        drone_id = int(name.removeprefix('Drone'))
        drone_ids.append(drone_id)
    id = np.identity(n)
    G = np.array([[adj[i][j] + id[i][j]  for j in range(len(adj[0]))] for i in range(len(adj))])
    
    rounds = 10
    global w_rounds, t_rounds
    for j in range(rounds):
        task_assigned = 0
        #print("Round {}".format(j + 1))
         # Phase I: Bid
        for drone_name in drones.keys():
            generate_bids(drone_name, drones[drone_name], targets, j)
        # Phase 2: Determine Taskings
        for i, drone_name in enumerate(drones.keys()):
            transmit_bids(drone_name, drones, drone_ids)
        for i, drone_name in enumerate(drones.keys()):
            determine_task_list(drone_name, drones[drone_name])

        for name, info in drones.items():
            #print("{}: {}".format(name, info["TaskList"]))
            task_assigned += sum(info["TaskList"])

        #print("\n")

        if task_assigned == len(targets):
            print("Task assignments complete")
            print("Total number of rounds:", j + 1)
            if((j + 1) > w_rounds): w_rounds = (j + 1)
            t_rounds += (j + 1)
            break #comment this out once we can replicate
    end = time.time()
    timmy = end - start
    global w_time, t_time
    if(timmy > w_time): w_time = timmy
    t_time += timmy
    print("Task Allocation took {} seconds".format(timmy))
    print("\n")
    plot_targets(targets, drones, False, True)

def generate_bids(drone_name: str, drone_info: dict, targets: list, rnd: int) -> float:
    position = drone_info["Position"]
    drone_info["Bids"] = [0 for _ in range(len(targets))]
    pos_diffs = list()
    if debug:
        print("{}: {}".format(drone_name, drone_info["WinningBids"]))
        print("{}: {}".format(drone_name, drone_info["RawBids"]))
    if np.sum(drone_info["TaskList"]) == 0:
        availability = drone_info["TaskAvailability"]
        for i, target in enumerate(targets):
            pos_diff = ned_position_difference(target, position)
            if pos_diff == 0:
                drone_info["RawBids"][i] = float("Inf")
                bid = float("Inf")
            else:
                drone_info["RawBids"][i] = (1 / pos_diff)
                bid = (1 / (pos_diff))
            pos_diffs.append(pos_diff)

            if bid > (drone_info["WinningBids"][i]):
                availability[i] = 1
                drone_info["Bids"][i] = bid
            else:
                availability[i] = 0
                # drone_info["Bids"][i] = (1 / (pos_diff + 500))
        if debug:
            print("{}: {}".format(drone_name, pos_diffs))
            print("{}: {}".format(drone_name, drone_info["Bids"]))
            print("{}: {}".format(drone_name, availability))
        if sum(availability) != 0:
            task_selected = np.argmax([drone_info["Bids"][i] * availability[i] for i in range(len(targets))])
            drone_info["WinningBids"][task_selected] = drone_info["Bids"][task_selected]
            drone_info["SwarmBids"][drone_name.lstrip("Drone")] = copy.deepcopy(drone_info["WinningBids"])

            drone_info["TaskList"][task_selected] = 1
   
    if debug:
        print("{}: {}".format(drone_name, drone_info["WinningBids"]))
        print("{}: {}".format(drone_name, drone_info["RawBids"]))
        print("{} Task List: {}".format(drone_name, drone_info["TaskList"]))


def transmit_bids(drone_name: str, drones: dict, drone_ids: list) -> None:
    bid_list = drones[drone_name]["WinningBids"]
    drone_id = int(drone_name.lstrip("Drone"))
    for swarm_id in drone_ids:
        if swarm_id != drone_id:
            swarm_name = "Drone{}".format(swarm_id)
            drones[swarm_name]["SwarmBids"][str(drone_id)] = copy.deepcopy(bid_list)


def determine_task_list(drone_name: str, drone_info: dict) -> None:
    drone_id = int(drone_name.lstrip("Drone")) - 1
    swarm_bids = drone_info["SwarmBids"]
    task_list = drone_info["TaskList"]
    if debug:
        print("{} Task List: {}".format(drone_name, task_list))
        print("{} Winning Bids: {}".format(drone_name, drone_info["WinningBids"]))
    bid_columns = [list() for _ in range(len(swarm_bids.keys()))]
    selections = [None for _ in range(len(swarm_bids.keys()))]
    for id, bid_list in swarm_bids.items():
        for i, bid in enumerate(bid_list):
            bid_columns[i].append(bid)

    for j, col in enumerate(bid_columns):
        if debug:
            print("Task {} Bids: {}".format(j + 1, col))
        best_bid = np.max(col)
        if drone_info["Winners"][j] and best_bid == drone_info["Winners"][j]["bid"]:
            continue
        selections[j] = np.argmax(col)
        drone_info["Winners"][j] = dict(id=selections[j], bid=col[selections[j]])
        drone_info["WinningBids"][j] = col[selections[j]]
    if debug:
        print("{} Raw Bids: {}".format(drone_name, drone_info["RawBids"]))
    for j, winner in enumerate(drone_info["Winners"]):
        winner_id = winner["id"]
        if winner_id != drone_id:
            task_list[j] = 0
    if debug:
        print("{} tasks: {}".format(drone_name, task_list))


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

    total_dist = 0
    for name, info in drones.items():
            for i in range(len(info["TaskList"])):
                if(info["TaskList"][i] == 1):
                    #print("{} goes to target {}".format(name, i+1))
                    xs = [target_list_x[i], info["Position"].X]
                    ys = [target_list_y[i], info["Position"].Y]
                    total_dist += calculate_costs(info["Position"], (target_list_x[i], target_list_y[i]))
                    if show_lines:
                        plt.plot(xs, ys, 'g--')
    #plt.xlim(-400, 400)
    #plt.ylim(-400, 400)
    global w_cost, t_cost, dist_list
    print("Total Cost is {:.2f} meters\n".format(total_dist))
    if(total_dist > w_cost): w_cost = total_dist
    t_cost += total_dist
    dist_list.append(total_dist)
    plt.axis('square')
    plt.xlabel("X-Axis (meters)")
    plt.xlabel("Y-Axis (meters)")
    #plt.show()
    filename = 'output_images/CBAA_test_{}_figure.png'.format(test_num)
    plt.savefig(filename)
    plt.clf()

#%%
#--------------------TEST------------------
if __name__ == "__main__":
    ##add in how many bits of communication it takes
    debug = False
    gen_drones = True
    gen_targets = True
    map_boundary = {"+X": 800,
                    "-X": -800,
                    "+Y": 800,
                    "-Y": -800} # Meters
    min_separation_distance = 25 # meters
    numb_drones = numb_targets =  6 #test data is in groups of 6
    random.seed(1)
    for j in range(9): #run 100 all random tests
        print('\nBeginning Test ' +str(j))
        test_num = j
        #if(i > 9): #random_testing
            #numb_drones = numb_targets = i*2
        new_targets, new_drones = test_cases(j) #see utils/test_cases.py for usage
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
                "5": None,
                "6": None
            }
        }
        Adj = []
        CBAA_swarm(Adj, new_targets, drones)
    print(dist_list)
    sys.stdout = sys.__stdout__ #set it back
    print("\nDone Testing\n")
    print("Time: average was {} and worst was {}".format(t_time / 100, w_time))
    print("Cost: average was {} and worst was {}".format(t_cost / 100, w_cost))
    print("Rounds: average was {} and worst was {}".format(t_rounds / 100, w_rounds))

# %%