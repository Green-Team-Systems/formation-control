#%%
from hashlib import new
from matplotlib.collections import LineCollection
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import random
import time

from numpy.core.fromnumeric import shape

from utils.data_classes import PosVec3, Quaternion
from utils.distance_utils import ned_position_difference

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

    for j in range(rounds):
        task_assigned = 0
        print("Round {}".format(j + 1))
         # Phase I: Bid
        for drone_name in drones.keys():
            generate_bids(drone_name, drones[drone_name], targets, j)
        # Phase 2: Determine Taskings
        for i, drone_name in enumerate(drones.keys()):
            transmit_bids(drone_name, drones, drone_ids)
        for i, drone_name in enumerate(drones.keys()):
            determine_task_list(drone_name, drones[drone_name])

        for name, info in drones.items():
            print("{}: {}".format(name, info["TaskList"]))
            task_assigned += sum(info["TaskList"])

        print("\n")

        if task_assigned == len(targets):
            print("Task assignments complete")
            print("Total number of rounds:", j + 1)
            break #comment this out once we can replicate
    end = time.time()
    print("Task Allocation took {} seconds".format(end-start))
    print("\n")
    plot_targets(targets, drones)

def generate_bids(drone_name: str, drone_info: dict, targets: list, rnd: int) -> float:
    #print(drone_info)
    position = drone_info["Position"]
    drone_info["Bids"] = [0 for _ in range(len(targets))]
    availability = drone_info["TaskAvailability"]
    pos_diffs = list()
    if debug:
        print("{}: {}".format(drone_name, drone_info["WinningBids"]))
        print("{}: {}".format(drone_name, drone_info["RawBids"]))
    if np.sum(drone_info["TaskList"]) == 0:
        for i, target in enumerate(targets):
            pos_diff = ned_position_difference(target, position)
            if(not pos_diff): drone_info["RawBids"][i] = float('inf') #temp fix to get around div by 0
            else: drone_info["RawBids"][i] = (1 / pos_diff)
            pos_diffs.append(pos_diff)
            if(not pos_diff): bid = float('inf')#same as above
            else: bid = (1 / (pos_diff))

            if bid >= (drone_info["WinningBids"][i]):
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
            drone_info["SwarmBids"][drone_name.removeprefix('Drone')] = copy.deepcopy(drone_info["WinningBids"])

            drone_info["TaskList"][task_selected] = 1
    else:
        task_selected = np.argmax(drone_info["TaskList"])

        for i, raw_bid in enumerate(drone_info["RawBids"]):
            if i != task_selected and (raw_bid in drone_info["WinningBids"]):
                drone_info["WinningBids"][i] = 0

                # drone_info["Bids"][i] = (1 / (pos_diff + 500))
        
        # print("{} selected Task {}".format(drone_name, task_selected))
    if debug:
        print("{}: {}".format(drone_name, drone_info["WinningBids"]))
        print("{}: {}".format(drone_name, drone_info["RawBids"]))
        print("{} Task List: {}".format(drone_name, drone_info["TaskList"]))


def transmit_bids(drone_name: str, drones: dict, drone_ids: list) -> None:
    bid_list = drones[drone_name]["WinningBids"]
    drone_id = int(drone_name.removeprefix('Drone'))
    for swarm_id in drone_ids:
        if swarm_id != drone_id:
            swarm_name = "Drone{}".format(swarm_id)
            drones[swarm_name]["SwarmBids"][str(drone_id)] = copy.deepcopy(bid_list)


def determine_task_list(drone_name: str, drone_info: dict) -> None:
    #print("reached begin determine_task_list")
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
        #print("type is",type(np.argmax(col))) #find out the type if there are multiple
        selections[j] = np.argmax(col)
        drone_info["Winners"][j] = dict(id=selections[j], bid=col[selections[j]])
        drone_info["WinningBids"][j] = col[selections[j]]
    if debug:
        print("{} Raw Bids: {}".format(drone_name, drone_info["RawBids"]))
    for i, winner in enumerate(drone_info["WinningBids"]):
        if winner not in drone_info["RawBids"]:
            task_list[i] = 0
    if debug:
        print("{} tasks: {}".format(drone_name, task_list))


def plot_targets(targets, drones):
    #updating to show paths
    target_list_x = list()
    target_list_y = list()
    labels = list()
    for i, target in enumerate(targets):
        labels.append("Target {}".format(i + 1))
        target_list_x.append(target.X)
        target_list_y.append(target.Y)
    
    plt.scatter(target_list_x, target_list_y)
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
        
    for (label, x, y) in zip(labels, drone_pos_x, drone_pos_y):
        plt.text(x=x, y=y, s=label)

    
    for name, info in drones.items():
            for i in range(len(info["TaskList"])):
                if(info["TaskList"][i] == 1):
                    print("{} goes to target {}".format(name, i+1))
                    xs = [target_list_x[i], info["Position"].X]
                    ys = [target_list_y[i], info["Position"].Y]
                    plt.plot(xs, ys, 'g--')
    
    plt.xlabel("X-Axis (meters)")
    plt.xlabel("Y-Axis (meters)")
    plt.show()


def generate_random_targets(numb_targets):
    targets = list()
    for _ in range(numb_targets):
        targets.append(
            PosVec3(
                X=random.randrange(map_boundary["-X"],
                                   map_boundary["+X"],
                                   min_separation_distance),
                Y=random.randrange(map_boundary["-X"],
                                   map_boundary["+X"],
                                   min_separation_distance),
                Z=-1
            )
        )
    return targets

def generate_equal_targets(num):
    targets = list()
    targets.append(PosVec3(X=100, Y=-250, Z=-1))
    targets.append(PosVec3(X=-25, Y=225, Z=-1))
    targets.append(PosVec3(X=375, Y=200, Z=-1))
    targets.append(PosVec3(X=125, Y=-75, Z=-1))
    targets.append(PosVec3(X=-150, Y=-75, Z=-1))
    targets.append(PosVec3(X=325, Y=-300, Z=-1))
    return targets

def generate_equal_drones(num):
    drones = list()
    drones.append(PosVec3(X=-150, Y=375, Z=-1))
    drones.append(PosVec3(X=225, Y=-325, Z=-1))
    drones.append(PosVec3(X=-50, Y=-100, Z=-1))
    drones.append(PosVec3(X=350, Y=-200, Z=-1))#changed from 200
    drones.append(PosVec3(X=200, Y=175, Z=-1))
    drones.append(PosVec3(X=50, Y=75, Z=-1))
    return drones


#%%
#--------------------TEST------------------
if __name__ == "__main__":
    debug = False
    gen_drones = False #changed to read from pre-written test scenario, make sure numb_drones is 6
    gen_targets = False
    map_boundary = {"+X": 400,
                    "-X": -400,
                    "+Y": 400,
                    "-Y": -400} # Meters
    min_separation_distance = 25 # meters
    numb_drones = 6 #changed to make faster
    numb_targets = numb_drones
    if gen_drones:
        new_drones = generate_random_targets(numb_drones)    
    else:
        new_drones = generate_equal_drones(numb_drones)
        """
        with open("settings.json", "r") as f:
            settings = json.load(f)
        """
    if gen_targets:
        new_targets = generate_random_targets(numb_targets)
    else:
        new_targets = generate_equal_targets(numb_targets)
        """
        with open("targets.json", "r") as f:
            targets = json.load(f)
        new_targets = list()
        for target in targets["Targets"]:
            new_targets.append(
                PosVec3(
                    X=target["X"],
                    Y=target["Y"],
                    Z=target["Z"]
                )
            )
        """
    drones = dict()
    for i in range(numb_drones):
        drone_name = "Drone{}".format(i + 1)
        drones[drone_name] = {
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
        # print(drones[drone_name])        
        
        if gen_drones:
            drones[drone_name]['Position'] = new_drones[i]
        else:
            drones[drone_name]['Position'] = new_drones[i]
            """
            drones[drone_name]["Position"]= PosVec3(
                X=settings["Vehicles"][drone_name]["X"],
                Y=settings["Vehicles"][drone_name]["Y"],
                Z=settings["Vehicles"][drone_name]["Z"],
                frame="global"
            )
            """
        
    #fully_connected Adjacency Matrix
    # Adj = np.ones(numb_drones) - np.identity(numb_drones)
    Adj = []

    CBAA_swarm(Adj, new_targets, drones)
# %%
