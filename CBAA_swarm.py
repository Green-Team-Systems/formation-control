import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import random

from numpy.core.fromnumeric import shape

from utils.data_classes import PosVec3, Quaternion
from utils.distance_utils import ned_position_difference


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
    n = len(list(drones.keys())) #number of agents
    drone_ids = list()
    for name in drones.keys():
        drone_id = int(name[-1])
        drone_ids.append(drone_id)
    id = np.identity(n)
    G = np.array([[adj[i][j] + id[i][j]  for j in range(len(adj[0]))] for i in range(len(adj))])

    for j in range(10):
        print("Round {}".format(j + 1))
         # Phase I: Bid
        for drone_name in drones.keys():
            generate_bids(drone_name, drones[drone_name], targets, j)
        # Phase 2: Determine Taskings
        for i, drone_name in enumerate(drones.keys()):
            transmit_bids(drone_name, drones, drone_ids)
        
        for i, drone_name in enumerate(drones.keys()):
            # print("=" * 80)
            determine_task_list(drone_name, drones[drone_name])
            # print("=" * 80)
    
        for name, info in drones.items():
            print("{}: {}".format(name, info["TaskList"]))
        
        print("\n")

    plot_targets(targets, drones)

def generate_bids(drone_name: str, drone_info: dict, targets: list, rnd: int) -> float:
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
            drone_info["RawBids"][i] = (1 / pos_diff)
            pos_diffs.append(pos_diff)
            bid = (1 / (pos_diff))

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
            drone_info["SwarmBids"][drone_name[-1]] = copy.deepcopy(drone_info["WinningBids"])

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
    drone_id = int(drone_name[-1])
    for swarm_id in drone_ids:
        if swarm_id != drone_id:
            swarm_name = "Drone{}".format(swarm_id)
            drones[swarm_name]["SwarmBids"][str(drone_id)] = copy.deepcopy(bid_list)


def determine_task_list(drone_name: str, drone_info: dict) -> None:
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
    
    target_list_x = list()
    target_list_y = list()
    labels = list()
    for i, (drone_name, drone_info) in enumerate(drones.items()):
        labels.append(drone_name)
        target_list_y.append(drone_info["Position"].Y)
        target_list_x.append(drone_info["Position"].X)
    
    plt.scatter(target_list_x, target_list_y)
    for (label, x, y) in zip(labels, target_list_x, target_list_y):
        plt.text(x=x, y=y, s=label)
    
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

#--------------------TEST------------------
if __name__ == "__main__":
    debug = False
    gen_targets = True
    map_boundary = {"+X": 400,
                    "-X": -400,
                    "+Y": 400,
                    "-Y": -400} # Meters
    min_separation_distance = 25 # meters
    numb_drones = 5
    numb_targets = numb_drones
    with open("settings.json", "r") as f:
        settings = json.load(f)
    if gen_targets:
        new_targets = generate_random_targets(numb_targets)
    else:
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
            "RawBids": [0 for _ in range(len(new_targets))],
            "SwarmBids": {
                "1": None,
                "2": None,
                "3": None,
                "4": None,
                "5": None
            }
        }
    Adj = np.ones(numb_drones) - np.identity(numb_drones)

    CBAA_swarm(Adj, new_targets, drones)
