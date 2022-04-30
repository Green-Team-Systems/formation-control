# %%
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from utils.data_classes import PosVec3
from utils.distance_utils import ned_position_difference

# %%

def ACBBA_swarm(adj: np.array, targets: np.array, drones: dict) -> np.array:
    # begin timer
    start = time.time()
    n = len(list(drones.keys()))  # number of agents
    drone_ids = list()
    redrones = list()
    for name in drones.keys():
        drone_id = int(name.removeprefix('Drone'))
        drone_ids.append(drone_id)
    id = np.identity(n)
    G = np.array([[adj[i][j] + id[i][j] for j in range(len(adj[0]))] for i in range(len(adj))])
    rounds = 10

    for j in range(rounds):
        task_assigned = 0
        # Phase I: Bid
        for drone_name in drones.keys():
            generate_bids(drone_name, drones[drone_name], targets, j)
        # Phase 2: Determine Taskings
        for i, drone_name in enumerate(drones.keys()):
            transmit_bids(drone_name, drones, drone_ids)
        for i, drone_name in enumerate(drones.keys()):
            determine_task_list(drone_name, drones[drone_name])
        for name, info in drones.items():
            task_assigned += sum(info["TaskList"])

    end = time.time()
    plot_targets(targets, drones)

def generate_bids(drone_name: str, drone_info: dict, targets: list, rnd: int):
    position = drone_info["Position"]
    drone_info["Bids"] = [0 for _ in range(len(targets))]
    pos_diffs = list()
    pos_diff = 0
    while np.sum(drone_info["TaskList"]) != 1:
        availability = drone_info["TaskAvailability"]
        for i, target in enumerate(targets): #For loop to go through every single drone
            for t in targets: #For loop to go through every target for each drone
                pos_diff = ned_position_difference(t, drone_info["Position"]) + pos_diff #Sum of the dist between drone to each target
                drone_info["RawBids"][i] = (1 / pos_diff) #The bid that suggests how optimal the travel cost is
                bid = drone_info["RawBids"][i]

            pos_diffs.append(pos_diff)

            if bid > (drone_info["WinningBids"][i]):
                availability[i] = 1
                drone_info["Bids"][i] = bid
            else:
                availability[i] = 0

        if sum(availability) != 0:
            task_selected = np.argmax([drone_info["Bids"][i] * availability[i] for i in range(len(targets))])
            drone_info["WinningBids"][task_selected] = drone_info["Bids"][task_selected]
            drone_info["SwarmBids"][drone_name.lstrip("Drone")] = copy.deepcopy(drone_info["WinningBids"])

            drone_info["TaskList"][task_selected] = 1
            drone_info["Position"] = targets[i] #Change drone location to bid winning target

def transmit_bids(drone_name: str, drones: dict, drone_ids: list):
    bid_list = drones[drone_name]["WinningBids"]
    drone_id = int(drone_name.lstrip("Drone"))
    for swarm_id in drone_ids:
        if swarm_id != drone_id:
            swarm_name = "Drone{}".format(swarm_id)
            drones[swarm_name]["SwarmBids"][str(drone_id)] = copy.deepcopy(bid_list)

def determine_task_list(drone_name: str, drone_info: dict):
    drone_id = int(drone_name.lstrip("Drone")) - 1
    swarm_bids = drone_info["SwarmBids"]
    task_list = drone_info["TaskList"]
    bid_columns = [list() for _ in range(len(swarm_bids.keys()))]
    selections = [None for _ in range(len(swarm_bids.keys()))]
    for id, bid_list in swarm_bids.items():
        for i, bid in enumerate(bid_list):
            bid_columns[i].append(bid)

    for j, col in enumerate(bid_columns):
        best_bid = np.max(col)
        if drone_info["Winners"][j] and best_bid == drone_info["Winners"][j]["bid"]:
            continue
        selections[j] = np.argmax(col)
        drone_info["Winners"][j] = dict(id=selections[j], bid=col[selections[j]])
        drone_info["WinningBids"][j] = col[selections[j]]
    for j, winner in enumerate(drone_info["Winners"]):
        winner_id = winner["id"]
        if winner_id != drone_id:
            task_list[j] = 0

def plot_targets(targets, drones):
    # updating to show paths
    redrones = generate_equal_drones(numb_drones) #Create a duplicate list with original coordinates of drones
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
        drone_info['Position'] = redrones[i] #Change position of drone back to its original position for plotting
        labels.append(drone_name)
        drone_pos_y.append(drone_info["Position"].Y)
        drone_pos_x.append(drone_info["Position"].X)
    plt.scatter(drone_pos_x, drone_pos_y)

    for (label, x, y) in zip(labels, drone_pos_x, drone_pos_y):
        plt.text(x=x, y=y, s=label)

    for name, info in drones.items():
        for i in range(len(info["TaskList"])):
            if (info["TaskList"][i] == 1):
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
    # targets.append(PosVec3(X=375, Y=200, Z=-1))
    targets.append(PosVec3(X=125, Y=-75, Z=-1))
    # targets.append(PosVec3(X=-150, Y=-75, Z=-1))
    targets.append(PosVec3(X=325, Y=-300, Z=-1))
    return targets

def generate_equal_drones(num):
    drones = list()
    drones.append(PosVec3(X=-150, Y=375, Z=-1))
    drones.append(PosVec3(X=225, Y=-325, Z=-1))
    drones.append(PosVec3(X=350, Y=-200, Z=-1)) 
    drones.append(PosVec3(X=50, Y=75, Z=-1))
    return drones

# %%
# --------------------TEST------------------
if __name__ == "__main__":
    gen_drones = False  # changed to read from pre-written test scenario, make sure numb_drones is 6
    gen_targets = False
    map_boundary = {"+X": 400,
                    "-X": -400,
                    "+Y": 400,
                    "-Y": -400}  # Meters
    min_separation_distance = 25  # meters
    numb_drones = 4  # changed to make faster
    numb_targets = numb_drones
    if gen_drones:
        new_drones = generate_random_targets(numb_drones)
    else:
        new_drones = generate_equal_drones(numb_drones)
    if gen_targets:
        new_targets = generate_random_targets(numb_targets)
    else:
        new_targets = generate_equal_targets(numb_targets)

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
                # "5": None
            }
        }

        if gen_drones:
            drones[drone_name]['Position'] = new_drones[i]
        else:
            drones[drone_name]['Position'] = new_drones[i]

    Adj = []

    ACBBA_swarm(Adj, new_targets, drones)
# %%
