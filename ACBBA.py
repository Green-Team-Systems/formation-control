# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from utils.data_classes import PosVec3
from utils.distance_utils import ned_position_difference

# %%

num_agents: int
num_tasks: int
max_depth: int
time_window_flag: bool
duration_flag: bool
agent_types: list
task_types: list
space_limit_x: list
space_limit_y: list
space_limit_z: list
time_interval_list: list
agent_index_list: list
bunList: list
pathList: list
timeList: list
scores_list: list
bid_list: list
winList: list
winBidlist: list
graph: list
AgentList: list
TaskList: list


def CBBA_swarm(bunList: np.array, adj: np.array, targets: np.array, drones: dict) -> np.array:
    # begin timer
    start = time.time()
    n = len(list(drones.keys()))  # number of agents
    drone_ids = list()
    for name in drones.keys():
        drone_id = int(name.removeprefix('Drone'))
        drone_ids.append(drone_id)
    id = np.identity(n)
    G = np.array([[adj[i][j] + id[i][j] for j in range(len(adj[0]))] for i in range(len(adj))])

    rounds = 10

    for j in range(rounds):
        task_assigned = 0
        print("Round {}".format(j + 1))
        # Phase I: Bundle
        for drone_name in drones.keys():
            print("Hello")
            bidFlag = createBundle(drone_name)
        # Phase 2: Conflict Resolutions
        for n in range(numb_drones):
            for m in range(10):
                if bunList[n][m] == -1:
                    break
                else:
                    bunList[n][m] = TaskList[bunList[n][m]].task_id

                if pathList[n][m] == -1:
                    break
                else:
                    pathList[n][m] = TaskList[pathList[n][m]].task_id

        pathList = [list(filter(lambda a: a != -1,  pathList[i]))
                         for i in range(len( pathList))]

        print("\n")

        if task_assigned == len(targets):
            print("Task assignments complete")
            print("Total number of rounds:", j + 1)
            break  # comment this out once we can replicate
    end = time.time()
    print("Task Allocation took {} seconds".format(end - start))
    print("\n")
    plot_targets(targets, drones)

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

def plot_targets(targets, drones):
    # updating to show paths
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
            if (info["TaskList"][i] == 1):
                print("{} goes to target {}".format(name, i + 1))
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
    # drones.append(PosVec3(X=-50, Y=-100, Z=-1))
    drones.append(PosVec3(X=350, Y=-200, Z=-1))  # changed from 200
    # drones.append(PosVec3(X=200, Y=175, Z=-1))
    drones.append(PosVec3(X=50, Y=75, Z=-1))
    return drones

# %%
# --------------------TEST------------------
if __name__ == "__main__":
    debug = False
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
        # print(drones[drone_name])

        if gen_drones:
            drones[drone_name]['Position'] = new_drones[i]
        else:
            drones[drone_name]['Position'] = new_drones[i]
    # fully_connected Adjacency Matrix
    # Adj = np.ones(numb_drones) - np.identity(numb_drones)
    Adj = []

def createBundle(droneAgent: int):
    remBundle(droneAgent)
    return addBundle(droneAgent)

def addBundle(droneAgent: int):
    bidFlag = False
    value = 1e-5
    indArray = np.where(np.array(bunList[droneAgent]) == -1)[0]
    if len(indArray) > 0:
        fullBundle = False
    else:
        fullBundle = True

    feas = [[1] * (max_depth + 1) for _ in range(num_tasks)]

    while not fullBundle:
        [best_indices, task_times, feas] = generate_bids(droneAgent, feas)
        array1 = ((np.array(bid_list[droneAgent]) - np.array(winBidlist[droneAgent])) > value)
        array2 = (abs(np.array(bid_list[droneAgent]) - np.array(winBidlist[droneAgent])) <= value)
        array3 = (agent_index_list[droneAgent] < np.array(winList[droneAgent]))

        array_logical_result = np.logical_or(array1, np.logical_and(array2, array3))
        maxArray = np.array(bid_list[droneAgent]) * array_logical_result
        bestTask = maxArray.argmax()
        maxValue = max(maxArray)

        if maxValue > 0:
            bidFlag = True
            allValues = np.where(maxArray == maxValue)[0]
            if len(allValues) == 1:
                bestTask = allValues[0]
            else:
                earliest = sys.float_info.max
                for i in range(len(allValues)):
                    if  TaskList[allValues[i]].start_time < earliest:
                        earliest =  TaskList[allValues[i]].start_time
                        bestTask = allValues[i]

            winList[droneAgent][bestTask] =  AgentList[droneAgent].agent_id
            pathList[droneAgent].insert(best_indices[bestTask], bestTask)
            winBidlist[droneAgent][bestTask] = bid_list[droneAgent][bestTask]
            del  pathList[droneAgent][-1]
            timeList[droneAgent].insert(best_indices[bestTask], task_times[bestTask])
            del  timeList[droneAgent][-1]
            scores_list[droneAgent].insert(best_indices[bestTask], bid_list[droneAgent][bestTask])
            del  scores_list[droneAgent][-1]

            length = len(np.where(np.array( bunList[droneAgent]) > -1)[0])
            bunList[droneAgent][length] = bestTask

            for i in range( num_tasks):
                feas[i].insert(best_indices[bestTask], feas[i][best_indices[bestTask]])
                del feas[i][-1]
        else:
            break
        indArray = np.where(np.array( bunList[droneAgent]) == -1)[0]
        if len(indArray) > 0:
            fullBundle = False
        else:
            fullBundle = True

        return bidFlag

def remBundle(droneAgent: int):
    out_bid_for_task = False
    for idx in range(max_depth):
        if  bunList[droneAgent][idx] < 0:
            break
        else:
            if  winList[droneAgent][bunList[droneAgent][idx]] != agent_index_list[droneAgent]:
                out_bid_for_task = True

            if out_bid_for_task:
                if  winList[droneAgent][bunList[droneAgent][idx]] == \
                     agent_index_list[droneAgent]:
                     winList[droneAgent][bunList[droneAgent][idx]] = -1
                     winBidlist[droneAgent][bunList[droneAgent][idx]] = -1
                path_current = copy.deepcopy(pathList[droneAgent])
                idx_remove = path_current.index( bunList[droneAgent][idx])
                del  pathList[droneAgent][idx_remove]
                pathList[droneAgent].append(-1)
                del  timeList[droneAgent][idx_remove]
                timeList[droneAgent].append(-1)
                del  scores_list[droneAgent][idx_remove]
                scores_list[droneAgent].append(-1)
                bunList[droneAgent][idx] = -1

    CBBA_swarm(Adj, new_targets, drones)
# %%
