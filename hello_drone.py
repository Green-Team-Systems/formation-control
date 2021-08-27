import numpy as np
import json
import os
import tempfile
import pprint
import time
import copy

import setup_path
import airsim

from utils.dataclasses import PosVec3, Quaternion
from airsim.types import YawMode


def update_positions(vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    vehicles[vehicle_name]["local_position"] = PosVec3(
        X=state.kinematics_estimated.position.x_val,
        Y=state.kinematics_estimated.position.y_val,
        Z=state.kinematics_estimated.position.z_val,
        frame="local",
        starting=False
    )
    start_position = vehicles[name]["starting_position"]
    global_position = vehicles[name]["global_position"]
    local_position = vehicles[name]["local_position"]
    global_position.X = local_position.X - start_position.X
    global_position.Y = local_position.Y - start_position.Y
    global_position.Z = local_position.Z - start_position.Z


def calculate_v_formation_position(vehicle_name) -> PosVec3:
    leader_pos = copy.deepcopy(vehicles[leader]["global_position"])
    print("\nLeader's Position: {}".format(leader_pos))

    if vehicle_name == leader:
        return leader_pos
    
    swarm_pos = int(vehicle_name[-1])
    # If the designator of the drone is even
    if swarm_pos % 2 == 0:
        # The drone should be to the "left" of the lead
        # drone.
        angle = 180 - formation_angle # degrees
        # If we are past the first level, the leader position will
        # be the drone that is ID - 2 ahead of the swarm.
        if swarm_pos > 2:
            next_swarm_pos = swarm_pos - 2
            leader_pos = vehicles["Drone" + str(next_swarm_pos)]["global_position"]
    else:
        # The drone should be to the "right" of the lead
        # drone
        angle = 180 + formation_angle # degrees
        # If we are past the first level, the leader position will
        # be the drone that is ID - 2 ahead of the swarm.
        if swarm_pos > 3:
            next_swarm_pos = swarm_pos - 2
            leader_pos = vehicles["Drone" + str(next_swarm_pos)]["global_position"]

    next_pos = PosVec3(frame="global")
    if angle < 180:
        next_pos.X = leader_pos.X - (formation_spread * np.cos(np.radians(180 - angle)))
        next_pos.Y = leader_pos.Y - (formation_spread * np.sin(np.radians(180 - angle)))
        next_pos.Z = leader_pos.Z
    elif angle > 180:
        next_pos.X = leader_pos.X - (formation_spread * np.cos(np.radians(angle - 180)))
        next_pos.Y = leader_pos.Y + (formation_spread * np.sin(np.radians(angle - 180)))
        next_pos.Z = leader_pos.Z
    
    return next_pos


def fly_to_new_position(vehicle_name: str, position: PosVec3, speed: float):
    start_pos = vehicles[vehicle_name]["starting_position"]
    position.X = position.X + start_pos.X
    position.Y = position.Y + start_pos.Y
    position.Z = position.Z + start_pos.Z
    move_future = client.moveToPositionAsync(position.X,
                                             position.Y,
                                             position.Z,
                                             speed,
                                             yaw_mode=YawMode(),
                                             vehicle_name=vehicle_name)
    # update_new_local_position(vehicle_name, position)
    return move_future


def update_new_local_position(vehicle_name: str, position: PosVec3) -> None:
    start_pos = vehicles[vehicle_name]["starting_position"]
    vehicles[vehicle_name]["local_position"] = PosVec3(
        X = position.X + start_pos.X,
        Y = position.Y + start_pos.Y,
        Z = start_pos.Z - start_pos.Z,
    )


leader = "Drone1"
formation_spread = 2.0
formation_angle = 30

# Grab settings from local settings file
try:
    # TODO Make this a relative path to the repo
    with open("/home/codexlabs/Documents/AirSim/settings.json", "r") as f:
        settings = json.load(f)
except Exception:
    print("Please change the path to the JSON settings file!")
    exit

vehicles = {"Drone1": {
                "starting_position": None,
                "local_position": None,
                "global_position": None,
                "orientation": None,
                "leader": True},
            "Drone2": {
                "starting_position": None,
                "local_position": None,
                "global_position": None,
                "orientation": None},
            "Drone3":{
                "starting_position": None,
                "local_position": None,
                "global_position": None,
                "orientation": None}}

# Initialize positions and orientations of swarm from settings file
settings_vehicles = settings["Vehicles"]
for name in vehicles.keys():
    vehicles[name]["starting_position"] = PosVec3(
        X=settings_vehicles[name]["X"],
        Y=settings_vehicles[name]["Y"],
        Z=settings_vehicles[name]["Z"],
        starting=True,
        frame="global")
    vehicles[name]["local_position"] = PosVec3()
    vehicles[name]["global_position"] = PosVec3(
        X=settings_vehicles[name]["X"],
        Y=settings_vehicles[name]["Y"],
        Z=settings_vehicles[name]["Z"],
        starting=False,
        frame="global")
    vehicles[name]["orientation"] = Quaternion()

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
for name in vehicles.keys():
    client.enableApiControl(True, vehicle_name=name)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
for name in vehicles.keys():
    client.armDisarm(True, vehicle_name=name)
    take_off_future = client.takeoffAsync(vehicle_name=name)
    take_off_future.join()
print("All drones have taken off!")

airsim.wait_key('Press any key to push drones to a formation!')

# ======================= Initial Formation =================================
# Push the drones into some formation

for name in vehicles.keys():
    # new_pos = PosVec3(X=0.0, Y=0.0, Z=0.0, frame="global")
    next_pos = calculate_v_formation_position(name)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0)
    time.sleep(0.1)
    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)

airsim.wait_key("Press any key to fly to target!")

# ======================= Fly to Target =====================================
# Move to some new location
new_pos = PosVec3(X=5.0, Y=0.0, Z=-1.0, frame="global")
vehicles[leader]["global_position"] = new_pos

for name in vehicles.keys():
    next_pos = calculate_v_formation_position(name)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0)
    time.sleep(0.1)
    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)

# ======================= Surround Object ===================================
airsim.wait_key("Press any key to hover!")
for name in vehicles.keys():
    client.hoverAsync(vehicle_name=name)
    update_positions(name)
    time.sleep(0.1)

for name in vehicles.keys():
    print("\nName: {}\nLocal Position: {}".format(
        name, vehicles[name]["local_position"]))
    print("Global Position: {}\n".format(
        vehicles[name]["global_position"]))

airsim.wait_key('Press any key to reset to original state')

for name in vehicles.keys():
    client.reset()
    client.armDisarm(False, vehicle_name=name)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False, vehicle_name=name)

print("All vehicles reset!")
