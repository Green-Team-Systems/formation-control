import numpy as np
import json
import os
import tempfile
import pprint
import time
import copy

import setup_path
import airsim
import bezier

from utils.dataclasses import PosVec3, Quaternion
from airsim.types import DrivetrainType, YawMode


def calculate_moveable_space(start_point: PosVec3,
                             end_point: PosVec3) -> PosVec3:
    """
    When calculating Bezier curves, we must define N + 1
    points, which defines a N + 1 dimensional space that
    a Bezier curve can be detected on. To do this, we
    initialize the boundary point of the moveable space
    as the point that makes a right triangle in the defined
    action space.

      x_3 - - - - - x_2
       |        .
       |   .
      x_1
    
    Given x_1 and x_2, we define the Bezier curve in the
    surface (right triangle), as defined by the boundary
    points. This Bezier curve is contained on the surface.

    For a Bezier curve of degree N, we need N + 1 points,
    of which the dimension should also be N.

    Inputs:
    - start_point [PosVec3] Starting point of the Bezier Curve
    - end_point [PosVec3] End point of the Bezier Curve

    Outputs:
    PosVec3 representing the boundary point of the ambient space of
    the Bezier curve.
    """
    point = PosVec3(frame="global")
    point.X = start_point.X + end_point.X
    point.Y = start_point.Y
    point.Z = start_point.Z
    return point


def calculate_corner_points(start_point: PosVec3,
                            end_point: PosVec3) -> list:
    """
    Given a start and end point, calculate the Bezier curve to move
    along. Once the Bezier urve is found, evaluate the beginning,
    middle and end of the curve as waypoints to move along.

    Inputs:
    - start_point [PosVec3] Starting point of the Bezier Curve
    - end_point [PosVec3] End point of the Bezier Curve

    Output:
    A list of PosVec3 points, defining the points along the curve
    to move upon.
    """
    boundary_point = calculate_moveable_space(start_point, end_point)
    point_array = np.asfortranarray([
        [start_point.X, boundary_point.X, end_point.X],
        [start_point.Y, boundary_point.Y, end_point.Y]
                            ])
    # Calculate a simple polynomial curve
    curve = bezier.Curve(point_array, degree=2)
    # Evaluate the Bezier curve at the start, middle and end point
    eval_points = np.array([0.0, 0.5, 1.0])
    points = curve.evaluate_multi(eval_points)
    # Points are given as a 2 x len(points) matrix. To determine
    # the number of points, we access the first row (which contains) the X
    # values of the evaluated points.
    new_points = [[0,0] for _ in range(len(points[:][0]))]
    for j, row in enumerate(points[:]):
        for i, pt in enumerate(row):
            new_points[i][j] = pt
    for k, new_pt in enumerate(new_points):
        new_points[k] = PosVec3(X=new_pt[0],
                                Y=new_pt[1],
                                Z= end_point.Z)
    return new_points


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
    """
    To position the swarm in a V formation, we need to position
    each drone around a leader drone. This V formation is accomplished
    by taking the position of the leader and calculating an X, Y
    position based upon a specific angle and distance, that defines
    the spread of the formation. For all drones beyond the first level,
    so drones with a ID number greater then 3, we use the location of
    the drone of the immediate preceding level, so drone ID - 2.
    """
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
            leader_pos = (vehicles["Drone" + str(next_swarm_pos)]
                                  ["global_position"])
    else:
        # The drone should be to the "right" of the lead
        # drone
        angle = 180 + formation_angle # degrees
        # If we are past the first level, the leader position will
        # be the drone that is ID - 2 ahead of the swarm.
        if swarm_pos > 3:
            next_swarm_pos = swarm_pos - 2
            leader_pos = (vehicles["Drone" + str(next_swarm_pos)]
                                  ["global_position"])

    next_pos = PosVec3(frame="global")
    if angle < 180:
        next_pos.X = leader_pos.X - (formation_spread
                                    * np.cos(np.radians(180 - angle)))
        next_pos.Y = leader_pos.Y - (formation_spread
                                    * np.sin(np.radians(180 - angle)))
        next_pos.Z = leader_pos.Z
    elif angle > 180:
        next_pos.X = leader_pos.X - (formation_spread
                                    * np.cos(np.radians(angle - 180)))
        next_pos.Y = leader_pos.Y + (formation_spread
                                    * np.sin(np.radians(angle - 180)))
        next_pos.Z = leader_pos.Z
    
    return next_pos


def fly_to_new_position(vehicle_name: str,
                        position: PosVec3,
                        speed: float):
    """
    Given the next position of a vehicle, convert the global
    position given to the local position of the drone using
    the starting coordinates, and then call the approriate API call
    to AirSim. For the API call, the forward only drivetrain forces
    the vehicle to always point in the direction of travel.

    *Note*: The Z value is negative, which corresponds to the coord.
    frame of Unreal Engine. Any positive Z value would be going down,
    so ensure that the Z value is negative if you want to go up.

    Inputs:
    - vehicle_name [str] ID of the vehicle
    - position [PosVec3] X, Y, Z location to move to in global reference
                         frame.
    - speed [float] speed to travel in meters / second

    Outputs:
    Returns a future, which resolves once the action has been completed.
    For our purposes, we only wait for the last drone to complete
    movement to ensure the swarm moves in that direction.
    """
    start_pos = vehicles[vehicle_name]["starting_position"]
    position.X = position.X + start_pos.X
    position.Y = position.Y + start_pos.Y
    position.Z = position.Z + start_pos.Z
    move_future = client.moveToPositionAsync(position.X,
                                             position.Y,
                                             position.Z,
                                             speed,
                                             drivetrain=DrivetrainType.ForwardOnly,
                                             vehicle_name=vehicle_name)
    return move_future


def update_new_local_position(vehicle_name: str, position: PosVec3) -> None:
    """
    Given a global position, update the local position of the drone.
    For AirSim, the API that determines position is given in relative
    coordinates to that drone. If initializing in other points other
    then the origin, the actual commanded position will not be the
    true commanded position. Therefore, we need to ensure that the
    local position in the local reference frame is accurate.

    Inputs:
    - vehicle_name [str] Name of the Vehicle
    - position [PosVec3] Current position of the vehicle in the global
                         refernce frame.
    """
    start_pos = vehicles[vehicle_name]["starting_position"]
    vehicles[vehicle_name]["local_position"] = PosVec3(
        X = position.X + start_pos.X,
        Y = position.Y + start_pos.Y,
        Z = start_pos.Z - start_pos.Z,
    )


leader = "Drone1"
formation_spread = 2.0
formation_angle = 60

# The bridge is defined as to the East (Y-direction)
bridge_position = PosVec3(X=0, Y=50, Z=20, frame="global")

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
                "trajectory": None,
                "leader": True},
            "Drone2": {
                "starting_position": None,
                "local_position": None,
                "global_position": None,
                "trajectory": None,
                "orientation": None},
            "Drone3":{
                "starting_position": None,
                "local_position": None,
                "global_position": None,
                "trajectory": None,
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
