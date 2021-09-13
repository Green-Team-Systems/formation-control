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
    new_points = [[0, 0] for _ in range(len(points[:][0]))]
    for j, row in enumerate(points[:]):
        for i, pt in enumerate(row):
            new_points[i][j] = pt
    for k, new_pt in enumerate(new_points):
        new_points[k] = PosVec3(X=new_pt[0],
                                Y=new_pt[1],
                                Z=end_point.Z)
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


def calculate_v_formation_position(vehicle_name: str,
                                   heading: float) -> PosVec3:
    """
    To position the swarm in a V formation, we need to position
    each drone around a leader drone. This V formation is accomplished
    by taking the position of the leader and calculating an X, Y
    position based upon a specific angle and distance, that defines
    the spread of the formation. For all drones beyond the first level,
    so drones with a ID number greater then 3, we use the location of
    the drone of the immediate preceding level, so drone ID - 2.

    Inputs:
    - vehicle_name [str] ID of the vehicle
    - headings [float] Orientation of vehicles in degrees

    Outputs:
    PosVec3 next location in global coordinates that the vehicle
    should move to.
    """
    leader_pos = copy.deepcopy(vehicles[leader]["global_position"])
    print("\nLeader's Position: {}".format(leader_pos))

    if vehicle_name == leader:
        return leader_pos

    backward_angle = (180 + heading)
    swarm_pos = int(vehicle_name[-1])
    # If the designator of the drone is even
    if swarm_pos % 2 == 0:
        # The drone should be to the "left" of the lead
        # drone.
        angle = backward_angle - formation_angle  # degrees
        # If we are past the first level, the leader position will
        # be the drone that is ID - 2 ahead of the swarm.
        if swarm_pos > 2:
            next_swarm_pos = swarm_pos - 2
            leader_pos = (vehicles["Drone" + str(next_swarm_pos)]
                                  ["global_position"])
    else:
        # The drone should be to the "right" of the lead
        # drone
        angle = backward_angle + formation_angle  # degrees
        # If we are past the first level, the leader position will
        # be the drone that is ID - 2 ahead of the swarm.
        if swarm_pos > 3:
            next_swarm_pos = swarm_pos - 2
            leader_pos = (vehicles["Drone" + str(next_swarm_pos)]
                                  ["global_position"])

    next_pos = PosVec3(frame="global")
    if (heading < 45.0 or heading > 315.0):
        if angle < backward_angle:
            next_pos.X = leader_pos.X - (formation_spread
                                         * np.cos(
                                             np.radians(formation_angle)))
            next_pos.Y = leader_pos.Y - (formation_spread
                                         * np.sin(
                                             np.radians(formation_angle)))
            next_pos.Z = leader_pos.Z
        elif angle > backward_angle:
            next_pos.X = leader_pos.X - (formation_spread
                                         * np.cos(
                                             np.radians(formation_angle)))
            next_pos.Y = leader_pos.Y + (formation_spread
                                         * np.sin(
                                             np.radians(formation_angle)))
            next_pos.Z = leader_pos.Z
    elif (heading >= 45.0 and heading < 135.0):
        if angle < backward_angle:
            next_pos.X = leader_pos.X + (formation_spread
                                         * np.cos(
                                             np.radians(formation_angle)))
            next_pos.Y = leader_pos.Y - (formation_spread
                                         * np.sin(
                                             np.radians(formation_angle)))
            next_pos.Z = leader_pos.Z
        elif angle > backward_angle:
            next_pos.X = leader_pos.X - (formation_spread
                                         * np.cos(
                                             np.radians(formation_angle)))
            next_pos.Y = leader_pos.Y - (formation_spread
                                         * np.sin(
                                             np.radians(formation_angle)))
            next_pos.Z = leader_pos.Z

    vehicles[name]["global_position"] = next_pos

    return next_pos


def calculate_star_formation(vehicle_name: str,
                                   heading: float) -> PosVec3:
    '''


    '''
    leader_pos = copy.deepcopy(vehicles[leader]["global_position"])
    print("\nLeader's Position: {}".format(leader_pos))

    if vehicle_name == leader:
        return leader_pos
    
    equi_angle = 360.0 / num_vehicles 
    swarm_pos = int(vehicle_name[-1])

    

    next_pos = PosVec3(frame="global")
    
    
    



    return next_pos


def fly_to_new_position(vehicle_name: str,
                        position: PosVec3,
                        speed: float,
                        yaw_angle: float,
                        drive_train_type: str = "free"):
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
    position.X = position.X + start_pos.X - start_offset.X
    # We have offsets in the Y. We want to reverse those offsets when
    # we move in the right direction.
    id_number = int(vehicle_name[-1])

    if id_number > 3:
        factor = 2
    else:
        factor = 1

    if (yaw_angle > 0.0):
        position.Y = (position.Y
                      - (start_pos.Y / factor)
                      - start_offset.Y)
    else:
        pass
        # Otherwise, we act as normal with our Y offsets.
        # position.Y = (position.Y
        #              + start_pos.Y
        #              - start_offset.Y)
    if (id_number > 3):
        position.Z = position.Z + 1 + start_pos.Z
    else:
        position.Z = position.Z + start_pos.Z
    print("{} flyting to {}".format(vehicle_name, position))
    if drive_train_type == "free":
        drive_train = DrivetrainType.MaxDegreeOfFreedom
    elif drive_train_type == "forward":
        drive_train = DrivetrainType.ForwardOnly,
    move_future = client.moveToPositionAsync(position.X,
                                             position.Y,
                                             position.Z,
                                             speed,
                                             drivetrain=drive_train,
                                             yaw_mode=airsim.YawMode(
                                                 False, yaw_angle),
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
        X=position.X + start_pos.X,
        Y=position.Y + start_pos.Y,
        Z=start_pos.Z - start_pos.Z,
    )

# ======================= Variable Initialization =============================


leader = "Drone1"
formation_spread = 2.0
formation_angle = 45.0
# Determines if the drones will execute the mission or wait for your commands
wait = False
# TODO Make this a command line input.
# The bridge is defined as to the East (Y-direction)
bridge_position = PosVec3(X=50.0, Y=140.0, Z=25.0, frame="global")
start_offset = PosVec3(X=5, Y=0, Z=0, frame="global")

# Grab settings from local settings file
try:
    # TODO Make this a relative path to the repo
    with open("/home/rptamin/Documents/AirSim/settings.json", "r") as f:
        settings = json.load(f)
except Exception:
    print("Please change the path to the JSON settings file!")
    exit

# TODO Conver to SWARM Drone class
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
    "Drone3": {
    "starting_position": None,
    "local_position": None,
    "global_position": None,
    "trajectory": None,
    "orientation": None},
    "Drone4": {
    "starting_position": None,
    "local_position": None,
    "global_position": None,
    "trajectory": None,
    "orientation": None},
    "Drone5": {
    "starting_position": None,
    "local_position": None,
    "global_position": None,
    "trajectory": None,
    "orientation": None}, }

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

num_vehicles = len(vehicles)
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
if wait:
    airsim.wait_key('\nPress any key to push drones to a formation!')
else:
    time.sleep(1)

# ======================= Initial Formation =================================
# Push the drones into some formation

heading = 0.0
for name in vehicles.keys():
    # new_pos = PosVec3(X=0.0, Y=0.0, Z=0.0, frame="global")
    next_pos = calculate_v_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, 0.0)
    time.sleep(0.1)
    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)

if wait:
    airsim.wait_key("\nPress any key to fly to target!")
else:
    time.sleep(2)


# ======================= Fly to Target =====================================
# Move to some new location
new_pos = PosVec3(X=5.0, Y=10.0, Z=2.0, frame="global")
heading = 90.0  # degrees
vehicles[leader]["global_position"] = new_pos

for name in vehicles.keys():
    next_pos = calculate_v_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, heading)
    # movement.join()

    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)

if wait:
    airsim.wait_key("\nPress any key for Waypoint 1!")
else:
    time.sleep(1)

new_pos = PosVec3(X=10.0, Y=40.0, Z=5.0, frame="global")
heading = 80.0  # degrees
vehicles[leader]["global_position"] = new_pos

for name in vehicles.keys():
    next_pos = calculate_v_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, heading)
    # movement.join()

    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)


if wait:
    airsim.wait_key("\nPress any key to fly to continue to Waypoint 2!")
else:
    time.sleep(1)
new_pos = PosVec3(X=20.0, Y=100.0, Z=14.0, frame="global")
heading = 70.0  # degrees
vehicles[leader]["global_position"] = new_pos

for name in vehicles.keys():
    next_pos = calculate_v_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, heading)
    # movement.join()

    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)


if wait:
    airsim.wait_key("\nPress any key to continue to Waypoint 3!")
else:
    time.sleep(1)
new_pos = PosVec3(X=50.0, Y=140.0, Z=25.0, frame="global")
heading = 45.0  # degrees
vehicles[leader]["global_position"] = new_pos

for name in vehicles.keys():
    next_pos = calculate_v_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, heading)
    # movement.join()

    if name == list(vehicles.keys())[-1]:
        movement.join()

for name in vehicles.keys():
    update_positions(name)

# ======================= Surround Object ===================================
"""
for name in vehicles.keys():
    next_pos = calculate_guard_formation_position(name, heading)
    print("Drone {} Next Pos: {}".format(name, next_pos))
    movement = fly_to_new_position(name, next_pos, 5.0, heading)
    # movement.join()

    if name == list(vehicles.keys())[-1]:
        movement.join()
"""
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
    state = client.getMultirotorState(vehicle_name=name)
    print("Actual Position: {X}, {Y}, {Z}".format(
        X=state.kinematics_estimated.position.x_val,
        Y=state.kinematics_estimated.position.y_val,
        Z=state.kinematics_estimated.position.z_val))

airsim.wait_key('Press any key to reset to original state')

for name in vehicles.keys():
    client.reset()
    client.armDisarm(False, vehicle_name=name)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False, vehicle_name=name)

print("All vehicles reset!")
