from airsim.types import YawMode
import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import time

vehicle_names = ["Drone1", "Drone2", "Drone3"]

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
for name in vehicle_names:
    client.enableApiControl(True, vehicle_name=name)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
for name in vehicle_names:
    client.armDisarm(True, vehicle_name=name)
    take_off_future = client.takeoffAsync(vehicle_name=name)
    take_off_future.join()
print("All drones have taken off!")

airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
for name in vehicle_names:
    movement = client.moveToPositionAsync(-10, 10, -10, 5, yaw_mode=YawMode(), vehicle_name=name)
    time.sleep(0.1)
    if name == vehicle_names[-1]:
        movement.join()

for name in vehicle_names:
    client.hoverAsync(vehicle_name=name)
    time.sleep(0.1)

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

for name in vehicle_names:
    client.reset()
    client.armDisarm(False, vehicle_name=name)

    # that's enough fun for now. let's quit cleanly 
    client.enableApiControl(False, vehicle_name=name)

print("All vehicles reset!")
