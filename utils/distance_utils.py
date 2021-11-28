# ================================================================
# Copyright 2021. Codex Laboratories LLC
# Created by: Tyler Fedrizzi
# Created On: 21 March 2021
# Updated On: 
#
# Description: Utilty methods for calculating distance
# ================================================================
import math
import numpy as np

from utils.data_classes import PosVec3


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points of
    latitude and longitude.
    """
    # distance between latitudes 
    # and longitudes 
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
  
    # convert to radians 
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
  
    # apply formulae 
    a = (pow(math.sin(dLat / 2), 2) + 
         pow(math.sin(dLon / 2), 2) * 
             math.cos(lat1) * math.cos(lat2)); 
    rad = 6371 # kilometers
    c = 2 * math.asin(math.sqrt(a)) 
    return rad * c # kilometers


def build_vehicle_distance_matrix_euclidean(positions):
    """
    Build the vehicle distance matrix, taking the Euclidean distance
    between each set of drones.

    Inputs:
    - positions [dict] - dictionary of the positions of the swarm
    """
    distance_matrix = np.zeros((len(positions.keys()), len(positions.keys())), dtype=float)
    for i, row in enumerate(distance_matrix):
        for j, _ in enumerate(row):
            if i != j:
                first_drone = positions["Drone{}".format(i + 1)]["pos_vec3"]
                second_drone = positions["Drone{}".format(j + 1)]["pos_vec3"]
                distance_matrix[i, j] = math.sqrt(
                    (first_drone[0] - second_drone[0])**2 + 
                    (first_drone[1] - second_drone[1])**2 +
                    (first_drone[2] - second_drone[2])**2)
            else:
                distance_matrix[i, j] = 0
    return distance_matrix


def build_vehicle_distance_matrix_gps(positions: dict) -> np.array:
    """
    Build the vehicle distance matrix, taking the Haversine distance
    between each set of drones.

    Inputs:
    - drones [dict] - dictionary of the drones of the swarm
    """
    distance_matrix = np.zeros((len(positions.keys()), len(positions.Keys())), dtype=float)
    for i, row in enumerate(distance_matrix):
        for j, column in enumerate(row):
            if i != j:
                first_drone = positions["Drone{}".format(i + 1)]["pos_vec3"]
                second_drone = positions["Drone{}".format(j + 1)]["pos_vec3"]
                distance_matrix[i, j] = round(haversine(
                    first_drone[0],
                    first_drone[1],
                    second_drone[0],
                    second_drone[1])*1000, 3)
            else:
                distance_matrix[i, j] = 0
    return distance_matrix


def ned_position_difference(first_pos: PosVec3,
                            second_pos: PosVec3) -> float:
    return math.sqrt(pow((first_pos.X - second_pos.X), 2)
                     + pow((first_pos.Y - second_pos.Y), 2)
                     + pow((first_pos.Z - second_pos.Z), 2))