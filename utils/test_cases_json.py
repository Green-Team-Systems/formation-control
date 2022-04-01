# ================================================================
# Created by: Jack Roy
# Created On: Febuary 2022
# Updated On: Febuary 2022
#
# Description: Utility function for testing Task Allocation Algorithms
# ================================================================
import math
import numpy as np
import random

from utils.data_classes import PosVec3

def test_cases(case_num, rand_num=6):
    '''
    Full set of N test cases for Task Allocation Algorithm
    calling test_cases(n) will yield the drone and position locations for that case
    Number of Drones is 6 unless noted otherwise
    //expand to make json files?
    Case List:
     1. (easy) Each drone is the same distance from exactly 1 target
     2. (easy) Each drone is closest to one target but different distances
     3. (easy) Drones move from cluster to spread out
     4. (easy) Drones move from spread out to cluster (inverse of 3)
     5. (med)  Drones move from cluster to line (back of cluster is not optimal for any 1 task)
     6. (med)  Drones "pair up" one moves a little other comes near
     7. (med)  Drone 1 does not move, other drones come around it
     8. (hard) Test case from earlier, not sure how to describe it
     9. (hard) Drones and targets have a lot of equivalent distances
    10. (hard) rand_num random drones
    NOTE: random generation does not check for overlaps
    '''
    #no switch statements in python :(
    #set random seed for testing
    #random.seed(1)
    if(case_num == 1):
        targets = case1targets()
        drones = case1drones()
    elif(case_num == 2):
        targets = case2targets()
        drones = case2drones()
    elif(case_num == 3):
        targets = case34targets()
        drones = case34drones()
    elif(case_num == 4):
        targets = case34drones()
        drones = case34targets()
    elif(case_num == 5):
        targets = case5targets()
        drones = case5drones()
    elif(case_num == 6):
        targets = case6targets()
        drones = case6drones()
    elif(case_num == 7):
        targets = case7targets()
        drones = case7drones()
    elif(case_num == 8):
        targets = case8targets()
        drones = case8drones()
    elif(case_num == 9):
        targets = case9targets()
        drones = case9drones()
    else:
        targets = caseRandom(rand_num)
        drones = caseRandom(rand_num)
    return targets, drones

#case 1
def case1targets():
    targets = list()
    targets.append(PosVec3(X=-250, Y=100, Z=-1))
    targets.append(PosVec3(X=-150, Y=100, Z=-1))
    targets.append(PosVec3(X=-50, Y=100, Z=-1))
    targets.append(PosVec3(X=50, Y=100, Z=-1))
    targets.append(PosVec3(X=150, Y=100, Z=-1))
    targets.append(PosVec3(X=250, Y=100, Z=-1))
    return targets
def case1drones():
    drones = list()
    drones.append(PosVec3(X=-250, Y=-100, Z=-1))
    drones.append(PosVec3(X=-150, Y=-100, Z=-1))
    drones.append(PosVec3(X=-50, Y=-100, Z=-1))
    drones.append(PosVec3(X=50, Y=-100, Z=-1))
    drones.append(PosVec3(X=150, Y=-100, Z=-1))
    drones.append(PosVec3(X=250, Y=-100, Z=-1))
    return drones

#case 2
def case2targets():
    targets = list()
    targets.append(PosVec3(X=-300, Y=150, Z=-1))
    targets.append(PosVec3(X=-300, Y=0, Z=-1))
    targets.append(PosVec3(X=-300, Y=-150, Z=-1))
    targets.append(PosVec3(X=300, Y=150, Z=-1))
    targets.append(PosVec3(X=300, Y=0, Z=-1))
    targets.append(PosVec3(X=300, Y=-150, Z=-1))
    return targets
def case2drones():
    drones = list()
    drones.append(PosVec3(X=-200, Y=0, Z=-1))
    drones.append(PosVec3(X=-75, Y=100, Z=-1))
    drones.append(PosVec3(X=75, Y=100, Z=-1))
    drones.append(PosVec3(X=200, Y=0, Z=-1))
    drones.append(PosVec3(X=75, Y=-100, Z=-1))
    drones.append(PosVec3(X=-75, Y=-100, Z=-1))
    return drones

#cases 3 and 4
def case34targets():
    targets = list()
    targets.append(PosVec3(X=-300, Y=100, Z=-1))
    targets.append(PosVec3(X=0, Y=250, Z=-1))
    targets.append(PosVec3(X=300, Y=100, Z=-1))
    targets.append(PosVec3(X=300, Y=-100, Z=-1))
    targets.append(PosVec3(X=0, Y=-250, Z=-1))
    targets.append(PosVec3(X=-300, Y=-100, Z=-1))
    return targets
def case34drones():
    drones = list()
    drones.append(PosVec3(X=-50, Y=-25, Z=-1))
    drones.append(PosVec3(X=0, Y=-25, Z=-1))
    drones.append(PosVec3(X=50, Y=-25, Z=-1))
    drones.append(PosVec3(X=-50, Y=25, Z=-1))
    drones.append(PosVec3(X=0, Y=25, Z=-1))
    drones.append(PosVec3(X=50, Y=25, Z=-1))
    return drones

#case 5
def case5targets():
    targets = list()
    targets.append(PosVec3(X=-250, Y=100, Z=-1))
    targets.append(PosVec3(X=-150, Y=100, Z=-1))
    targets.append(PosVec3(X=-50, Y=100, Z=-1))
    targets.append(PosVec3(X=50, Y=100, Z=-1))
    targets.append(PosVec3(X=150, Y=100, Z=-1))
    targets.append(PosVec3(X=250, Y=100, Z=-1))
    return targets
def case5drones():
    drones = list()
    drones.append(PosVec3(X=-50, Y=-25, Z=-1))
    drones.append(PosVec3(X=0, Y=-25, Z=-1))
    drones.append(PosVec3(X=50, Y=-25, Z=-1))
    drones.append(PosVec3(X=-50, Y=25, Z=-1))
    drones.append(PosVec3(X=0, Y=25, Z=-1))
    drones.append(PosVec3(X=50, Y=25, Z=-1))
    return drones

#case 6
def case6targets():
    targets = list()
    targets.append(PosVec3(X=0, Y=150, Z=-1))
    targets.append(PosVec3(X=-25, Y=150, Z=-1))
    targets.append(PosVec3(X=225, Y=-125, Z=-1))
    targets.append(PosVec3(X=200, Y=-125, Z=-1))
    targets.append(PosVec3(X=-300, Y=-50, Z=-1))
    targets.append(PosVec3(X=-275, Y=-50, Z=-1))
    return targets
def case6drones():
    drones = list()
    drones.append(PosVec3(X=0, Y=200, Z=-1))
    drones.append(PosVec3(X=50, Y=150, Z=-1))
    drones.append(PosVec3(X=250, Y=-125, Z=-1))
    drones.append(PosVec3(X=-75, Y=0, Z=-1))
    drones.append(PosVec3(X=-300, Y=-100, Z=-1))
    drones.append(PosVec3(X=-100, Y=25, Z=-1))
    return drones

#case 7
def case7targets():
    targets = list()
    targets.append(PosVec3(X=-0, Y=0, Z=-1))
    targets.append(PosVec3(X=0, Y=50, Z=-1))
    targets.append(PosVec3(X=50, Y=25, Z=-1))
    targets.append(PosVec3(X=25, Y=-25, Z=-1))
    targets.append(PosVec3(X=-25, Y=-25, Z=-1))
    targets.append(PosVec3(X=-50, Y=25, Z=-1))
    return targets
def case7drones():
    drones = list()
    drones.append(PosVec3(X=0, Y=0, Z=-1))
    drones.append(PosVec3(X=325, Y=100, Z=-1))
    drones.append(PosVec3(X=-325, Y=100, Z=-1))
    drones.append(PosVec3(X=250, Y=0, Z=-1))
    drones.append(PosVec3(X=-100, Y=-200, Z=-1))
    drones.append(PosVec3(X=--75, Y=-100, Z=-1))
    return drones

#case 8
def case8targets():
    targets = list()
    targets.append(PosVec3(X=100, Y=-250, Z=-1))
    targets.append(PosVec3(X=-25, Y=225, Z=-1))
    targets.append(PosVec3(X=375, Y=200, Z=-1))
    targets.append(PosVec3(X=125, Y=-75, Z=-1))
    targets.append(PosVec3(X=-150, Y=-75, Z=-1))
    targets.append(PosVec3(X=325, Y=-300, Z=-1))
    return targets
def case8drones():
    drones = list()
    drones.append(PosVec3(X=-150, Y=375, Z=-1))
    drones.append(PosVec3(X=225, Y=-325, Z=-1))
    drones.append(PosVec3(X=-50, Y=-100, Z=-1))
    drones.append(PosVec3(X=350, Y=-200, Z=-1))
    drones.append(PosVec3(X=200, Y=175, Z=-1))
    drones.append(PosVec3(X=50, Y=75, Z=-1))
    return drones

#case 9
def case9targets():
    targets = list()
    targets.append(PosVec3(X=0, Y=300, Z=-1))
    targets.append(PosVec3(X=0, Y=200, Z=-1))
    targets.append(PosVec3(X=0, Y=0, Z=-1))
    targets.append(PosVec3(X=0, Y=-300, Z=-1))
    targets.append(PosVec3(X=300, Y=200, Z=-1))
    targets.append(PosVec3(X=300, Y=-50, Z=-1))
    return targets
def case9drones():
    drones = list()
    drones.append(PosVec3(X=-200, Y=-100, Z=-1))
    drones.append(PosVec3(X=-200, Y=0, Z=-1))
    drones.append(PosVec3(X=-100, Y=100, Z=-1))
    drones.append(PosVec3(X=100, Y=100, Z=-1))
    drones.append(PosVec3(X=200, Y=0, Z=-1))
    drones.append(PosVec3(X=200, Y=-100, Z=-1))
    return drones

#random cases
def caseRandom(num):
    targets = list()
    for _ in range(num):
        targets.append(
            PosVec3(
                X=random.randrange(-800, 800, 25),
                Y=random.randrange(-800, 800, 25),
                Z=-1
            )
        )
    return targets
