#!/usr/bin/env python
# coding: utf-8
import copy
import sys
import time
import pybullet_data
import math
import pybullet as p
import numpy as np
import json
import zmq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import threading
from matplotlib.animation import FuncAnimation


def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

def get_rotation_chain(joint, hierarchy, frame_rotations):

    hierarchy = hierarchy[::-1]

    #this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = get_R_z(angles[0])@get_R_x(angles[1])@get_R_y(angles[2])
        R = R @ _R

    return R

"""

def draw_skeleton_from_joint_angles(kpts, bones, ax):

    #print("recieved info", kpts)
    #get a dictionary containing the rotations for the current frame
    frame_rotations = {}
    for joint in kpts['joints']:
        frame_rotations[joint] = kpts[joint+'_angles']


    #for plotting
    for _j in kpts['joints']:
        if _j == 'hips': continue
        #print("plotting joint: ", _j)

        #get hierarchy of how the joint connects back to root joint
        hierarchy = kpts['hierarchy'][_j]

        #get the current position of the parent joint
        #r1 = kpts['hips']/kpts['normalization']
        r1 = np.array([0,0,0])
        for parent in hierarchy:
            if parent == 'hips': continue
            R = get_rotation_chain(parent, kpts['hierarchy'][parent], frame_rotations)
            r1 = r1 + R @ kpts['base_skeleton'][parent]

        #get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
        r2 = r1 + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]
        bones[_j].set_data([r1[0], r2[0]], [r1[1], r2[1]])
        bones[_j].set_3d_properties([r1[2], r2[2]])
        #plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'red')

        #print("final position of joint",_j, r1, r2)


    #plt.close()
    
"""
def draw_skeleton_from_joint_angles(kpts, bones, ax):
    global particles_initialized, particle_positions, particle_speeds, scatter_obj

    # --- Skeleton drawing ---
    frame_rotations = {}
    for joint in kpts['joints']:
        frame_rotations[joint] = kpts[joint + '_angles']

    for _j in kpts['joints']:
        if _j == 'hips':
            continue

        hierarchy = kpts['hierarchy'][_j]
        r1 = np.array([0, 0, 0])
        for parent in hierarchy:
            if parent == 'hips':
                continue
            R = get_rotation_chain(parent, kpts['hierarchy'][parent], frame_rotations)
            r1 = r1 + R @ kpts['base_skeleton'][parent]

        r2 = r1 + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]
        bones[_j].set_data([r1[0], r2[0]], [r1[1], r2[1]])
        bones[_j].set_3d_properties([r1[2], r2[2]])


# Function to read joint values from JSON file
def get_joint_values(data, joint_name, angle_type="Radian"):
    angles = data["Angles"]
    status = data["Status"]

    joint_angle_dict = {
        "HeadPitch": angles["Head"]["Pitch"][angle_type],
        "HeadYaw": angles["Head"]["Yaw"][angle_type],

        "LShoulderRoll": angles["Shoulders"]["Left"]["Roll"][angle_type],
        "LShoulderPitch": angles["Shoulders"]["Left"]["Pitch"][angle_type],
        "LShoulderYaw": angles["Shoulders"]["Left"]["Yaw"][angle_type],

        "RShoulderRoll": angles["Shoulders"]["Right"]["Roll"][angle_type],
        "RShoulderPitch": angles["Shoulders"]["Right"]["Pitch"][angle_type],
        "RShoulderYaw": angles["Shoulders"]["Right"]["Yaw"][angle_type],

        "LElbowRoll": angles["Elbows"]["Left"]["Roll"][angle_type],
        "RElbowRoll": angles["Elbows"]["Right"]["Roll"][angle_type],

        "LHand": status["Hands"]["Left"]["is_open"],
        "RHand": status["Hands"]["Right"]["is_open"],
    }


    return joint_angle_dict.get(joint_name, 0) or 0


# Function to read JSON file safely
def get_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as dfile:
            return json.load(dfile)
    except Exception as e:
        print(f"Couldn't read JSON file !!!!!!!: {e}")
        return None


# Function to run simulation
def run_simulation():
    counter = 0

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:5555")  # Connect to the publisher



    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    #plt.ion()

    joint_data = None
    joint_data_lock = threading.Lock()

    def subscriber():
        nonlocal joint_data
        while True:
            serialised_data = socket.recv()
            with joint_data_lock:
                joint_data = pickle.loads(serialised_data)
            print("received dict")


    thread = threading.Thread(target=subscriber, daemon=True)
    thread.start()

    def init():
        '''ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])'''
        ax.azim = 90
        ax.elev = -85
        ax.set_title('Pose from joint angles')
        ax.set_xlim3d(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylim3d(-2, 2)
        ax.set_ylabel('y')
        ax.set_zlim3d(-2, 2)
        ax.set_zlabel('z')

    #draw_skeleton_from_joint_angles(data, ax)

    #ani = FuncAnimation(fig, draw_skeleton_from_joint_angles(data, ax), init_func=init, interval=100, blit=False)
    first_package_received = False
    bones = {}

    def init_skeleton(joint_names):
        for joint in joint_names:
            bone, = ax.plot(xs=[], ys=[], zs=[], color='red')
            bones[joint] = bone

    def update_frame(frame):
        nonlocal joint_data, first_package_received
        with joint_data_lock:
            if joint_data is None:
                return
            if not first_package_received:
                init_skeleton(joint_data['joints'])
                first_package_received = True
            kpts = copy.deepcopy(joint_data)
        draw_skeleton_from_joint_angles(kpts, bones, ax)

    ani = FuncAnimation(fig, update_frame, init_func=init, interval=100, blit=False)
    plt.show()


run_simulation()
