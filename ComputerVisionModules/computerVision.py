import random
import time

import cv2
import mediapipe as mp
import numpy as np
import zmq
import math
import json

from ComputerVisionModules import Shoulders, LandmarksModule, Elbows, Head, Hands, Hips
from dataclasses import dataclass, field
from ProgramOutputModule import OutputModule

import logging
import os

#for angle calc
import numpy as np
import sys
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import copy
import socket


drawing_mp   = mp.solutions.drawing_utils
pose_mp      = mp.solutions.pose
face_mesh_mp = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands

face_mesh  = face_mesh_mp.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
hands_pose = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)
pose       = pose_mp.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

shoulders    = Shoulders.Shoulders()
elbows       = Elbows.Elbows()
hips         = Hips.Hips()
hands_module = Hands

landmark_handler = LandmarksModule.Landmarks()
program_output   = OutputModule.Output()


ANGLE_TYPE = "Degree"
arr = []


def Decompose_R_ZXY(R):
    #decomposes as RzRXRy. Note the order: ZXY <- rotation by y first
    thetaz = np.arctan2(-R[0,1], R[1,1])
    thetay = np.arctan2(-R[2,0], R[2,2])
    thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

    return thetaz, thetay, thetax


def Get_R2(A, B):
    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)

    return R


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


def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):
    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0: continue
        _r_angles = frame_rotations[parent_name]
        R = get_R_z(_r_angles[0]) @ get_R_x(_r_angles[1]) @ get_R_y(_r_angles[2])
        _invR = _invR@R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

    _R = Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = Decompose_R_ZXY(_R)

    joint_rs = np.array([tz, tx, ty])
    #print("ANGLES:",joint_name, np.degrees(joint_rs))

    return joint_rs


def get_hips_position_and_rotation(frame_pos, root_joint = 'hips', root_define_joints = ['left_hip', 'neck']):
    #root position is saved directly
    root_position = frame_pos[root_joint]

    #calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u/np.sqrt(np.sum(np.square(root_u)))

    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v/np.sqrt(np.sum(np.square(root_v)))

    # Gram-Schmidt: make root_v perpendicular to root_u so that C is a
    # proper rotation matrix (det = 1).  Without this, root_u (hip width)
    # and root_v (spine) are only ~90° in a perfect T-pose; as the body
    # moves they become non-orthogonal and Decompose_R_ZXY couples the
    # yaw/pitch/roll channels.
    root_v = root_v - np.dot(root_v, root_u) * root_u
    root_v = root_v / np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    #Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz, thetay, thetax = Decompose_R_ZXY(C)

    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation


def calculate_joint_angles(kpts):
    #get the keypoints positions in the current frame
    frame_pos = {}
    for joint in kpts['joints']:
        frame_pos[joint] = kpts[joint]

    root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

    frame_rotations = {'hips': root_rotation}

    #center the body pose
    for joint in kpts['joints']:
        frame_pos[joint] = frame_pos[joint] - root_position

    #get the max joints connectsion
    max_connected_joints = 0
    for joint in kpts['joints']:
        if len(kpts['hierarchy'][joint]) > max_connected_joints:
            max_connected_joints = len(kpts['hierarchy'][joint])

    depth = 2
    while(depth <= max_connected_joints):
        pending = {}
        for joint in kpts['joints']:
            if len(kpts['hierarchy'][joint]) == depth:
                joint_rs = get_joint_rotations(joint, kpts['hierarchy'], kpts['offset_directions'], frame_rotations, frame_pos)
                parent = kpts['hierarchy'][joint][0]
                if parent not in pending:
                    pending[parent] = []
                pending[parent].append(joint_rs)
        for parent, angle_list in pending.items():
            frame_rotations[parent] = np.mean(angle_list, axis=0)
        depth += 1

    #for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
    for _j in kpts['joints']:
        if _j not in list(frame_rotations.keys()):
            frame_rotations[_j] = np.array([0.,0.,0.])

    #update dictionary with current angles.

    """
    angles_dict = {}
    for joint in kpts["joints"]:
        angles_dict[joint] = frame_rotations[joint].tolist()

    """
    
    # Sign adjustment for Unity's left-handed coordinate system
    # These are applied at the OUTPUT stage only (not inside
    # get_joint_rotations) so that frame_rotations keeps the physically
    # correct angles needed by the parent-chain computation.

    # Left-side joints: negate tz and ty (mirrored relative to right side).
    # Hips: swap tx↔ty (bone axis mismatch) + negate tz (roll direction).
    # Neck: negate tz (roll direction is inverted without this).
    # Right-side joints: no adjustment
    for joint in kpts['joints']:
        angles = frame_rotations[joint]
        if joint == 'hips':
            angles = np.array([-angles[0], angles[2], angles[1]])
        elif joint.startswith("left_"):
            angles = np.array([-angles[0], angles[1], -angles[2]])
        elif joint == 'neck':
            angles = np.array([-angles[0], angles[1], angles[2]])
        kpts[joint + '_angles'] = angles


    #convert joint angles list to numpy arrays.
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = np.array(kpts[joint + '_angles'])
        #print(joint, kpts[joint+'_angles'].shape)

    return kpts


def get_fixed_bone_lengths(kpts):
    bone_lengths = {'left_hip': 0.5,
                        'left_knee': 2.4,
                            'left_ankle': 2.4,
                    'right_hip': 0.5,
                        'right_knee': 2.4,
                            'right_ankle': 2.4,
                    'left_shoulder': 0.8,
                        'left_elbow': 1.3,
                            'left_wrist': 2.1,
                    'right_shoulder': 0.8,
                        'right_elbow': 1.3,
                            'right_wrist': 2.1,
                    'neck': 3.0}

    kpts['bone_lengths'] = bone_lengths


def get_fixed_bone_lengths_factor(kpts, factor):
    bone_lengths = {'left_hip': 0.5 * factor/5,
                        'left_knee': 2.4* factor/5,
                            'left_ankle': 2.4* factor/5,
                    'right_hip': 0.5* factor/5,
                        'right_knee': 2.4* factor/5,
                            'right_ankle': 2.4* factor/5,
                    'left_shoulder': 0.8* factor/5,
                        'left_elbow': 1.3* factor/5,
                            'left_wrist': 2.1* factor/5,
                    'right_shoulder': 0.8* factor/5,
                        'right_elbow': 1.3* factor/5,
                            'right_wrist': 2.1* factor/5,
                    'neck': 3.0 * factor/5}

    #for key in bone_lengths.keys():
    #    bone_lengths[key] *= factor
    #bone_lengths['left_elbow'] *= factor/5

    kpts['bone_lengths'] = bone_lengths


def get_bone_lengths(kpts, factor):
    """
    Define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips': continue
        parent = kpts['hierarchy'][joint][0]

        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1))

        _bone_length = np.median(_bone_lengths) * factor
        bone_lengths[joint] = _bone_length

        # plt.hist(bone_lengths, bins = 25)
        # plt.title(joint)
        # plt.show()

    #print(bone_lengths)
    kpts['bone_lengths'] = bone_lengths
    return


#Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.
def get_base_skeleton(kpts, normalization_bone = 'neck'):
    #this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']

    #define skeleton offset directions
    offset_directions = {}
    offset_directions['left_hip']       = np.array([ 1,  0, 0])
    offset_directions['left_knee']      = np.array([ 0, -1, 0])
    offset_directions['left_ankle']     = np.array([ 0, -1, 0])

    offset_directions['right_hip']      = np.array([-1,  0, 0])
    offset_directions['right_knee']     = np.array([ 0, -1, 0])
    offset_directions['right_ankle']    = np.array([ 0, -1, 0])

    offset_directions['neck']           = np.array([ 0,  1, 0])

    offset_directions['left_shoulder']  = np.array([ 1,  0, 0])
    offset_directions['left_elbow']     = np.array([ 1,  0, 0])
    offset_directions['left_wrist']     = np.array([ 1,  0, 0])

    offset_directions['right_shoulder'] = np.array([-1,  0, 0])
    offset_directions['right_elbow']    = np.array([-1,  0, 0])
    offset_directions['right_wrist']    = np.array([-1,  0, 0])

    #set bone normalization length. Set to 1 if you dont want normalization
    normalization = kpts['bone_lengths'][normalization_bone]
    #normalization = 1

    #base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {'hips': np.array([0,0,0])}
    def _set_length(joint_type):
        base_skeleton['left_' + joint_type]  = offset_directions['left_' + joint_type]  * ((body_lengths['left_' + joint_type] + body_lengths['right_' + joint_type])/(2 * normalization))
        base_skeleton['right_' + joint_type] = offset_directions['right_' + joint_type] * ((body_lengths['left_' + joint_type] + body_lengths['right_' + joint_type])/(2 * normalization))

    _set_length('hip')
    _set_length('knee')
    _set_length('ankle')
    _set_length('shoulder')
    _set_length('elbow')
    _set_length('wrist')
    base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck']/normalization)

    kpts['offset_directions'] = offset_directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return


def median_filter(kpts, window_size = 3):
    import copy
    filtered = copy.deepcopy(kpts)

    from scipy.signal import medfilt

    for joint in filtered['joints']:
        joint_kpts = np.array(kpts[joint])
        filtered[joint] = medfilt(joint_kpts, kernel_size=window_size)

    return filtered


def add_hips_and_neck(kpts):
    #we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    #add hips kpts
    difference = kpts['left_hip'] - kpts['right_hip']
    difference = difference/2
    hips = kpts['right_hip'] + difference
    kpts['hips'] = hips
    kpts['joints'].append('hips')

    #add neck kpts
    difference = kpts['left_shoulder'] - kpts['right_shoulder']
    difference = difference/2
    neck = kpts['right_shoulder'] + difference
    kpts['neck'] = neck
    kpts['joints'].append('neck')

    #define the hierarchy of the joints
    hierarchy = {'hips': [],
                    'left_hip': ['hips'],
                        'left_knee': ['left_hip', 'hips'],
                            'left_ankle': ['left_knee', 'left_hip', 'hips'],
                    'right_hip': ['hips'],
                        'right_knee': ['right_hip', 'hips'],
                            'right_ankle': ['right_knee', 'right_hip', 'hips'],
                    'neck': ['hips'],
                        'left_shoulder': ['neck', 'hips'],
                            'left_elbow': ['left_shoulder', 'neck', 'hips'],
                                'left_wrist': ['left_elbow', 'left_shoulder', 'neck', 'hips'],
                        'right_shoulder': ['neck', 'hips'],
                            'right_elbow': ['right_shoulder', 'neck', 'hips'],
                                'right_wrist': ['right_elbow', 'right_shoulder', 'neck', 'hips']
                 }

    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts


def convert_to_dictionary(kpts): # as array mb?
    # its easier to manipulate keypoints by joint name (as in pybullet)
    keypoints_to_index = {'left_hip': 6,       'left_knee': 8,   'left_ankle': 10,
                          'right_hip': 7,      'right_knee': 9,  'right_ankle': 11,
                          'left_shoulder': 0,  'left_elbow': 2,  'left_wrist': 4,
                          'right_shoulder': 1, 'right_elbow': 3, 'right_wrist': 5
                          }

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        x, y, z = kpts[k_index]
        # MediaPipe world frame: x = screen-left (person's right), y = up,
        # z = depth (increases away from camera).
        # Skeleton convention: +x = person's left, +y = up, +z = backward.
        # Negate x so person's left maps to +x.  Z is kept as-is (positive
        # = away from camera = backward), giving a left-handed frame that
        # pairs with the sign convention in JointReceiver.cs.
        kpts_dict[key] = np.array([-x*10, y*10, z*10])

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    #print(kpts_dict)

    return kpts_dict


def get_xyz_as_array(results): # as array mb?
    mp_joint_id = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Based on Mediapipe order

    keypoints= []
    for id in mp_joint_id:
        lm = results.pose_world_landmarks.landmark[id]  # 3D world coords (meters), not image-plane
        keypoints.append([lm.x, lm.y, lm.z])

    return np.array(keypoints)


def add_r_mirroring(filtered_kpts):
    """
    Reversed mirroring: the avatar mirrors the user as if looking
    in a mirror — the user's right limb drives the avatar's left limb
    and vice versa.

    Mathematically, a sagittal-plane reflection negates tz and ty while
    preserving tx.  But calculate_joint_angles already applies per-side
    sign adjustments (left joints get [-tz, tx, -ty]).  These adjustments
    exactly cancel the mirror reflection, so the net operation is a
    straight swap of the adjusted angle arrays — no extra sign changes.
    """
    get_fixed_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)
    angles_dict = calculate_joint_angles(filtered_kpts)

    left_right_pairs = [
        ("left_hip_angles",      "right_hip_angles"),
        ("left_knee_angles",     "right_knee_angles"),
        ("left_ankle_angles",    "right_ankle_angles"),
        ("left_shoulder_angles", "right_shoulder_angles"),
        ("left_elbow_angles",    "right_elbow_angles"),
        ("left_wrist_angles",    "right_wrist_angles"),
    ]

    swapped_dict = copy.deepcopy(filtered_kpts)

    for left_key, right_key in left_right_pairs:
        if left_key in swapped_dict and right_key in swapped_dict:
            # Simple swap — the sign adjustments from calculate_joint_angles
            # and the mirror reflection cancel each other out exactly.
            left_copy = np.array(swapped_dict[left_key])
            swapped_dict[left_key] = np.array(swapped_dict[right_key])
            swapped_dict[right_key] = left_copy

    return swapped_dict


def add_contrasting(filtered_kpts):
    get_fixed_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)
    angles_dict = calculate_joint_angles(filtered_kpts)

    # Upper body: reflection over X axis — negate tz and ty, keep tx
    upper_body_joints = [
        "neck_angles",
        "left_shoulder_angles",  "right_shoulder_angles",
        "left_elbow_angles",     "right_elbow_angles",
        "left_wrist_angles",     "right_wrist_angles",
    ]
    for key in upper_body_joints:
        if key in filtered_kpts:
            a = filtered_kpts[key]
            filtered_kpts[key] = np.array([-a[0], a[1], -a[2]])

    # Lower body: reflection over Y axis — negate tz and tx, keep ty
    lower_body_joints = [
        "left_hip_angles",    "right_hip_angles",
        "left_knee_angles",   "right_knee_angles",
        "left_ankle_angles",  "right_ankle_angles",
    ]
    for key in lower_body_joints:
        if key in filtered_kpts:
            a = filtered_kpts[key]
            filtered_kpts[key] = np.array([-a[0], -a[1], a[2]])

    return filtered_kpts


def add_amplifying(filtered_kpts, factor):
    get_fixed_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)
    calculate_joint_angles(filtered_kpts)

    amplified_kpts = copy.deepcopy(filtered_kpts)

    angle_threshold = 0.5
    length_factor = 1.0 + factor / 10


    def maybe_amplify(joint_name, angle_index=1):
        angles = amplified_kpts.get(joint_name + "_angles")
        if angles is None:
            return 1.0  # no change

        if abs(angles[angle_index]) > angle_threshold:
            return length_factor
        return 1.0


    bone_lengths = {
        "left_elbow":  maybe_amplify("left_shoulder")  * 1.3,
        "right_elbow": maybe_amplify("right_shoulder") * 1.3,
        "left_wrist":  maybe_amplify("left_elbow")     * 2.1,
        "right_wrist": maybe_amplify("right_elbow")    * 2.1,
    }

    #preserve symmetry
    for key in ["left", "right"]:
        amplified_kpts[f"{key}_shoulder_angles"] *= factor / 5
        amplified_kpts[f"{key}_elbow_angles"] *= factor / 5

    #override bone lengths
    amplified_kpts["bone_lengths"].update(bone_lengths)

    return amplified_kpts


def setup_mode_logger():
    os.makedirs("logs", exist_ok=True)
    session_id = time.strftime("%Y-%m-%d__%H.%M.%S")
    log_path = os.path.join("logs", f"session_{session_id}.log")
    logger = logging.getLogger("ModeSwitch")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S")
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Session started — log: {log_path}")
    
    return logger


mode_log = setup_mode_logger()

SESSION_LENGTH = 180

AUTO_SWITCH_TIMES = [45, 68, 86, 109, 128, 145, 158, 173]  # 8 switches
AUTO_BREAK_POOL   = ["contrasting", "reversed mirroring"]


@dataclass
class AutoModeState:
    start_time: float   = 0.0
    current_mode: str   = "mirroring"
    switch_index: int   = 0
    break_history: list = field(default_factory=list)


def update_auto_mode(state):
    elapsed = time.time() - state.start_time

    while (state.switch_index < len(AUTO_SWITCH_TIMES)
           and elapsed >= AUTO_SWITCH_TIMES[state.switch_index]):
        prev = state.current_mode
        if state.switch_index % 2 == 0:
            recent = state.break_history[-1] if state.break_history else None
            candidates = [m for m in AUTO_BREAK_POOL if m != recent]
            if not candidates:
                candidates = list(AUTO_BREAK_POOL)
            chosen = random.choice(candidates)
            state.break_history.append(chosen)
            state.current_mode = chosen
        else:
            state.current_mode = "mirroring"
        state.switch_index += 1
        mode_log.info(
            f"AUTO switch {state.switch_index}/8 at t={elapsed:.1f}s: "
            f"'{prev}' → '{state.current_mode}'"
        )

    return state.current_mode


RANDOM_NUM_SWITCHES = 8
RANDOM_MIN_SEGMENT  = 10  # minumum seconds per switch
RANDOM_MODE_POOL    = ["mirroring", "contrasting", "reversed mirroring"]


def _generate_random_switch_times():
    num_segments = RANDOM_NUM_SWITCHES + 1
    min_total = num_segments * RANDOM_MIN_SEGMENT
    
    slack = SESSION_LENGTH - min_total
    
    cuts = sorted(random.uniform(0, slack) for _ in range(num_segments - 1))
    extras = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [slack - cuts[-1]]
    
    segments = [RANDOM_MIN_SEGMENT + e for e in extras]
    switch_times = []
    
    t = 0.0
    for s in segments[:-1]:
        t += s
        switch_times.append(t)
    
    return switch_times


@dataclass
class RandomModeState:
    start_time: float   = 0.0
    current_mode: str   = "mirroring"
    switch_index: int   = 0
    switch_times: list  = field(default_factory=list)

    def __post_init__(self):
        self.start_time = time.time()
        self.switch_times = _generate_random_switch_times()
        mode_log.info(
            f"RANDOM session started — switch times: "
            f"{[f'{t:.1f}s' for t in self.switch_times]}"
        )


def update_random_mode(state):
    elapsed = time.time() - state.start_time

    while (state.switch_index < len(state.switch_times)
           and elapsed >= state.switch_times[state.switch_index]):
        prev = state.current_mode
        candidates = [m for m in RANDOM_MODE_POOL if m != state.current_mode]
        state.current_mode = random.choice(candidates)
        state.switch_index += 1
        mode_log.info(
            f"RANDOM switch {state.switch_index}/8 at t={elapsed:.1f}s: "
            f"'{prev}' → '{state.current_mode}'"
        )

    return state.current_mode


auto_mode_state   = None
random_mode_state = None


def run_computer_vision(frame, interface_inputs, socket):
    a_factor = interface_inputs["aSliderValue"]
    auto = interface_inputs["auto_mode"]
    arr = []
    frame_counter = 1

    # Recolor image BGR to RGB
    image = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
    image.flags.writeable = False

    face_results = face_mesh.process(image)
    body_results = pose.process(image)
    hand_results = hands_pose.process(image)

    if body_results.pose_landmarks:
        body_info = landmark_handler.get_body_landmarks_info(body_results , image)
        print("mediapipe:", body_info)

    else:
        return image

    image.flags.writeable = True

    kpts = get_xyz_as_array(body_results)
    print('COORDINATES:',kpts)

    # Coordinate mapping is handled in convert_to_dictionary (x negated for left/right).
    # No Rz(pi) needed — world landmarks are already in a proper 3D frame.
    kpts = convert_to_dictionary(kpts)
    print("DICTIONARY:", kpts)
    add_hips_and_neck(kpts)

    filtered_kpts = kpts

    global auto_mode_state, random_mode_state
    random = interface_inputs["random_mode"]
    
    if auto == 1:
        if auto_mode_state is None:
            auto_mode_state = AutoModeState(start_time=time.time())
            mode_log.info(
                f"AUTO session started — switch times: "
                f"{[f'{t}s' for t in AUTO_SWITCH_TIMES]}"
            )
        random_mode_state = None
        mode = update_auto_mode(auto_mode_state)
    
    elif random == 1:
        if random_mode_state is None:
            random_mode_state = RandomModeState()
        auto_mode_state = None
        mode = update_random_mode(random_mode_state)
    
    else:
        auto_mode_state = None
        random_mode_state = None
        mode = interface_inputs['mode']

    information = copy.deepcopy(filtered_kpts)

    if mode == "reversed mirroring":
        angles_dict = add_r_mirroring(filtered_kpts)

    elif mode == "contrasting":
        angles_dict = add_contrasting(filtered_kpts)

    elif mode == "amplifying":
        angles_dict = add_amplifying(filtered_kpts, a_factor)

    else:
        get_fixed_bone_lengths(filtered_kpts)
        get_base_skeleton(filtered_kpts)
        angles_dict = calculate_joint_angles(filtered_kpts)
        print("angles dict: ", angles_dict)

    # Hip displacement: vertical (y) from world-landmark ankle average,
    # lateral (x) from image-plane hip center.
    # World landmarks are hip-relative (origin = hip center) so they
    # can't detect overall body translation — only the image-plane
    # landmarks move when the person steps sideways.
    avg_ankle = (filtered_kpts['left_ankle'] + filtered_kpts['right_ankle']) / 2.0

    lm = body_results.pose_landmarks.landmark
    hip_img_x = (lm[23].x + lm[24].x) / 2          # 0-1, image-plane
    hip_img_y = (lm[23].y + lm[24].y) / 2          # 0-1, image-plane
    # Image x: 0 = left edge, 1 = right edge.
    # Person's left = image right (person faces camera).
    # Skeleton +x = person's left.  So (x - 0.5) gives +x = person left.
    # Scale ≈ 25 maps a reasonable step (~20% of frame) to ~5 skeleton
    # units (~0.5 m × 10).  Camera-dependent; tune if needed.
    hip_lateral = (hip_img_x - 0.5) * 25

    # Image y: 0 = top, 1 = bottom. Invert so up = positive.
    # GroundFeet() in Unity prevents feet from going through the floor.
    hip_vertical = -(hip_img_y - 0.5) * 25

    angles_dict['hip_position'] = np.array([hip_lateral, hip_vertical, avg_ankle[2]])

    try:
        socket.send(pickle.dumps(angles_dict), zmq.NOBLOCK)

    except zmq.Again:
        print("skipped sending angles_dict")

    if interface_inputs["BlackBackground"]:
        image = np.zeros(image.shape, np.uint8)
    else:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    drawing_mp.draw_landmarks(image, body_results.pose_landmarks, pose_mp.POSE_CONNECTIONS,
                              drawing_mp.DrawingSpec(color=(255, 255, 255), thickness = 5,  circle_radius = 5),
                              drawing_mp.DrawingSpec(color=(0,    94, 255), thickness = 15, circle_radius = 5)
                              )

    frame_counter += 1

    return image