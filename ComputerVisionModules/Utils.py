import numpy as np
from ComputerVisionModules import BodyComponents, computerVision
import cv2
import math


class Utils:
    def __init__(self, filter_on = True, filter_deviation = 1, filter_difference = 20, filter_frame_number= 10):
        self.landmarks_name_id_dict = BodyComponents.BODY_LANDMARKS
        self.previous_frames_YZ_angles = []
        self.previous_frames_XY_angles = []
        self.previous_frames_XZ_angles = []
        self.filter_deviation = filter_deviation
        self.filter_difference = filter_difference
        self.filter_frame_number = filter_frame_number
        self.filter_on = filter_on
        self.Z_MAGNITUDE = 2.5

    def visualize_text(self, image, coordinate,
                       text="No text", color=(250, 250, 250),
                       size= 1, thickness=3,
                       font=cv2.FONT_HERSHEY_SIMPLEX,
                       lineType=cv2.LINE_AA):

        cv2.putText(image, text,
                    coordinate,
                    font, size, color, thickness, lineType)


    def get_coordinate_of_point(self, landmark_point, image):
        x = landmark_point[0]
        y = landmark_point[1]
        z = landmark_point[2]
        image_width, image_height = image.shape[1], image.shape[0]
        coordinate = (x * image_width, y * image_height, z * image_width/self.Z_MAGNITUDE)
        return coordinate


    def is_points_visible(self, points):
        for i in range(len(points)):
            visibility = points[i][3]
            if visibility < 90:
                return False
        return True

    def get_angle(self, points, dimension):

        if self.is_points_visible(points):
            print("Please take a good position ! Not all needed points are visible to calculate accurate angle")
            return None

        p1, p2, p3 = points

        points_in_dimension = self.get_dimension_axis(p1, p2, p3, dimension)
        degree, radian = self.calculateAngle(points_in_dimension)

        if self.filter_on:
            if dimension == "YZ": #roll
                self.previous_frames_YZ_angles.append(degree)
                degree = self.get_filtered_angle(self.previous_frames_YZ_angles, degree)

            if dimension == "XY": #pitch
                self.previous_frames_XY_angles.append(degree)
                degree = self.get_filtered_angle(self.previous_frames_XY_angles, degree)

            if dimension == "ZX":  # Yaw
                self.previous_frames_XZ_angles.append(degree)
                degree = self.get_filtered_angle(self.previous_frames_XZ_angles, degree)

        radian = np.deg2rad(degree)

        return {
            "Degree": degree,
            "Radian": radian
        }

    def get_dimension_axis(self, p1, p2, p3, dimension):
        if dimension == "XY":
            return ((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]))  # roll

        elif dimension == "YZ":
            return ((p1[1], p1[2]), (p2[1], p2[2]), (p3[1], p3[2]))  # pitch

        elif dimension == "XZ":
            return ((p1[0], p1[2]), (p2[0], p2[2]), (p3[0], p3[2]))  # yaw


    def calculateAngle(self, points):
        point1, point2, point3 = points

        #print("POINNNT:::::",point1)

        value1 = np.arctan2(point3[1] - point2[1], point3[0] - point2[0])
        value2 = np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
        radian = value1 - value2
        #radian = (radian + np.pi) % (2 * np.pi) - np.pi

        #elbow to shoulder
        #v1 = np.arctan2(point1[0] - point2[0], point1[1] - point2[1])  # Elbow to Shoulder
        #v2 = np.arctan2(point3[0] - point2[0], point3[1] - point2[1])  # Hip to Shoulder

        #theta_rad = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
        #theta_deg = np.degrees(theta_rad)

        if radian > np.pi:
            radian = 2*np.pi - radian


        #v1 = ((point1[0] - point2[0]), (point1[1] - point2[1]))
        #v2 = ((point3[0] - point2[0]), (point3[1] - point2[1]))

        #product = (point1[0] - point2[0]) * (point3[0] - point2[0]) + (point1[1] - point2[1]) * (point3[1] - point2[1])
        #lenV1 = math.sqrt((point1[0] - point2[0])** 2 + (point1[1] - point2[1])** 2)
        #lenV2 = math.sqrt((point3[0] - point2[0])** 2 + (point3[1] - point2[1])** 2)

        #radian = np.arccos(product/(lenV1 * lenV2))

        degree = np.rad2deg(radian)

        #theta_rad = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])


        #theta_deg = np.degrees(theta_rad)

        #return int(theta_deg), round(theta_rad, 4)

        return int(degree), round(radian, 4)

    def get_filtered_angle(self, frames_list, angle):

        if len(frames_list) > self.filter_frame_number:
            frames_list = frames_list[-self.filter_frame_number:]

        if len(frames_list) == 0:
            return angle

        slot_average = round(np.sum(frames_list) / len(frames_list))

        if max(frames_list) - min(frames_list) < self.filter_difference:
        # if there is a big difference between angles, do not apply this filter beceuse
        # that means there is a fast change between movements. Thus we do not want to affect
        # new totally different movement.

            if abs(angle - slot_average) > self.filter_deviation:
                return slot_average

        return angle





