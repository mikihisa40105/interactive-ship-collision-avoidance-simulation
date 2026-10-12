import os
import math
import datetime
import random
from enum import Enum
import warnings
import random
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, lognorm
from matplotlib.patches import Wedge
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from .geometry import calculate_distance_and_bearing, calculate_bearing_from_xy, \
calculate_lat_lon, rotate_bearing, co_xy_conv, az_180_180, az_0_to_360, \
is_near_zero, cal_dxdy, get_arctan, calc_rel_motion, calc_cpa, get_collision_point_azi_dst


class IniBasePos:
    def __init__(self, base_lat, base_lon) -> None:
        self.base_lat = base_lat
        self.base_lon = base_lon

    def calc_lat_lon(self, azi, dst):
        lat, lon = calculate_lat_lon(self.base_lat, self.base_lon, azi, dst)
        return lat, lon


class ShipType(Enum):
    Constant = 1
    ANS = 2
    Model = 3


class Ship:
    def __init__(self, ini_lat, ini_lon, ini_course, ini_speed, length, integrated_model, ship_type) -> None:
        self.INTERVAL = 30
        self.MAZ_TURN_ANGLE = 40
        self.TURN_ANGLES = [i for i in range(-self.MAZ_TURN_ANGLE, self.MAZ_TURN_ANGLE+1, 1)]
        self.MAX_TURN_ANGLE_1STEP = 5
        self.TGT_CO_CHANGE_THRESHOLD = 10
        if ship_type == ShipType.ANS:
            self.TGT_CO_CHANGE_THRESHOLD = 30
        self.integrated_model = integrated_model
        self.ship_type = ship_type
        self.length = length
        self.ini_course = ini_course
        self.course = ini_course
        self.speed = ini_speed
        self.lat_list = [ini_lat]
        self.lon_list = [ini_lon]
        self.cog_list = [ini_course]
        self.min_dst_list = []
        self.decision_made_time_idxs = []
        self.tgt_cog_decision_made = 0.0
        self.pd_history = []
        self.x_cp_history = []
        self.future_cog_list = []
        self.x_cp_futute = None
        self.is_reserved_future_cp = False
        self.dcpa_threshold=0.5
        self.tcpa_threshold=0.2
        self.port=10
        self.commercial=0.5

    def set_ans_params(self, dcpa_threshold, tcpa_threshold, port, commercial):
        self.dcpa_threshold = dcpa_threshold
        self.tcpa_threshold = tcpa_threshold
        self.port = port
        self.commercial = commercial

    def move(self, azi, dst):
        azi += random.gauss(mu=0.0, sigma=0.1)
        new_lat, new_lon =  calculate_lat_lon(self.lat_list[-1], self.lon_list[-1], azi, dst)
        self.lat_list.append(new_lat)
        self.lon_list.append(new_lon)
        self.cog_list.append(azi)
        self.course = azi

    def decide_maneuver_in_constraints(self, tgt):
        # Constant-speed, constant-course vessel.
        if self.ship_type == ShipType.Constant:
            return 0

        # Reset reserved future course when target course changes substantially.
        diff = abs(az_180_180(tgt.course - self.tgt_cog_decision_made))

        if diff > self.TGT_CO_CHANGE_THRESHOLD:
            dst, azi = calculate_distance_and_bearing(
                self.lat_list[-1], self.lon_list[-1],
                tgt.lat_list[-1], tgt.lon_list[-1]
            )

            dcpa, tcpa = calc_cpa(
                np.array([self.course]),
                np.array([self.speed]),
                np.array([tgt.course]),
                np.array([tgt.speed]),
                np.array([azi]),
                np.array([dst])
            )

            if float(dcpa[0]) <= 0.5:
                pass
            self.future_cog_list = []
            self.is_reserved_future_cp = False
            self.x_cp_futute = None

        # Continue an already reserved course change before making a new decision.
        if len(self.future_cog_list) > 0:
            return 0

        turn_angle = 0
        if self.ship_type == ShipType.ANS:
            turn_angle = self.decide_maneuver_by_ans(tgt)
        else:
            turn_angle = self.decide_maneuver_using_nn(tgt)

        self.tgt_cog_decision_made = tgt.course
        self.decision_made_time_idxs.append(len(self.lat_list)-1)

        # Apply small turns immediately; reserve larger turns over future steps.
        if abs(turn_angle) <= self.MAX_TURN_ANGLE_1STEP:
            return turn_angle

        num_max_turn = int(abs(turn_angle) / self.MAX_TURN_ANGLE_1STEP)
        if turn_angle > 0:
            for i in range(1, num_max_turn+1):
                self.future_cog_list.append(self.cog_list[-1] + i * self.MAX_TURN_ANGLE_1STEP)
            self.future_cog_list.append(self.cog_list[-1] + turn_angle)
            return self.MAX_TURN_ANGLE_1STEP
        else:
            for i in range(1, num_max_turn+1):
                self.future_cog_list.append(self.cog_list[-1] - i * self.MAX_TURN_ANGLE_1STEP)
            self.future_cog_list.append(self.cog_list[-1] + turn_angle)
            return -self.MAX_TURN_ANGLE_1STEP

    def decide_maneuver_by_ans(self, tgt):
        q_list = [100 for i in range(-self.MAZ_TURN_ANGLE , self.MAZ_TURN_ANGLE + 1)]
        dst, azi = calculate_distance_and_bearing(
            self.lat_list[-1], self.lon_list[-1],
            tgt.lat_list[-1], tgt.lon_list[-1]
        )
        for i in range(len(q_list)):
            co = self.course + self.TURN_ANGLES[i]
            delta_co = az_180_180(co - self.ini_course)
            q_list[i] -= abs(delta_co) * self.commercial
            if (delta_co < 0):
                q_list[i] -= self.port
            dcpa, tcpa = calc_cpa(np.array([co]), np.array([self.speed]), np.array([tgt.course]), np.array([tgt.speed]), np.array([azi]), np.array([dst]))
            if dcpa < self.dcpa_threshold and tcpa > 0 and tcpa < self.tcpa_threshold:
                if dcpa < 0.1:
                    dcpa = 0.1
                q_list[i] -= 10 / dcpa
        # get index of max value in q_list
        max_idx = q_list.index(max(q_list))
        return self.TURN_ANGLES[max_idx]

    def decide_maneuver_using_nn(self, tgt):
        X = []
        x_cp_now_list = []

        dst, azi = calculate_distance_and_bearing(
            self.lat_list[-1], self.lon_list[-1],
            tgt.lat_list[-1], tgt.lon_list[-1]
        )
        azi_rotated = az_180_180(rotate_bearing(azi, self.course))
        course_rotated = az_0_to_360(rotate_bearing(tgt.course, self.course))
        cp_azi, cp_dst = get_collision_point_azi_dst(azi, dst, tgt.course, tgt.speed, self.course, self.speed)
        cp_azi = az_180_180(cp_azi)
        x_cp_now, y_cp_now = cal_dxdy(cp_azi, cp_dst)
        self.x_cp_history.append(x_cp_now)

        # Return toward the initial course when the predicted CP leaves the avoidance range.
        delta_course = az_180_180(self.ini_course - self.course)
        if x_cp_now < -1.5 or x_cp_now > 1.5 or y_cp_now < 0.2 or y_cp_now > 7 or abs(delta_course)  >= self.MAZ_TURN_ANGLE:
            self.pd_history.append(-1)
            self.future_cog_list = []
            self.is_reserved_future_cp = False
            if delta_course > self.MAX_TURN_ANGLE_1STEP / 2:
                return delta_course #self.MAX_TURN_ANGLE_1STEP / 2
            elif delta_course < -self.MAX_TURN_ANGLE_1STEP / 2:
                return delta_course # -self.MAX_TURN_ANGLE_1STEP / 2
            else:
                return delta_course

        # get future CP x and y(avoid timing)
        if not self.is_reserved_future_cp:
            self.x_cp_futute = self.integrated_model.sampling_x_cp_future(
                x_cp_now, y_cp_now, tgt.length, self.length, self.speed, course_rotated)
            self.is_reserved_future_cp = True

        # if x_cp_futute and x_cp_now is already same value
        if abs(x_cp_now) - abs(self.x_cp_futute) > 0:
            return 0

        # Search the course change that brings the future CP closest to the target.
        min_delta_idx = -1
        min_delta_x_cp_future = 100.0
        for i in range(len(self.TURN_ANGLES)):
            own_cog = az_180_180(self.course + self.TURN_ANGLES[i])
            if abs(az_180_180(self.ini_course - own_cog)) > self.MAZ_TURN_ANGLE:
                continue
            dst, azi = calculate_distance_and_bearing(
                self.lat_list[-1], self.lon_list[-1],
                tgt.lat_list[-1], tgt.lon_list[-1]
            )
            azi_rotated = az_180_180(rotate_bearing(azi, own_cog))
            course_rotated = az_180_180(rotate_bearing(tgt.course, own_cog))
            cp_azi, cp_dst = get_collision_point_azi_dst(azi, dst, tgt.course, tgt.speed, own_cog, self.speed)
            cp_azi = az_180_180(cp_azi)
            _x_cp_future, _ = cal_dxdy(cp_azi, cp_dst)
            delta_x_cp_future = abs(_x_cp_future - self.x_cp_futute)
            if delta_x_cp_future < min_delta_x_cp_future:
                min_delta_x_cp_future = delta_x_cp_future
                min_delta_idx = i
        # Max Turn is not appropriate
        if min_delta_idx == 0 or min_delta_idx == len(self.TURN_ANGLES)-1:
            return 0
        else:
            return self.TURN_ANGLES[min_delta_idx]

