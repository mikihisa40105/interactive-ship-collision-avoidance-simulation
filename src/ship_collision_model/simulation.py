import os
import math
import datetime
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
from .ships import Ship, IniBasePos, ShipType


class Simulation:
    def __init__(self):
        self.INTERVAL = 30  # sec
        self.ship_list = []
        self.cnt = 1
        self.frames = []
        self.min_dst = 10
        self.dcpa_list = []
        self.dst_list = []
        self.x_cp_history = []
        self.passed_under_1cable = False
        self.passed_under_2cable = False

    def add_ship(self, ship):
        self.ship_list.append(ship)

    def step(self):
        turn_angles = []
        # Decide each ship's turn angle from the current state.
        if self.cnt == 1:
            for i in range(len(self.ship_list)):
                turn_angles.append(0)
        else:
            for i in range(len(self.ship_list)):
                tgts = self.ship_list.copy()
                tgts.pop(i)
                turn_angles.append(self.ship_list[i].decide_maneuver_in_constraints(tgts[0]))
        # Move each ship and update tracks.
        for i in range(len(self.ship_list)):
            if len(self.ship_list[i].future_cog_list) > 0:
                azi = self.ship_list[i].future_cog_list.pop(0)
            else:
                azi = self.ship_list[i].course + turn_angles[i]
            dst = self.ship_list[i].speed / 3600 * self.INTERVAL
            self.ship_list[i].move(azi, dst)

        # Calculate pairwise distances between ships.
        for i in range(len(self.ship_list)):
            min_dst_for_ship = 10
            for j in range(len(self.ship_list)):
                if i == j:
                    continue
                dst, azi = calculate_distance_and_bearing(
                    self.ship_list[i].lat_list[-1],
                    self.ship_list[i].lon_list[-1],
                    self.ship_list[j].lat_list[-1],
                    self.ship_list[j].lon_list[-1]
                )
                dcpa, tcpa = calc_cpa(
                    np.array([self.ship_list[i].course]), np.array([self.ship_list[i].speed]),
                    np.array([self.ship_list[j].course]), np.array([self.ship_list[j].speed]),
                    np.array([azi]), np.array([dst]))

                if i==0 and tcpa[0] >= 0:
                    # Cal. Collision Point
                    azi_rotated = az_180_180(rotate_bearing(azi, self.ship_list[0].course))
                    course_rotated = az_0_to_360(rotate_bearing(self.ship_list[1].course, self.ship_list[0].course))
                    cp_azi, cp_dst = get_collision_point_azi_dst(azi, dst, self.ship_list[1].course, self.ship_list[1].speed, self.ship_list[0].course, self.ship_list[0].speed)
                    cp_azi = az_180_180(cp_azi)
                    x_cp_now, y_cp_now = cal_dxdy(cp_azi, cp_dst)
                    self.dst_list.append(dst)
                    self.dcpa_list.append(dcpa[0])
                    self.x_cp_history.append(x_cp_now)

                if dcpa[0] < dst and tcpa[0] > 0.0 and tcpa[0] < 1.0/60.0:
                    dst = dcpa[0]
                if dst < self.min_dst:
                    self.min_dst = dst
                if dst < min_dst_for_ship:
                    min_dst_for_ship = dst
            self.ship_list[i].min_dst_list.append(min_dst_for_ship)
        if self.min_dst < 0.1:
            self.passed_under_1cable = True
        if self.min_dst < 0.2:
            self.passed_under_2cable = True
        self.cnt += 1

    def visualize(self, own_idx=0, save_only=False):
        for time in range(0, len(self.ship_list[0].cog_list)-1):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.subplots()[1]
            for i in range(len(self.ship_list)):
                # Expand plot extent around the initial ship position.
                for delta_az, length in [(-20,  0.45), (0, 0.4), (20, 0.45), (180, 0.08)]:
                    azi = az_180_180(self.ship_list[i].cog_list[0] + delta_az)
                    length = self.ship_list[i].speed * length
                    length_lat, length_lon = calculate_lat_lon(
                        self.ship_list[i].lat_list[0], self.ship_list[i].lon_list[0], azi, length)
                    plt.plot(
                        [self.ship_list[i].lon_list[0], length_lon],
                        [self.ship_list[i].lat_list[0], length_lat],
                        color="white", linewidth=0.001, alpha=0.001)
                cnt = 0
                for k in self.ship_list[i].decision_made_time_idxs:
                    cnt += 1
                plt.plot(self.ship_list[i].lon_list, self.ship_list[i].lat_list, '.-', alpha=0.3, color="green", linewidth=1, ms=1)
                lat, lon = self.ship_list[i].lat_list[time], self.ship_list[i].lon_list[time]
                plt.scatter(lon, lat, color="blue", s=6)
                vec_lat, vec_lon = calculate_lat_lon(
                    lat, lon, self.ship_list[i].cog_list[time], self.ship_list[i].speed/10)
                plt.plot([lon, vec_lon], [lat, vec_lat], color="blue", linewidth=0.5)
                azi_array = np.arange(0, 361, 20)
                lat_list, lon_list = [], []
                for azi in azi_array:
                    _lat, _lon = calculate_lat_lon(lat, lon, azi, 0.2)
                    lat_list.append(_lat)
                    lon_list.append(_lon)
                plt.plot(lon_list, lat_list, linestyle="dotted", color="blue", alpha=0.3, linewidth=0.6)
            ax.set_aspect(1/np.cos(np.deg2rad(self.ship_list[0].lat_list[0])))
            plt.xticks(color="None")
            plt.yticks(color="None")
            plt.grid(True)
            plt.title(f"frame{time}")
            frame_path = f"frame_{time}.png"
            plt.savefig(frame_path, dpi=200)
            self.frames.append(frame_path)
            if save_only:
                plt.close()

    def create_gif(self, idx, output_dir="outputs/anime"):
        os.makedirs(output_dir, exist_ok=True)
        images = [Image.open(frame) for frame in self.frames]
        images[0].save(
            os.path.join(output_dir, "animated" + str(idx) + ".gif"),
            save_all=True,
            append_images=images[1:],
            duration=300,
            loop=0,
        )


class SimulationRepetition:
    def __init__(self):
        self.all_lat_list1 = []
        self.all_lon_list1 = []
        self.all_lat_list2 = []
        self.all_lon_list2 = []
        self.all_dst_list = []
        self.all_dcpa_list = []
        self.all_x_cp_list = []
        self.min_pass_dst_list = []
        self.danger_idxs = []
        self.own_type = None
        self.tgt_type = None
        
        self.dst_threshold_for_evaluatrion = 2.0
        self.dcpa_threshold_for_evaluatrion = 0.2
        
    def simulate_and_visualize_all_tracks(self, own_type, tgt_type, integrated_model, instance="1", plot=None,
                                          NUM_TRIAL=100, OWN_LENGTH=200, TGT_LENGTH=200, TGT_DST=5, TGT_AZI=90,
                                          dcpa_threshold=0.5, tcpa_threshold=0.2, port=10, commercial=0.5):
        self.own_type = own_type
        self.tgt_type = tgt_type
        for i in range(NUM_TRIAL):
            ini_base = IniBasePos(35.0, 139.95)
            sim = Simulation()
            ship1 = Ship(35.0, 139.95, 90.0, 12.0, OWN_LENGTH, integrated_model, own_type)
            ship1.set_ans_params(dcpa_threshold, tcpa_threshold, port, commercial)
            sim.add_ship(ship1)
            lat, lon = ini_base.calc_lat_lon(TGT_AZI, TGT_DST)
            ship2 = Ship(lat, lon, 270.0, 12.0, TGT_LENGTH, integrated_model, tgt_type)
            ship2.set_ans_params(dcpa_threshold, tcpa_threshold, port, commercial)
            sim.add_ship(ship2)
            
            num_timestep = 40
            if TGT_DST <= 3: num_timestep = 17
            elif TGT_DST <= 4: num_timestep = 23
            elif TGT_DST <= 5: num_timestep = 28
            elif TGT_DST <= 6: num_timestep = 33
            elif TGT_DST <= 7: num_timestep = 39
            for time in range(num_timestep):
                sim.step()
            self.min_pass_dst_list.append(sim.min_dst)
            self.all_dst_list.append(sim.dst_list)
            self.all_dcpa_list.append(sim.dcpa_list)
            self.all_lat_list1.append(sim.ship_list[0].lat_list)
            self.all_lon_list1.append(sim.ship_list[0].lon_list)
            self.all_lat_list2.append(sim.ship_list[1].lat_list)
            self.all_lon_list2.append(sim.ship_list[1].lon_list)
            self.all_x_cp_list.append(sim.x_cp_history)

            if i != 0 and i % 10000 == 0:
                print(str(datetime.datetime.now()) + ":" + str(i) + "simulations completed.")
        
        fig, ax = plt.subplots(figsize=(4, 2))
        for i in range(len(self.all_lat_list1)):
            count = sum(1 for dst, dcpa in zip(self.all_dst_list[i], self.all_dcpa_list[i]) if dst <= self.dst_threshold_for_evaluatrion and dcpa <= self.dcpa_threshold_for_evaluatrion)
            if (count >= 1):
                self.danger_idxs.append(i)
            if plot=='all':
                ax.plot(self.all_lon_list1[i], self.all_lat_list1[i], '.-', alpha=0.2, color="green", linewidth=0.5, ms=0.5)
                ax.plot(self.all_lon_list2[i], self.all_lat_list2[i], '.-', alpha=0.2, color="red", linewidth=0.5, ms=0.5)
            elif plot == 'danger' and count >= 1:
                ax.plot(self.all_lon_list1[i], self.all_lat_list1[i], '.-', alpha=0.2, color="green", linewidth=0.5, ms=0.5)
                ax.plot(self.all_lon_list2[i], self.all_lat_list2[i], '.-', alpha=0.2, color="red", linewidth=0.5, ms=0.5)

        if plot in {"all", "danger"}:
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(f"SimTrack_{instance}.png", dpi=600)
        else:
            plt.close(fig)
    
        fig = plt.figure(figsize=(4, 2))
        data = np.array(self.min_pass_dst_list)
        weights = np.ones_like(data) / len(data)
        plt.hist(data, bins=24, weights=weights, range=(0, 1.2))
        plt.grid(True)
        plt.xlim(0.0, 1.2)
        plt.xlabel("Minimum Passing Distance [nm]")
        plt.ylabel("Relative Frequency [%]")
        plt.savefig(f"Histgram{instance}.png", dpi=600)
        plt.show()
        
        idxs = np.where(np.array(self.min_pass_dst_list) < 0.2)[0]
        print(f"Min Pass Dist < 0.2nm :  {idxs.shape[0]}")
        print(f"DCPA < 0.2nm & Dst < 2nm : {len(self.danger_idxs)}")


def plot_from_log(sim, name, plot_all=False):
    fig, ax = plt.subplots(figsize=(8, 3.1))
    
    lat_grids = []
    lon_grids = []
    for i in range(-1, 37, 1):
        _, lon = calculate_lat_lon(35.0, 139.95, 90, i*0.2)
        lon_grids.append(lon)
    for i in range(-7, 7, 1):
        lat, _ = calculate_lat_lon(35.0, 139.95, 0, i*0.2)
        lat_grids.append(lat)
    for i in range(len(lat_grids)):
        x = [lon_grids[0], lon_grids[-1]]
        y = [lat_grids[i], lat_grids[i]]
        plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.3, linewidth=0.6)
        if (i % 5 == 2):
            plt.text(lon_grids[0], lat_grids[i], round(i * 0.2 - 1.4))
            plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.9, linewidth=0.6)
    for i in range(len(lon_grids)):
        x = [lon_grids[i], lon_grids[i]]
        y = [lat_grids[0], lat_grids[-1]]
        plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.3, linewidth=0.6)
        if (i % 5 == 1):
            plt.text(lon_grids[i], lat_grids[0], int(i * 0.2 - 0.2))
            plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.9, linewidth=0.6)
            
    plt.scatter(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0], alpha=0.6, color="green", s=25)  # Initial Point
    plt.scatter(sim.all_lon_list2[0][0], sim.all_lat_list2[0][0], alpha=0.6, color="red", s=25)    # Initial Point
    # plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "Uniform linear motion", color='green');
    plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "MASS", color='green')
    plt.text(sim.all_lon_list2[0][0] - 1.2/60, sim.all_lat_list2[0][0] - 0.6/60, "Target Model", color='red')

    if plot_all:
        for i in range(len(sim.all_lon_list1)):
            plt.plot(sim.all_lon_list1[i], sim.all_lat_list1[i], '.-', alpha=0.3, color="green", linewidth=0.5, ms=0.5)
            plt.plot(sim.all_lon_list2[i], sim.all_lat_list2[i], '.-', alpha=0.3, color="red", linewidth=0.5, ms=0.5)
    else:
        for i in sim.danger_idxs:
            plt.plot(sim.all_lon_list1[i], sim.all_lat_list1[i], '.-', alpha=0.3, color="green", linewidth=0.5, ms=0.5)
            plt.plot(sim.all_lon_list2[i], sim.all_lat_list2[i], '.-', alpha=0.3, color="red", linewidth=0.5, ms=0.5)
    
    
    ax.set_aspect(1/np.cos(np.deg2rad(35.0)))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("X [nm]")
    plt.ylabel("Y [nm]")
    plt.tight_layout()
    plt.savefig(f"SimTrack_{name}.png", dpi=600)
    plt.show()


def plot_from_log_extracted(sim, name, plot_idxs):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    lat_grids = []
    lon_grids = []
    for i in range(-1, 32, 1):  # 37->7nm
        _, lon = calculate_lat_lon(35.0, 139.95, 90, i*0.2)
        lon_grids.append(lon)
    for i in range(-8, 8, 1):
        lat, _ = calculate_lat_lon(35.0, 139.95, 0, i*0.2)
        lat_grids.append(lat)
    for i in range(len(lat_grids)):
        x = [lon_grids[0], lon_grids[-1]]
        y = [lat_grids[i], lat_grids[i]]
        plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.3, linewidth=0.6)
        if (i % 5 == 3):
            plt.text(lon_grids[0], lat_grids[i], round(i * 0.2 - 1.4))
            plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.9, linewidth=0.6)
    for i in range(len(lon_grids)):
        x = [lon_grids[i], lon_grids[i]]
        y = [lat_grids[0], lat_grids[-1]]
        plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.3, linewidth=0.6)
        if (i % 5 == 1):
            plt.text(lon_grids[i], lat_grids[0], int(i * 0.2 - 0.2))
            plt.plot(x, y, linestyle="dotted", color="gray", alpha=0.9, linewidth=0.6)
            
    plt.scatter(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0], alpha=0.6, color="green", s=25)  # Initial Point
    plt.scatter(sim.all_lon_list2[0][0], sim.all_lat_list2[0][0], alpha=0.6, color="red", s=25)    # Initial Point
    # plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "Uniform linear motion", color='green');
    if sim.own_type == ShipType.Constant:
        plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "Uniform Linear Motion", color='green')
    elif sim.own_type == ShipType.ANS:
        plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "CA Algorithm", color='green')
    else:
        plt.text(sim.all_lon_list1[0][0], sim.all_lat_list1[0][0] - 0.6/60, "Target Model", color='green')
    if sim.tgt_type == ShipType.Constant:
        plt.text(sim.all_lon_list2[0][0] - 1.2/60, sim.all_lat_list2[0][0] - 0.6/60, "Uniform Linear Motion", color='red')
    elif sim.tgt_type == ShipType.ANS:
        plt.text(sim.all_lon_list2[0][0] - 1.2/60, sim.all_lat_list2[0][0] - 0.6/60, "CA Algorithm", color='red')
    else: 
        plt.text(sim.all_lon_list2[0][0] - 1.2/60, sim.all_lat_list2[0][0] - 0.6/60, "Target Model", color='red')

    ALPHA, LINEWIDTH = get_alpha_linewidth(len(plot_idxs))

    for plot_idx in plot_idxs:
        plt.plot(sim.all_lon_list1[plot_idx], sim.all_lat_list1[plot_idx], '.-', alpha=ALPHA, color="green", linewidth=LINEWIDTH, ms=0.5)
        plt.plot(sim.all_lon_list2[plot_idx], sim.all_lat_list2[plot_idx], '.-', alpha=ALPHA, color="red", linewidth=LINEWIDTH, ms=0.5)
    
    ax.set_aspect(1/np.cos(np.deg2rad(35.0)))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("X [nm]")
    plt.ylabel("Y [nm]")
    plt.tight_layout()
    plt.savefig(f"SimTrack_{name}.png", dpi=500)
    plt.show()

    plt.figure(figsize=(4, 3))
    for plot_idx in plot_idxs:
        plt.plot(sim.all_dst_list[plot_idx], sim.all_x_cp_list[plot_idx], '-', linewidth=LINEWIDTH, color="blue", alpha=ALPHA)
    plt.ylim(-1.6, 1.6)
    plt.xlim(0.6, 6.0)
    plt.grid(True)
    plt.xlabel("Relative Distance [nm]")
    plt.ylabel("PPC Y-coordinate [nm]")
    plt.tight_layout()
    plt.savefig('ppc_plot_' + name + '.png', dpi=300)


def get_alpha_linewidth(num):
    if (num <= 3): return 1.0, 1.8
    if (num <= 10): return 0.7, 1.0
    elif (num <= 50): return 0.5, 0.5
    elif (num <= 100): return 0.3, 0.3
    elif (num <= 500): return 0.2, 0.2
    elif (num <= 1000): return 0.1, 0.1
    else: return 0.05, 0.05
        
        


#TGT_DST=3.0, TGT_AZI=93.82,
#TGT_DST=4.0, TGT_AZI=92.87,
#TGT_DST=5.0, TGT_AZI=92.29,
#TGT_DST=6.0, TGT_AZI=91.91,
#TGT_DST=7.0, TGT_AZI=91.64,

