import math
import numpy as np


def calculate_distance_and_bearing(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    mid_lat = (lat1_rad + lat2_rad) / 2

    x = delta_lon * np.cos(mid_lat)
    y = lat2_rad - lat1_rad

    distance_km = R * np.sqrt(x**2 + y**2)
    distance_nm = distance_km * 0.539957  # Convert to nautical miles

    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)

    bearing_deg = np.where(bearing_deg < 0, bearing_deg + 360, bearing_deg)

    return distance_nm, bearing_deg


def calculate_bearing_from_xy(x, y):
    """
    Calculate the bearing (azimuth) from the origin to the point (x, y).
    Bearing is measured in degrees clockwise from north.

    Parameters:
        x (float): X-coordinate
        y (float): Y-coordinate

    Returns:
        float: Bearing in degrees
    """
    if x == 0 and y == 0:
        return None

    angle_radians = math.atan2(x, y)

    angle_degrees = math.degrees(angle_radians)

    if angle_degrees < 0:
        angle_degrees += 360

    return angle_degrees


def calculate_lat_lon(lat1, lon1, azimuth, distance):
    """
    Calculate the destination point given a start latitude and longitude, azimuth, and distance.

    :param lat1: Starting latitude in degrees
    :param lon1: Starting longitude in degrees
    :param azimuth: Bearing in degrees (clockwise from north)
    :param distance: Distance to travel in nm
    :return: Destination latitude and longitude in degrees
    """
    # Earth's radius in meters
    R = 6371000

    # Convert distance from nm to meter
    distance *= 1852

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    azimuth = math.radians(azimuth)

    # Calculate the destination latitude and longitude
    lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) +
                     math.cos(lat1) * math.sin(distance / R) * math.cos(azimuth))

    lon2 = lon1 + math.atan2(math.sin(azimuth) * math.sin(distance / R) * math.cos(lat1),
                             math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))

    # Convert latitude and longitude from radians to degrees
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lat2, lon2


def rotate_bearing(target_bearing, base_bearing):
  rotated_bearing = (target_bearing - base_bearing + 360) % 360
  return rotated_bearing


def co_xy_conv(theta):
    theta = 90 - theta
    if isinstance(theta, (int, float)):
        if theta < 0:
            theta += 360
        return theta
    else:
        theta[theta < 0] += 360
        return theta


def az_180_180(az):
    return az + 360 if az < -180 else az - 360 if az >= 180 else az


def az_0_to_360(azi):
    return (azi+360)%360


def is_near_zero(value, tolerance=1e-6):
    return np.abs(value) < tolerance


def cal_dxdy(azi, dst):
    dx = dst*np.sin(np.deg2rad(azi))
    dy = dst*np.cos(np.deg2rad(azi))
    return dx, dy


def get_arctan(dx, dy):
    theta = np.zeros_like(dx)  # Initialize theta with the same shape as dx

    # Use boolean indexing to handle different cases
    near_zero_dx = is_near_zero(dx)
    positive_dy = dy >= 0
    negative_dy = dy < 0

    theta[near_zero_dx & positive_dy] = 90
    theta[near_zero_dx & negative_dy] = 270

    # Calculate theta for other cases using vectorized operations
    non_zero_dx = ~near_zero_dx
    theta[non_zero_dx] = np.rad2deg(np.arctan(dy[non_zero_dx] / dx[non_zero_dx]))

    theta[non_zero_dx & (dx < 0)] += 180
    theta[non_zero_dx & (dx > 0) & (dy < 0)] += 360
    return theta


def calc_rel_motion(own_co, own_spd, tgt_co, tgt_spd, azi, dst):
    own_co = co_xy_conv(own_co)
    tgt_co = co_xy_conv(tgt_co)
    azi = co_xy_conv(azi)

    rel_vx = tgt_spd*np.cos(np.deg2rad(tgt_co)) - own_spd*np.cos(np.deg2rad(own_co))
    rel_vy = tgt_spd*np.sin(np.deg2rad(tgt_co)) - own_spd*np.sin(np.deg2rad(own_co))
    rel_co = get_arctan(rel_vx, rel_vy)
    rel_co = co_xy_conv(rel_co)
    rel_spd = np.sqrt(rel_vx**2 + rel_vy**2)

    # Decide whether the relative motion is approaching or separating.
    dx = np.cos(np.deg2rad(azi)) * dst
    dy = np.sin(np.deg2rad(azi)) * dst
    dot_product = rel_vx * dx + rel_vy * dy

    if isinstance(dot_product, (int, float)):
      if dot_product > 0:
        rel_spd *= -1
    else:
      rel_spd[dot_product > 0] *= -1

    return rel_co, rel_spd


def calc_cpa(own_co, own_spd, tgt_co, tgt_spd, azi, dst):
    rel_co, rel_spd = calc_rel_motion(own_co, own_spd, tgt_co, tgt_spd, azi, dst)

    phi = azi + 180 - rel_co
    dcpa = dst * np.abs(np.sin(np.deg2rad(phi)))

    # Treat very small or separating relative speed as invalid for TCPA.
    rel_spd[rel_spd <= 0.1] = -1

    cpa_dsts = dst * np.cos(np.deg2rad(phi))
    cpa_dsts[cpa_dsts < 0] = 1
    tcpa = cpa_dsts / rel_spd

    dcpa[tcpa < 0] = dst[tcpa < 0]
    tcpa[tcpa < 0] = -1

    return dcpa, tcpa


def calc_ppc(azi, dst, cog, sog, own_cog, own_sog):
    """Calculate the point PPC using the principal-arcsine formula.

    All bearings/courses are degrees clockwise from north in the same
    reference frame; ``azi`` is the target's bearing from own ship.
    Distance is nm and speed is kn. Return (relative bearing in [0, 360],
    distance in nm, time in hours). The relative bearing is measured from
    ``own_cog``; own course does not constrain the collision course.

    Only the principal-arcsine solution specified in my thesis is used,
    even when a second collision course exists. No finite future PPC (or a
    stationary own ship) returns three NaNs. Invalid inputs raise ValueError.
    """
    azi, dst, cog, sog, own_cog, own_sog = map(
        float, (azi, dst, cog, sog, own_cog, own_sog)
    )
    if not all(math.isfinite(v) for v in
               (azi, dst, cog, sog, own_cog, own_sog)):
        raise ValueError("PPC inputs must be finite scalars")
    if dst < 0 or sog < 0 or own_sog < 0:
        raise ValueError("Distance and speeds must be non-negative")
    missing = (math.nan, math.nan, math.nan)
    if own_sog == 0:
        return missing
    # Coincident positions have no unique bearing to a PPC.
    if dst == 0:
        return missing

    azi %= 360.0
    cog %= 360.0
    beta = (sog / own_sog) * math.sin(math.radians(azi - cog))
    # Clip only floating-point roundoff at the arcsine domain boundary.
    if abs(beta) > 1.0 + 1e-12:
        return missing
    beta = max(-1.0, min(1.0, beta))
    ppc_course = azi - math.degrees(math.asin(beta))
    vrx = (sog * math.sin(math.radians(cog))
           - own_sog * math.sin(math.radians(ppc_course)))
    vry = (sog * math.cos(math.radians(cog))
           - own_sog * math.cos(math.radians(ppc_course)))
    relative_speed = math.hypot(vrx, vry)
    speed_tolerance = 1e-12 * max(own_sog, sog)
    closing_speed = -(vrx * math.sin(math.radians(azi))
                      + vry * math.cos(math.radians(azi)))
    # The arcsine-domain condition alone also permits separating motion.
    if relative_speed <= speed_tolerance or closing_speed <= speed_tolerance:
        return missing
    ppc_time = dst / relative_speed
    ppc_distance = own_sog * ppc_time
    return rotate_bearing(ppc_course, own_cog), ppc_distance, ppc_time


def get_collision_point_azi_dst(azi, dst, cog, sog, own_cog, own_sog):
    """Return point PPC (relative bearing, distance), preserving the old API.

    Missing PPC is represented by (NaN, NaN), not an extrapolated target
    position. Callers must check validity, e.g. np.isfinite(cp_dst), before
    using PPC as a model input or a maneuver-planning target.
    """
    cp_azi, cp_dst, _ = calc_ppc(azi, dst, cog, sog, own_cog, own_sog)
    return cp_azi, cp_dst
