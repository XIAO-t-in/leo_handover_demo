"""
Coordinate conversion and orbital mechanics utilities.
"""

import math
import numpy as np


# Earth physical constants
EARTH_RADIUS_KM = 6371.0       # mean radius in km
EARTH_MU = 3.986004418e5       # gravitational parameter, km^3/s^2
EARTH_OMEGA = 7.2921150e-5     # rotation rate, rad/s
J2 = 1.08262668e-3             # second zonal harmonic coefficient


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """
    Convert geodetic (lat, lon, alt) to Earth-Centered Earth-Fixed (ECEF) coordinates.

    Parameters
    ----------
    lat_deg : float  – geodetic latitude in degrees
    lon_deg : float  – geodetic longitude in degrees
    alt_km  : float  – altitude above mean sea level in km

    Returns
    -------
    np.ndarray shape (3,)  – [x, y, z] in km
    """
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z])


def ecef_to_lla(pos_km: np.ndarray):
    """
    Convert ECEF position (km) to geodetic latitude, longitude, altitude.

    Returns
    -------
    (lat_deg, lon_deg, alt_km)
    """
    x, y, z = pos_km
    r = math.sqrt(x**2 + y**2 + z**2)
    lat = rad2deg(math.asin(z / r))
    lon = rad2deg(math.atan2(y, x))
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt


def orbital_period(semi_major_axis_km: float) -> float:
    """
    Compute Keplerian orbital period in seconds.
    """
    return 2.0 * math.pi * math.sqrt(semi_major_axis_km**3 / EARTH_MU)


def mean_motion(semi_major_axis_km: float) -> float:
    """
    Compute mean motion n (rad/s) for a circular orbit.
    """
    return math.sqrt(EARTH_MU / semi_major_axis_km**3)


def kepler_equation(M: float, e: float, tol: float = 1e-10) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E using
    Newton-Raphson iteration.

    Parameters
    ----------
    M   : mean anomaly (rad)
    e   : eccentricity
    tol : convergence tolerance

    Returns
    -------
    E : eccentric anomaly (rad)
    """
    E = M if e < 0.8 else math.pi
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def true_anomaly_from_mean(M: float, e: float) -> float:
    """
    Convert mean anomaly (rad) to true anomaly (rad).
    """
    E = kepler_equation(M, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )
    return nu


def orbit_to_inertial(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: float,
) -> tuple:
    """
    Compute ECI (Earth-Centered Inertial) position and velocity from
    Keplerian orbital elements.

    Parameters
    ----------
    a    : semi-major axis (km)
    e    : eccentricity
    inc  : inclination (rad)
    raan : right ascension of ascending node (rad)
    argp : argument of perigee (rad)
    nu   : true anomaly (rad)

    Returns
    -------
    (pos_eci, vel_eci)  each np.ndarray of shape (3,)  [km, km/s]
    """
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * math.cos(nu))

    # Position in perifocal frame
    rx = r * math.cos(nu)
    ry = r * math.sin(nu)

    sqrt_mu_p = math.sqrt(EARTH_MU / p)
    vx = -sqrt_mu_p * math.sin(nu)
    vy = sqrt_mu_p * (e + math.cos(nu))

    # Rotation matrices: argp, inc, raan
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(argp), math.sin(argp)

    # R = Rz(-raan) * Rx(-inc) * Rz(-argp)
    R = np.array([
        [
            cos_o * cos_w - sin_o * sin_w * cos_i,
            -cos_o * sin_w - sin_o * cos_w * cos_i,
            sin_o * sin_i,
        ],
        [
            sin_o * cos_w + cos_o * sin_w * cos_i,
            -sin_o * sin_w + cos_o * cos_w * cos_i,
            -cos_o * sin_i,
        ],
        [sin_w * sin_i, cos_w * sin_i, cos_i],
    ])

    pos_eci = R @ np.array([rx, ry, 0.0])
    vel_eci = R @ np.array([vx, vy, 0.0])

    return pos_eci, vel_eci


def eci_to_ecef(pos_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Rotate ECI coordinates to ECEF by Earth's rotation angle at time t (seconds
    from epoch).  Uses a simple single-rotation approximation (J2000 sidereal
    time ignored; suitable for short simulations).
    """
    theta = EARTH_OMEGA * t
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([
        [cos_t,  sin_t, 0.0],
        [-sin_t, cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ])
    return R @ pos_eci


def elevation_angle(gs_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """
    Compute elevation angle (degrees) from a ground station to a satellite,
    both positions in ECEF (km).

    Returns
    -------
    elevation_deg : float  (negative means below horizon)
    """
    diff = sat_ecef - gs_ecef
    gs_norm = gs_ecef / np.linalg.norm(gs_ecef)
    diff_norm = np.linalg.norm(diff)
    if diff_norm == 0:
        return 90.0
    cos_nadir = np.dot(diff, gs_norm) / diff_norm
    # elevation is complementary to the angle between diff and nadir direction
    elev_rad = math.asin(min(1.0, max(-1.0, cos_nadir)))
    return rad2deg(elev_rad)


def distance_km(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Euclidean distance between two ECEF positions (km)."""
    return float(np.linalg.norm(pos1 - pos2))
