"""
Tests for utility functions (coordinate conversions, orbital mechanics).
"""

import math
import pytest
import numpy as np

from leo_env.utils import (
    EARTH_RADIUS_KM,
    deg2rad,
    rad2deg,
    lla_to_ecef,
    ecef_to_lla,
    orbital_period,
    mean_motion,
    kepler_equation,
    true_anomaly_from_mean,
    orbit_to_inertial,
    eci_to_ecef,
    elevation_angle,
    distance_km,
)


class TestAngleConversions:
    def test_deg2rad_zero(self):
        assert deg2rad(0) == 0.0

    def test_deg2rad_180(self):
        assert abs(deg2rad(180) - math.pi) < 1e-12

    def test_rad2deg_pi(self):
        assert abs(rad2deg(math.pi) - 180.0) < 1e-10

    def test_roundtrip(self):
        for d in [0, 45, 90, 135, 180, 270, 360]:
            assert abs(rad2deg(deg2rad(d)) - d) < 1e-10


class TestCoordinateConversions:
    def test_lla_to_ecef_equator(self):
        """Point on equator at prime meridian should be on +x axis."""
        pos = lla_to_ecef(0.0, 0.0, 0.0)
        assert abs(pos[0] - EARTH_RADIUS_KM) < 1e-6
        assert abs(pos[1]) < 1e-6
        assert abs(pos[2]) < 1e-6

    def test_lla_to_ecef_north_pole(self):
        """North pole should be on +z axis."""
        pos = lla_to_ecef(90.0, 0.0, 0.0)
        assert abs(pos[0]) < 1e-4
        assert abs(pos[1]) < 1e-4
        assert abs(pos[2] - EARTH_RADIUS_KM) < 1e-4

    def test_lla_roundtrip(self):
        """Convert LLA -> ECEF -> LLA and check round-trip accuracy."""
        lat, lon, alt = 35.0, 116.0, 0.05
        ecef = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(ecef)
        assert abs(lat - lat2) < 1e-4
        assert abs(lon - lon2) < 1e-4
        assert abs(alt - alt2) < 1e-4

    def test_altitude_in_ecef(self):
        """Altitude should increase the ECEF radius."""
        pos_surface = lla_to_ecef(0.0, 0.0, 0.0)
        pos_alt = lla_to_ecef(0.0, 0.0, 550.0)
        r_surface = np.linalg.norm(pos_surface)
        r_alt = np.linalg.norm(pos_alt)
        assert abs(r_alt - r_surface - 550.0) < 1e-4


class TestOrbitalMechanics:
    def test_orbital_period_iss(self):
        """ISS at ~400 km should have ~92-min period."""
        a = EARTH_RADIUS_KM + 400.0
        T = orbital_period(a)
        assert 5400 < T < 5700  # 90-95 minutes

    def test_orbital_period_starlink(self):
        """Starlink at 550 km should have ~95-min period."""
        a = EARTH_RADIUS_KM + 550.0
        T = orbital_period(a)
        assert 5600 < T < 5900

    def test_mean_motion_inverse_period(self):
        """n = 2π / T"""
        a = EARTH_RADIUS_KM + 550.0
        n = mean_motion(a)
        T = orbital_period(a)
        assert abs(n - 2 * math.pi / T) < 1e-12

    def test_kepler_equation_circular(self):
        """For e=0, E should equal M."""
        for M in [0.0, 1.0, 2.5, 5.0]:
            E = kepler_equation(M, 0.0)
            assert abs(E - M) < 1e-9

    def test_true_anomaly_circular(self):
        """For e=0, true anomaly should equal mean anomaly."""
        for M in [0.0, 0.5, 1.5, 3.0]:
            nu = true_anomaly_from_mean(M, 0.0)
            assert abs(nu - M) < 1e-8

    def test_orbit_to_inertial_radius(self):
        """At perigee (nu=0), r = a(1-e)."""
        a = EARTH_RADIUS_KM + 550.0
        e = 0.001
        pos, _ = orbit_to_inertial(a, e, 0.0, 0.0, 0.0, 0.0)
        r = np.linalg.norm(pos)
        expected = a * (1 - e)
        assert abs(r - expected) / expected < 1e-6

    def test_orbit_to_inertial_circular_constant_radius(self):
        """For circular orbit, radius should be constant at any true anomaly."""
        a = EARTH_RADIUS_KM + 550.0
        radii = []
        for nu in [0, 1, 2, 3, 4, 5, 6]:
            pos, _ = orbit_to_inertial(a, 0.0, math.radians(53.0), 0.0, 0.0, float(nu))
            radii.append(np.linalg.norm(pos))
        assert max(radii) - min(radii) < 1e-4  # all equal within 100 m

    def test_eci_to_ecef_at_t0(self):
        """At t=0 the rotation angle is 0, so ECEF == ECI."""
        pos_eci = np.array([7000.0, 0.0, 0.0])
        pos_ecef = eci_to_ecef(pos_eci, 0.0)
        np.testing.assert_allclose(pos_ecef, pos_eci, atol=1e-10)


class TestElevationAngle:
    def test_satellite_directly_overhead(self):
        """Satellite directly above the ground station → 90° elevation."""
        gs = lla_to_ecef(0.0, 0.0, 0.0)
        sat = lla_to_ecef(0.0, 0.0, 550.0)
        el = elevation_angle(gs, sat)
        assert abs(el - 90.0) < 0.1

    def test_satellite_below_horizon(self):
        """Satellite on the opposite side of Earth → negative elevation."""
        gs = lla_to_ecef(0.0, 0.0, 0.0)
        sat = lla_to_ecef(0.0, 180.0, 550.0)
        el = elevation_angle(gs, sat)
        assert el < 0.0

    def test_distance(self):
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([3.0, 4.0, 0.0])
        assert abs(distance_km(pos1, pos2) - 5.0) < 1e-10
