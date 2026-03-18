"""
Tests for Satellite and Constellation classes.
"""

import math
import pytest
import numpy as np

from leo_env.satellite import Satellite
from leo_env.constellation import (
    Constellation,
    StarLinkShellConfig,
    STARLINK_SHELLS,
    STARLINK_SHELLS_MINI,
)
from leo_env.utils import EARTH_RADIUS_KM, orbital_period


class TestSatellite:
    def _make_sat(self, **kwargs) -> Satellite:
        defaults = dict(
            sat_id="test-sat",
            a=EARTH_RADIUS_KM + 550.0,
            e=0.0,
            inc_deg=53.0,
            raan_deg=0.0,
            argp_deg=0.0,
            M0_deg=0.0,
        )
        defaults.update(kwargs)
        return Satellite(**defaults)

    def test_altitude(self):
        sat = self._make_sat()
        assert abs(sat.altitude_km - 550.0) < 1e-6

    def test_period(self):
        sat = self._make_sat()
        T = sat.period_s
        assert 5600 < T < 5900  # ~95 min for 550 km

    def test_propagate_returns_self(self):
        sat = self._make_sat()
        result = sat.propagate(0.0)
        assert result is sat

    def test_propagate_initial_position(self):
        """At t=0, M=M0=0, so satellite is at perigee."""
        sat = self._make_sat(inc_deg=0.0, raan_deg=0.0, argp_deg=0.0, M0_deg=0.0)
        sat.propagate(0.0)
        r = np.linalg.norm(sat.pos_eci)
        expected = EARTH_RADIUS_KM + 550.0
        assert abs(r - expected) / expected < 1e-5

    def test_propagate_constant_radius(self):
        """For a circular orbit, radius should remain constant over time."""
        sat = self._make_sat()
        radii = []
        T = sat.period_s
        for frac in [0.0, 0.25, 0.5, 0.75]:
            sat.propagate(frac * T)
            radii.append(np.linalg.norm(sat.pos_eci))
        assert max(radii) - min(radii) < 1.0  # within 1 km

    def test_propagate_one_period(self):
        """After one orbital period, satellite should be at the same ECI position."""
        sat = self._make_sat()
        sat.propagate(0.0)
        pos_start = sat.pos_eci.copy()
        T = sat.period_s
        sat.propagate(T)
        pos_end = sat.pos_eci.copy()
        dist = np.linalg.norm(pos_end - pos_start)
        assert dist < 10.0  # within 10 km (numerical integration error)

    def test_lat_lon_alt(self):
        sat = self._make_sat()
        sat.propagate(0.0)
        lat, lon, alt = sat.lat_lon_alt
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert 500 < alt < 600  # roughly 550 km


class TestStarLinkShellConfig:
    def test_total_sats(self):
        shell = StarLinkShellConfig(
            shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=72, n_sats=22
        )
        assert shell.total_sats == 72 * 22

    def test_semi_major_axis(self):
        shell = StarLinkShellConfig(
            shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=6, n_sats=11
        )
        assert abs(shell.semi_major_axis_km - (EARTH_RADIUS_KM + 550.0)) < 1e-6


class TestConstellation:
    def test_default_total_sats(self):
        c = Constellation()
        expected = sum(s.n_planes * s.n_sats for s in STARLINK_SHELLS_MINI)
        assert c.total_satellites == expected

    def test_custom_shells(self):
        shells = [
            StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=3, n_sats=5)
        ]
        c = Constellation(shells=shells)
        assert c.total_satellites == 15

    def test_satellite_ids_unique(self):
        c = Constellation()
        ids = [s.sat_id for s in c.satellites]
        assert len(ids) == len(set(ids))

    def test_get_shell(self):
        shells = [
            StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=2, n_sats=3),
            StarLinkShellConfig(shell_id=1, altitude_km=570.0, inc_deg=70.0, n_planes=2, n_sats=3),
        ]
        c = Constellation(shells=shells)
        shell0 = c.get_shell(0)
        shell1 = c.get_shell(1)
        assert len(shell0) == 6
        assert len(shell1) == 6
        assert all(s.shell_id == 0 for s in shell0)
        assert all(s.shell_id == 1 for s in shell1)

    def test_get_plane(self):
        shells = [
            StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=3, n_sats=4)
        ]
        c = Constellation(shells=shells)
        plane = c.get_plane(0, 1)
        assert len(plane) == 4
        assert all(s.plane_id == 1 for s in plane)

    def test_propagate_all(self):
        c = Constellation()
        c.propagate(100.0)
        for sat in c.satellites:
            # After propagation, positions should be non-zero
            assert np.linalg.norm(sat.pos_ecef) > 0.0

    def test_starlink_shells_config(self):
        """Verify the full Starlink shell list has sensible parameters."""
        assert len(STARLINK_SHELLS) == 5
        total = sum(s.total_sats for s in STARLINK_SHELLS)
        assert total > 3000  # Starlink Gen1 has >4000 sats

    def test_raan_spacing(self):
        """Satellites in different planes should have different RAANs."""
        shells = [
            StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=4, n_sats=2)
        ]
        c = Constellation(shells=shells)
        raans = sorted(set(s.raan_deg for s in c.satellites))
        assert len(raans) == 4  # 4 distinct RAANs
        expected_step = 360.0 / 4
        for i in range(1, len(raans)):
            assert abs(raans[i] - raans[i - 1] - expected_step) < 1e-8
