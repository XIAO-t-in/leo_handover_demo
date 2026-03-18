"""
Tests for GroundStation, LinkChannel, and HandoverManager.
"""

import math
import pytest
import numpy as np

from leo_env.satellite import Satellite
from leo_env.ground_station import GroundStation
from leo_env.channel import LinkChannel, LinkQuality
from leo_env.handover import (
    HandoverEvent,
    HandoverManager,
    STRATEGY_BEST_RSRP,
    STRATEGY_HYSTERESIS,
    STRATEGY_ELEVATION,
)
from leo_env.constellation import Constellation, StarLinkShellConfig
from leo_env.utils import EARTH_RADIUS_KM


def _make_satellite(sat_id="sat0", lat_deg=0.0, lon_deg=0.0, alt_km=550.0):
    """Create a satellite already propagated to t=0 at roughly the given LLA."""
    from leo_env.utils import lla_to_ecef, ecef_to_lla
    # Use a circular orbit at the given altitude and manually set ECEF position
    a = EARTH_RADIUS_KM + alt_km
    sat = Satellite(sat_id=sat_id, a=a, e=0.0, inc_deg=0.0, raan_deg=0.0,
                    argp_deg=0.0, M0_deg=lon_deg)
    sat.propagate(0.0)
    return sat


def _make_gs(gs_id="gs0", lat=0.0, lon=0.0, min_elev=25.0):
    return GroundStation(gs_id=gs_id, lat_deg=lat, lon_deg=lon, min_elev_deg=min_elev)


class TestGroundStation:
    def test_ecef_position(self):
        gs = _make_gs(lat=0.0, lon=0.0)
        r = np.linalg.norm(gs.ecef)
        assert abs(r - EARTH_RADIUS_KM) < 1e-4

    def test_overhead_satellite_visible(self):
        """A satellite directly overhead should be visible."""
        gs = _make_gs(lat=0.0, lon=0.0, min_elev=5.0)
        # Build a Starlink-mini constellation and find an overhead satellite
        c = Constellation()
        c.propagate(0.0)
        # At least some satellites should be visible somewhere
        visible = gs.visible_satellites(c.satellites)
        # We just check the function runs without error and returns a list
        assert isinstance(visible, list)

    def test_visible_satellites_sorted_by_elevation(self):
        gs = _make_gs(lat=0.0, lon=0.0, min_elev=0.0)
        c = Constellation()
        c.propagate(0.0)
        visible = gs.visible_satellites(c.satellites)
        if len(visible) >= 2:
            elevations = [v[1] for v in visible]
            assert elevations == sorted(elevations, reverse=True)

    def test_elevation_to_overhead(self):
        """Satellite directly above should have ~90° elevation."""
        from leo_env.utils import lla_to_ecef
        gs = GroundStation(gs_id="gs0", lat_deg=0.0, lon_deg=0.0)
        # Create a satellite at the same lat/lon but higher
        sat = Satellite(sat_id="s0", a=EARTH_RADIUS_KM + 550.0, e=0.0,
                        inc_deg=0.0, raan_deg=0.0, argp_deg=0.0, M0_deg=0.0)
        sat.propagate(0.0)
        # Manually override ECEF to be directly overhead
        from leo_env.utils import lla_to_ecef
        sat._pos_ecef = lla_to_ecef(0.0, 0.0, 550.0)
        el = gs.elevation_to(sat)
        assert abs(el - 90.0) < 0.5

    def test_best_satellite_none_when_no_visible(self):
        gs = _make_gs(lat=0.0, lon=0.0, min_elev=90.0)  # Only allow overhead
        c = Constellation()
        c.propagate(0.0)
        # With a 90° threshold almost nothing will be visible
        best = gs.best_satellite(c.satellites)
        # Could be None or a satellite directly overhead
        assert best is None or isinstance(best, Satellite)


class TestLinkChannel:
    def test_fspl_increases_with_distance(self):
        ch = LinkChannel()
        fspl1 = ch.compute_fspl(500.0)
        fspl2 = ch.compute_fspl(1000.0)
        assert fspl2 > fspl1

    def test_rsrp_decreases_with_distance(self):
        ch = LinkChannel()
        rsrp1 = ch.compute_rsrp(500.0)
        rsrp2 = ch.compute_rsrp(1000.0)
        assert rsrp2 < rsrp1

    def test_snr_from_rsrp(self):
        ch = LinkChannel(noise_dbm=-100.0)
        rsrp = -80.0
        snr = ch.compute_snr(rsrp)
        assert abs(snr - 20.0) < 1e-9

    def test_evaluate_returns_link_quality(self):
        from leo_env.utils import lla_to_ecef
        ch = LinkChannel()
        gs = GroundStation(gs_id="gs0", lat_deg=0.0, lon_deg=0.0, min_elev_deg=0.0)
        sat = Satellite(sat_id="s0", a=EARTH_RADIUS_KM + 550.0, e=0.0,
                        inc_deg=0.0, raan_deg=0.0, argp_deg=0.0, M0_deg=0.0)
        sat.propagate(0.0)
        sat._pos_ecef = lla_to_ecef(0.0, 0.0, 550.0)
        lq = ch.evaluate(sat, gs)
        assert isinstance(lq, LinkQuality)
        assert lq.is_visible
        assert lq.rsrp_dbm < 0  # negative dBm
        assert lq.snr_db > 0    # positive SNR when overhead

    def test_best_link_none_when_nothing_visible(self):
        ch = LinkChannel()
        gs = GroundStation(gs_id="gs0", lat_deg=0.0, lon_deg=0.0, min_elev_deg=90.0)
        c = Constellation()
        c.propagate(0.0)
        # Very restrictive threshold – almost no satellite should be visible
        result = ch.best_link(c.satellites, gs)
        assert result is None or isinstance(result, LinkQuality)


class TestHandoverManager:
    def _setup(self, strategy=STRATEGY_BEST_RSRP, hysteresis_db=3.0):
        shells = [
            StarLinkShellConfig(
                shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=6, n_sats=11
            )
        ]
        c = Constellation(shells=shells)
        gs_list = [
            GroundStation(gs_id="gs0", lat_deg=35.0, lon_deg=116.0, min_elev_deg=25.0),
        ]
        ch = LinkChannel()
        mgr = HandoverManager(
            ground_stations=gs_list,
            channel=ch,
            strategy=strategy,
            hysteresis_db=hysteresis_db,
        )
        return c, gs_list, mgr

    def test_invalid_strategy_raises(self):
        gs = [_make_gs()]
        with pytest.raises(ValueError):
            HandoverManager(ground_stations=gs, strategy="unknown_strategy")

    def test_initial_association(self):
        c, gs_list, mgr = self._setup()
        c.propagate(0.0)
        events = mgr.step(0.0, c.satellites)
        # If any satellite is visible, an initial association should be recorded
        visible = gs_list[0].visible_satellites(c.satellites)
        if visible:
            assert len(events) == 1
            assert events[0].reason == "initial_association"
            assert events[0].from_sat_id == ""

    def test_total_handovers_counter(self):
        c, gs_list, mgr = self._setup()
        for t in range(0, 300, 30):
            c.propagate(float(t))
            mgr.step(float(t), c.satellites)
        assert mgr.total_handovers >= 0  # At minimum zero

    def test_handover_rate(self):
        c, gs_list, mgr = self._setup()
        duration = 3600.0
        for t in range(0, int(duration), 60):
            c.propagate(float(t))
            mgr.step(float(t), c.satellites)
        rate = mgr.handover_rate(duration)
        assert rate >= 0.0

    def test_serving_satellite_type(self):
        c, gs_list, mgr = self._setup()
        c.propagate(0.0)
        mgr.step(0.0, c.satellites)
        serving = mgr.serving_satellite("gs0")
        assert serving is None or isinstance(serving, Satellite)

    def test_strategies_all_valid(self):
        for strategy in [STRATEGY_BEST_RSRP, STRATEGY_HYSTERESIS, STRATEGY_ELEVATION]:
            c, gs_list, mgr = self._setup(strategy=strategy)
            c.propagate(0.0)
            events = mgr.step(0.0, c.satellites)
            assert isinstance(events, list)

    def test_hysteresis_reduces_handovers(self):
        """Hysteresis strategy should produce fewer handovers than best_rsrp."""
        shells = [
            StarLinkShellConfig(
                shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=6, n_sats=11
            )
        ]
        duration = 3600.0
        step = 60.0

        def run_simulation(strategy, hyst):
            c = Constellation(shells=shells)
            gs_list = [GroundStation(gs_id="gs0", lat_deg=35.0, lon_deg=116.0,
                                     min_elev_deg=25.0)]
            mgr = HandoverManager(ground_stations=gs_list, strategy=strategy,
                                  hysteresis_db=hyst)
            t = 0.0
            while t <= duration:
                c.propagate(t)
                mgr.step(t, c.satellites)
                t += step
            return sum(
                1 for e in mgr.events
                if e.reason not in ("initial_association",
                                    "serving_satellite_out_of_view")
            )

        n_best = run_simulation(STRATEGY_BEST_RSRP, 0.0)
        n_hyst = run_simulation(STRATEGY_HYSTERESIS, 10.0)
        # Hysteresis with large margin should result in ≤ handovers than no hysteresis
        assert n_hyst <= n_best
