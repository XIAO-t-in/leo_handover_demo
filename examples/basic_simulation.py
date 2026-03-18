"""
Basic LEO Satellite Simulation Example
基于Starlink的低轨卫星环境仿真示例

This script demonstrates how to:
1. Create a Starlink-like LEO constellation
2. Set up ground stations
3. Run a simulation with handover detection
4. Summarise results
"""

from leo_env import (
    Constellation,
    StarLinkShellConfig,
    GroundStation,
    LinkChannel,
    HandoverManager,
)
from leo_env.handover import STRATEGY_HYSTERESIS


# ---------------------------------------------------------------------------
# 1. Define a compact Starlink-inspired constellation
# ---------------------------------------------------------------------------

shells = [
    # First shell: 550 km, 53° inclination – 6 planes × 11 sats (66 total)
    StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0,
                        n_planes=6, n_sats=11),
    # Second shell: 570 km, 70° inclination for higher-latitude coverage
    StarLinkShellConfig(shell_id=1, altitude_km=570.0, inc_deg=70.0,
                        n_planes=3, n_sats=10),
]

constellation = Constellation(shells=shells)
print(f"Constellation created: {constellation.total_satellites} satellites")

# ---------------------------------------------------------------------------
# 2. Define ground stations (user terminals)
# ---------------------------------------------------------------------------

ground_stations = [
    GroundStation(gs_id="Beijing",    lat_deg=39.9,  lon_deg=116.4, min_elev_deg=25.0),
    GroundStation(gs_id="New_York",   lat_deg=40.7,  lon_deg=-74.0, min_elev_deg=25.0),
    GroundStation(gs_id="London",     lat_deg=51.5,  lon_deg=-0.1,  min_elev_deg=25.0),
    GroundStation(gs_id="Sydney",     lat_deg=-33.9, lon_deg=151.2, min_elev_deg=25.0),
    GroundStation(gs_id="Reykjavik",  lat_deg=64.1,  lon_deg=-21.9, min_elev_deg=25.0),
]

print(f"Ground stations: {[gs.gs_id for gs in ground_stations]}")

# ---------------------------------------------------------------------------
# 3. Create channel model and handover manager
# ---------------------------------------------------------------------------

channel = LinkChannel(
    freq_hz=12.0e9,          # Ku-band downlink ~12 GHz
    tx_power_dbm=40.0,       # 10 W satellite transmit power
    tx_gain_dbi=34.0,        # Satellite antenna gain
    rx_gain_dbi=20.0,        # User terminal receive gain
    misc_loss_db=3.0,
    noise_dbm=-104.0,
)

handover_manager = HandoverManager(
    ground_stations=ground_stations,
    channel=channel,
    strategy=STRATEGY_HYSTERESIS,
    hysteresis_db=3.0,        # 3 dB RSRP gain required to trigger handover
)

# ---------------------------------------------------------------------------
# 4. Run 1-hour simulation with 30-second time steps
# ---------------------------------------------------------------------------

SIM_DURATION_S = 3600          # 1 hour
TIME_STEP_S = 30               # 30-second steps

print(f"\nRunning {SIM_DURATION_S // 60}-minute simulation "
      f"(step = {TIME_STEP_S} s) …")

t = 0.0
while t <= SIM_DURATION_S:
    constellation.propagate(t)
    events = handover_manager.step(t, constellation.satellites)
    for ev in events:
        if ev.reason not in ("initial_association",):
            print(f"  t={t:6.0f}s  [{ev.gs_id}]  "
                  f"HO: {ev.from_sat_id} → {ev.to_sat_id}  "
                  f"(RSRP {ev.from_rsrp_dbm:.1f} → {ev.to_rsrp_dbm:.1f} dBm)")
    t += TIME_STEP_S

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

print("\n─── Simulation Summary ───────────────────────────────────")
print(f"Total simulation time : {SIM_DURATION_S // 60} min")
print(f"Total handover events : {handover_manager.total_handovers}")
ho_rate = handover_manager.handover_rate(SIM_DURATION_S)
print(f"Handover rate         : {ho_rate * 3600:.2f} handovers / hour (all GS combined)")

print("\nPer ground station summary:")
for gs in ground_stations:
    serving = handover_manager.serving_satellite(gs.gs_id)
    n_ho = sum(
        1 for e in handover_manager.events
        if e.gs_id == gs.gs_id
        and e.reason not in ("initial_association", "serving_satellite_out_of_view")
    )
    serving_info = serving.sat_id if serving else "(none)"
    print(f"  {gs.gs_id:<12}  handovers={n_ho:3d}  serving={serving_info}")

print("─────────────────────────────────────────────────────────")

# ---------------------------------------------------------------------------
# 6. Show current link quality for each ground station
# ---------------------------------------------------------------------------

print("\nLink quality at end of simulation (t = {} s):".format(int(SIM_DURATION_S)))
for gs in ground_stations:
    serving = handover_manager.serving_satellite(gs.gs_id)
    if serving is not None:
        lq = channel.evaluate(serving, gs)
        print(f"  {gs.gs_id:<12}  sat={serving.sat_id:<20}  "
              f"elev={lq.elevation_deg:5.1f}°  "
              f"dist={lq.distance_km:6.0f} km  "
              f"rsrp={lq.rsrp_dbm:6.1f} dBm  "
              f"snr={lq.snr_db:5.1f} dB")
    else:
        print(f"  {gs.gs_id:<12}  (no visible satellite)")
