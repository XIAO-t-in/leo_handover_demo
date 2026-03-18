# leo_handover_demo
低轨卫星切换策略研究 – Low Earth Orbit Satellite Handover Strategy Research

A Python simulation environment for Low Earth Orbit (LEO) satellite constellations,
inspired by the Starlink architecture.  It models orbital mechanics, ground-to-satellite
link quality, and satellite handover decisions – providing a foundation for evaluating
handover strategies (including deep-learning-based approaches).

---

## Features

| Module | Description |
|---|---|
| `satellite.py` | Keplerian orbit propagation (two-body model) |
| `constellation.py` | Walker-Delta constellation builder; ships with Starlink shell parameters |
| `ground_station.py` | Ground terminal with elevation-angle visibility model |
| `channel.py` | Free-space path loss, RSRP, and SNR computation |
| `handover.py` | Handover event detection (`best_rsrp`, `hysteresis`, `elevation` strategies) |
| `utils.py` | Coordinate conversions (LLA ↔ ECEF ↔ ECI) and orbital mechanics helpers |

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Run the example simulation

```bash
python examples/basic_simulation.py
```

Sample output:

```
Constellation created: 96 satellites
Ground stations: ['Beijing', 'New_York', 'London', 'Sydney', 'Reykjavik']

Running 60-minute simulation (step = 30 s) …
  t=  1830s  [Beijing]  HO: sh0-p01-s07 → sh0-p00-s04  (RSRP -90.5 → -87.2 dBm)
  ...

─── Simulation Summary ───────────────────────────────────
Total simulation time : 60 min
Total handover events : 17
Handover rate         : 12.00 handovers / hour (all GS combined)
```

---

## Constellation Configuration

The default mini constellation ships two shells:

| Shell | Altitude | Inclination | Planes × Sats | Total |
|---|---|---|---|---|
| 0 | 550 km | 53.0° | 6 × 11 | 66 |
| 1 | 570 km | 70.0° | 3 × 10 | 30 |

The full Starlink reference configuration (`STARLINK_SHELLS`) is also provided:

| Shell | Altitude | Inclination | Planes × Sats | Total |
|---|---|---|---|---|
| 0 | 550 km | 53.0° | 72 × 22 | 1584 |
| 1 | 540 km | 53.2° | 72 × 22 | 1584 |
| 2 | 570 km | 70.0° | 36 × 20 | 720 |
| 3 | 560 km | 97.6° | 6 × 58 | 348 |
| 4 | 560 km | 97.6° | 4 × 43 | 172 |

To use the full constellation:

```python
from leo_env import Constellation, STARLINK_SHELLS

constellation = Constellation(shells=STARLINK_SHELLS)
```

---

## API Overview

```python
from leo_env import (
    Constellation, StarLinkShellConfig,
    GroundStation, LinkChannel, HandoverManager,
)
from leo_env.handover import STRATEGY_HYSTERESIS

# Build constellation
shells = [StarLinkShellConfig(shell_id=0, altitude_km=550.0,
                               inc_deg=53.0, n_planes=6, n_sats=11)]
constellation = Constellation(shells=shells)

# Create a ground station
gs = GroundStation(gs_id="Beijing", lat_deg=39.9, lon_deg=116.4, min_elev_deg=25.0)

# Propagate to t=0 and query visible satellites
constellation.propagate(0.0)
visible = gs.visible_satellites(constellation.satellites)

# Evaluate link quality
channel = LinkChannel()
best = channel.best_link(constellation.satellites, gs)
print(best)  # LinkQuality(...)

# Run handover manager
mgr = HandoverManager([gs], channel=channel,
                       strategy=STRATEGY_HYSTERESIS, hysteresis_db=3.0)
for t in range(0, 3600, 30):
    constellation.propagate(float(t))
    events = mgr.step(float(t), constellation.satellites)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/
```

---

## Project Structure

```
leo_handover_demo/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── leo_env/
│       ├── __init__.py
│       ├── satellite.py        # Satellite + Keplerian propagator
│       ├── constellation.py    # Constellation builder (Starlink-like)
│       ├── ground_station.py   # Ground terminal + visibility
│       ├── channel.py          # Link quality model (FSPL, RSRP, SNR)
│       ├── handover.py         # Handover detection & event system
│       └── utils.py            # Coordinate conversions & orbital mechanics
├── tests/
│   ├── test_utils.py
│   ├── test_satellite.py
│   └── test_handover.py
└── examples/
    └── basic_simulation.py
```
