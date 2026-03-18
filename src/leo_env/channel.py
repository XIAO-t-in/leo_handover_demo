"""
Channel model – satellite-to-ground link quality estimation.

Models the free-space path loss (FSPL) and a simplified received signal power
to give a Reference Signal Received Power (RSRP)-like metric.  This is
intentionally simple so that it can be swapped for more detailed models
(e.g. rain fade, beam-gain patterns) without changing the rest of the codebase.

Physical model
--------------
Free-Space Path Loss (dB):
    FSPL = 20·log10(d) + 20·log10(f) + 20·log10(4π/c)
         = 20·log10(d_km · 1000) + 20·log10(f_hz) - 147.55

Received power (dBm):
    P_rx = P_tx_dBm + G_tx_dBi + G_rx_dBi - FSPL - L_misc

where L_misc captures atmospheric, polarisation, and other losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .satellite import Satellite
from .ground_station import GroundStation
from .utils import distance_km, elevation_angle


# Speed of light (m/s)
_C = 2.99792458e8

# Starlink downlink Ku-band centre frequency (Hz) – approximately 12 GHz
_DEFAULT_FREQ_HZ = 12.0e9

# Satellite transmit power (dBm) – simplified flat value
_DEFAULT_TX_POWER_DBM = 40.0      # 10 W EIRP contribution

# Satellite antenna gain (dBi) – simplified isotropic equivalent
_DEFAULT_TX_GAIN_DBI = 34.0

# User-terminal receive gain (dBi)
_DEFAULT_RX_GAIN_DBI = 20.0

# Miscellaneous losses: atmospheric, rain, pointing, etc.  (dB)
_DEFAULT_MISC_LOSS_DB = 3.0

# Noise figure + thermal noise floor at receiver (dBm, for reference SNR)
_DEFAULT_NOISE_DBM = -104.0       # ~-104 dBm at 20 MHz BW, 10 dB NF


@dataclass
class LinkQuality:
    """
    Summary of a single satellite↔ground-station link quality estimate.

    Attributes
    ----------
    satellite    : the satellite in the link
    ground_station : the ground station
    elevation_deg: elevation angle (degrees)
    distance_km  : slant range (km)
    fspl_db      : free-space path loss (dB)
    rsrp_dbm     : received signal reference power (dBm)
    snr_db       : estimated signal-to-noise ratio (dB)
    is_visible   : True if the elevation is above the GS minimum threshold
    """

    satellite: Satellite
    ground_station: GroundStation
    elevation_deg: float
    distance_km: float
    fspl_db: float
    rsrp_dbm: float
    snr_db: float
    is_visible: bool

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LinkQuality(sat={self.satellite.sat_id!r}, "
            f"elev={self.elevation_deg:.1f}°, "
            f"dist={self.distance_km:.0f} km, "
            f"rsrp={self.rsrp_dbm:.1f} dBm, "
            f"snr={self.snr_db:.1f} dB)"
        )


class LinkChannel:
    """
    Computes link quality metrics between satellites and ground stations.

    Parameters
    ----------
    freq_hz        : carrier frequency in Hz
    tx_power_dbm   : satellite transmit power (dBm)
    tx_gain_dbi    : satellite antenna gain (dBi)
    rx_gain_dbi    : ground-terminal receive antenna gain (dBi)
    misc_loss_db   : miscellaneous losses (dB)
    noise_dbm      : receiver noise floor (dBm)
    """

    def __init__(
        self,
        freq_hz: float = _DEFAULT_FREQ_HZ,
        tx_power_dbm: float = _DEFAULT_TX_POWER_DBM,
        tx_gain_dbi: float = _DEFAULT_TX_GAIN_DBI,
        rx_gain_dbi: float = _DEFAULT_RX_GAIN_DBI,
        misc_loss_db: float = _DEFAULT_MISC_LOSS_DB,
        noise_dbm: float = _DEFAULT_NOISE_DBM,
    ):
        self.freq_hz = freq_hz
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_dbi = tx_gain_dbi
        self.rx_gain_dbi = rx_gain_dbi
        self.misc_loss_db = misc_loss_db
        self.noise_dbm = noise_dbm

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def compute_fspl(self, dist_km: float) -> float:
        """
        Free-space path loss in dB.

        Parameters
        ----------
        dist_km : slant range in km
        """
        dist_m = dist_km * 1000.0
        if dist_m <= 0:
            return 0.0
        fspl = (
            20.0 * math.log10(dist_m)
            + 20.0 * math.log10(self.freq_hz)
            + 20.0 * math.log10(4.0 * math.pi / _C)
        )
        return fspl

    def compute_rsrp(self, dist_km: float) -> float:
        """
        Received signal power (dBm) at the given slant range.
        """
        fspl = self.compute_fspl(dist_km)
        rsrp = (
            self.tx_power_dbm
            + self.tx_gain_dbi
            + self.rx_gain_dbi
            - fspl
            - self.misc_loss_db
        )
        return rsrp

    def compute_snr(self, rsrp_dbm: float) -> float:
        """SNR (dB) given the received power."""
        return rsrp_dbm - self.noise_dbm

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def evaluate(self, sat: Satellite, gs: GroundStation) -> LinkQuality:
        """
        Evaluate full link quality from *sat* to *gs*.

        Returns a LinkQuality object with all metrics filled in.
        Both the satellite and ground station must already have been propagated
        to the same simulation time before calling this method.
        """
        dist = distance_km(sat.pos_ecef, gs.ecef)
        elev = elevation_angle(gs.ecef, sat.pos_ecef)
        fspl = self.compute_fspl(dist)
        rsrp = self.compute_rsrp(dist)
        snr = self.compute_snr(rsrp)
        visible = elev >= gs.min_elev_deg

        return LinkQuality(
            satellite=sat,
            ground_station=gs,
            elevation_deg=elev,
            distance_km=dist,
            fspl_db=fspl,
            rsrp_dbm=rsrp,
            snr_db=snr,
            is_visible=visible,
        )

    def best_link(
        self,
        satellites: list,
        gs: GroundStation,
    ) -> Optional[LinkQuality]:
        """
        Evaluate all satellites and return the link with the highest RSRP
        among visible ones, or None if no satellite is visible.
        """
        best: Optional[LinkQuality] = None
        for sat in satellites:
            lq = self.evaluate(sat, gs)
            if lq.is_visible:
                if best is None or lq.rsrp_dbm > best.rsrp_dbm:
                    best = lq
        return best
