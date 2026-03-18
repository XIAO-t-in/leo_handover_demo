"""
LEO Satellite Environment
基于Starlink的低轨卫星仿真环境

A simulation environment for Low Earth Orbit (LEO) satellite constellations,
inspired by the Starlink architecture, supporting satellite handover research.
"""

from .satellite import Satellite
from .constellation import Constellation, StarLinkShellConfig, STARLINK_SHELLS
from .ground_station import GroundStation
from .channel import LinkChannel, LinkQuality
from .handover import HandoverEvent, HandoverManager

__all__ = [
    "Satellite",
    "Constellation",
    "StarLinkShellConfig",
    "STARLINK_SHELLS",
    "GroundStation",
    "LinkChannel",
    "LinkQuality",
    "HandoverEvent",
    "HandoverManager",
]
