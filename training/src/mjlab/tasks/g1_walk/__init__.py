"""Walking, standing and the stand-to-crawl transition for the braced 29-DoF G1.

Same robot as the crawl task's ``robot29.py`` (hands removed, wrist pitch/yaw
braced, printed forearm extensions), started upright. Tasks:

* ``Mjlab-Walk-Flat-Unitree-G1-29dof``: omnidirectional velocity tracking and
  standing still, hardened by the same robustness curriculum as the crawl.
* ``Mjlab-StandToCrawl-Unitree-G1-29dof``: a slow, scheduled transition from
  standing into the crawl task's initial pose, after which the crawl policy
  takes over (``scripts/play_switcher.py``).
"""
