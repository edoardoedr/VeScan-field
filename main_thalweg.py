#!/usr/bin/env python3
"""VeScan thalweg pipeline (centerlines from Mike Halle's thalweg) from the haversack ranked store:
one patient or a batch, everything set in the config.

    python main_thalweg.py --config configs/thalweg/example.json [--structures artery veins]

See README.md (section VeScan-thalweg) and vescan_thalweg/main.py.
"""
from vescan_thalweg.main import main

if __name__ == "__main__":
    main()
