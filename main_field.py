#!/usr/bin/env python3
"""VeScan field (VMTK-free) pipeline from the haversack ranked store: one patient or a batch,
everything set in the config.

    python3 main_field.py --config configs/field/example.json [--structures artery veins]

See README.md and vescan_field/main.py.
"""
from vescan_field.main import main

if __name__ == "__main__":
    main()
