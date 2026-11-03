"""Threads of the multithreaded steps (kd-tree queries), set once per process by vescan_field.main.

-1 = every core Python sees. A container can see many more cores than its CPU quota allows (a
Jupyter server: 128 seen, 16 allowed), and with several patients in parallel each process would
start a thread per seen core; main sets the share of the quota of each process instead.
"""
KDTREE_WORKERS = -1
