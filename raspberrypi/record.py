"""
Energy consumption calculator for Raspberry Pi power monitoring.
This module reads power consumption data from a CSV file (watt.csv) containing
timestamp and power measurements. 

It calculates the total energy consumed over a
specific time interval by integrating power readings using the trapezoidal method
(energy = power × time). The script captures an initial line count, waits 10 seconds
for new data to be recorded, then computes the energy delta for all new entries.
"""

import time
timestamp = int(time.time())
start = 0
with open('watt.csv', 'r') as rf:
    start = len(rf.readlines())

time.sleep(10)

lines = None
with open('watt.csv', 'r') as rf:
    lines = rf.readlines()
usable_lines = lines[start:]
prev = None
ans = 0
for line in usable_lines:
    t, p = line.split(',')[:2]
    if prev is not None:
        ans += (int(t) - prev) / 1000 * float(p)
    prev = int(t)

print(f"Total energy: {ans}")