"""
Timestamped run-directory management.

Creates a ``runs/<YYYY-MM-DD_HH-MM-SS_testname>/`` directory for every
simulation run and ``os.chdir`` into it so that all relative output paths
(``dataQW/``, ``fields/``, ``output/``) land inside that directory.

A ``runs/latest`` symlink always points to the most recent run.

Input files (``params/``, ``DC.txt``) are copied into the run directory
so the simulation can find them **and** so there is a permanent record
of which parameters produced each data set.
"""
