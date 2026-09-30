import sys
import os

# Ensure src is at the top of sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Remove root_dir from sys.path if present, so root/nerva cannot shadow src/nerva
while root_dir in sys.path:
    sys.path.remove(root_dir)
