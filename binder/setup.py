#!/usr/bin/env python3
import os, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root)
from setup import main
os.chdir(root)
main()
