"""
main.py
Author: Jacob Lu
Date: 2022.8.28
"""

from workspace import MarkowitzOptimizationWorkspace


if __name__ == '__main__':
    mow = MarkowitzOptimizationWorkspace()
    mow.initialize_port()
    mow.optimize()
    mow.backcasting()
    # mow.write_port()
