"""
File to start the jupyter notebooks in this directory
"""

import os
from subprocess import call


def main():
    """ Spawn a jupyter notebook kernel in this folder """
    call("jupyter notebook", cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    main()
