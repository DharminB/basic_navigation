#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
   packages=['basic_navigation'],
   package_dir={'basic_navigation': 'ros/src/basic_navigation'}
)

setup(**d)
