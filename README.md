# Basic Navigation
There are 2 ways to use this package.

1) Use the package without global planner. This will make the robot move in a
straight line from its current position to given goal position. The robot
rotates in place to align to the straight line, follows the line to reach the
goal position and finally rotates in place to reach goal orientation.

2) Use the package with global planner. The robot will try to get a global plan
from its current position to given goal. Once a plan is received, it breaks up
the plan into a set of straight lines. It then follows those straight lines
(loosely) to reach goal. It will replan if it fail, thus indirectly capable of
working in dynamic environments.
