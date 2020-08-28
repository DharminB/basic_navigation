uint8 UNDEFINED = 0
uint8 SUCCESS = 1
uint8 REACHED_WP = 2
uint8 CANCELLED = 3
uint8 FAILURE_OBSTACLES = 4
uint8 FAILURE_NO_CURRENT_POSE = 5
uint8 FAILURE_EMPTY_PLAN = 6
uint8 FAILURE_INVALID_PLAN = 7

uint8 status
geometry_msgs/Pose reached_wp
uint16 remaining_path_length
