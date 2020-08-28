cmake_minimum_required(VERSION 2.8.3)
project(basic_navigation)

find_package(catkin REQUIRED
    message_generation
    genmsg
    std_msgs
    geometry_msgs
    nav_msgs
    ropod_sim
)

add_message_files(
    DIRECTORY
        ros/msg
    FILES
        BasicNavigationFeedback.msg
)

catkin_python_setup()

generate_messages(DEPENDENCIES
    std_msgs
    geometry_msgs
    nav_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime std_msgs
)
