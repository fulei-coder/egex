cmake_minimum_required(VERSION 3.8)
project(uarm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Install Python executables
install(PROGRAMS
  scripts/cam_pub.py
  scripts/episode_recorder.py
  scripts/Follower_Arm/xarm/xarm_pub.py
  scripts/Follower_Arm/xarm/servo2xarm.py
  scripts/Follower_Arm/Dobot/servo2Dobot.py
  scripts/Uarm_teleop/servo_reader.py
  scripts/Uarm_teleop/servo_reader_fixed.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
