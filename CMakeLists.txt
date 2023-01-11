cmake_minimum_required(VERSION 2.8.3)
project(registration_visualizer)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(PCL REQUIRED)
find_package(Iridescence REQUIRED)

# Enable support for C++17
if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  fast_gicp
)

catkin_package(
  CATKIN_DEPENDS fast_gicp
  INCLUDE_DIRS include
  DEPENDS PCL Iridescence
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Iridescence_INCLUDE_DIRS}
)

file(GLOB SRC src/*.cpp)
add_executable(registration_visualizer ${SRC})
target_link_libraries(registration_visualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Iridescence_LIBRARIES})
