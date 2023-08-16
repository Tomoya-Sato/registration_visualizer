cmake_minimum_required(VERSION 2.8.3)
project(registration_visualizer)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_subdirectory(thirdparty/fast_gicp)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Iridescence REQUIRED)

# Enable support for C++17
if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

file(GLOB SRC src/*.cpp)
add_executable(registration_visualizer ${SRC})
target_include_directories(registration_visualizer PUBLIC
  include
  thirdparty/fast_gicp/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Iridescence_INCLUDE_DIRS}
)
target_link_libraries(registration_visualizer PUBLIC
  fast_gicp
  ${PCL_LIBRARIES}
  ${Iridescence_LIBRARIES}
)
