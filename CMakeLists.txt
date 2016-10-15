#####################################
cmake_minimum_required (VERSION 3.0) 
project (RPiDashcam)
add_compile_options(-std=c++11)
#include_directories ("${PROJECT_SOURCE_DIR}/LaneDetectLibraries")
#add_subdirectory (LaneDetectLibraries)
#set (EXTRA_LIBS ${EXTRA_LIBS} LaneDetectLibraries)
add_library(LaneDetectLibraries lane_detect_processor.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#troubleshooting
#message( STATUS "OPENCV DIRECTORY: " ${OpenCV_INCLUDE_DIRS})
#message( STATUS "OPENCV LIBRARIES: " ${OpenCV_LIBS})
add_executable (main main.cpp)
target_link_libraries(main
		${OpenCV_LIBS}
		LaneDetectLibraries)
#####################################
