################################################################################
# user build settings

SET(TRACE TRUE)
SET(HAVE_G2O FALSE)   #TRUE

SET(DEBUG_OUTPUT TRUE) # Only relevant if build without ROS

################################################################################

SET(PROJECT_NAME svo_edgelete_live)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo , Debug
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Set definitions
IF(TRACE)
  ADD_DEFINITIONS(-DSVO_TRACE)
ENDIF()


IF(DEBUG_OUTPUT)
  ADD_DEFINITIONS(-DSVO_DEBUG_OUTPUT)
ENDIF()

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV 2 REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
#FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)

find_package(Pangolin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        cv_bridge roslib
        cmake_modules
        image_transport
        pcl_ros
        geometry_msgs
        image_transport
        roscpp
        sensor_msgs
        std_msgs
        tf
        )

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES open_quadtree_mapping_node

        EPENDS OpenCV Eigen Boost
        CATKIN_DEPENDS cv_bridge image_transport  roscpp sensor_msgs std_msgs tf

)


#FIND_PACKAGE(fast REQUIRED)
#FIND_PACKAGE(vikit_common REQUIRED)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#sophus
set(Sophus_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Sophus_install/include)
set(Sophus_LIBRARIES  ${PROJECT_SOURCE_DIR}/Sophus_install/lib/libSophus.so)

#opencv
#set(OpenCV_INCLUDE_DIRS /home/lab/opencv2.4_workspace/include)
#set(OpenCV_LIBS  /home/lab/opencv2.4_workspace/lib)
message(${OpenCV_INCLUDE_DIRS})

# Include dirs
INCLUDE_DIRECTORIES(
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}

)

# Set link libraries
LIST(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)


# Set sourcefiles
LIST(APPEND SOURCEFILES
  src/frame_handler_mono.cpp
  src/frame_handler_base.cpp
  src/frame.cpp
  src/point.cpp
  src/map.cpp
  src/pose_optimizer.cpp
  src/initialization.cpp
  src/matcher.cpp
  src/reprojector.cpp
  src/feature_alignment.cpp
  src/feature_detection.cpp
  src/depth_filter.cpp
  src/config.cpp

  src/camera_model.cpp
  src/sparse_align.cpp
  src/debug.cpp
  src/math_utils.cpp
  src/homography.cpp
  src/robust_cost.cpp
  src/fast_10_score.cpp
  src/fast_nonmax_3x3.cpp
  src/fast_10.cpp
  src/faster_corner_10_sse.cpp

  src/slamviewer.cpp
  src/five-point.cpp
  src/precomp.cpp
  src/modelest.cpp
  )

IF(HAVE_G2O)
  ADD_DEFINITIONS(-DUSE_BUNDLE_ADJUSTMENT)
ENDIF()
# Add g2o if available
IF(HAVE_G2O)
  FIND_PACKAGE(G2O REQUIRED)

  message("g2o is found: ${G2O_INCLUDE_DIR}")

  INCLUDE_DIRECTORIES(
    ${G2O_INCLUDE_DIR}
    /usr/include/suitesparse # for cholmod
  )

 #LIST(APPEND LINK_LIBS
 # ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE}
  # ${G2O_SOLVER_PCG} ${G2O_TYPES_SBA} cholmod cxsparse
 #)
  LIST(APPEND LINK_LIBS  g2o_core_d  g2o_solver_csparse_d  g2o_csparse_extension_d g2o_types_sba_d g2o_solver_dense_d g2o_stuff_d g2o_parser_d g2o_solver_pcg_d  cholmod cxsparse )
  LIST(APPEND SOURCEFILES src/bundle_adjustment.cpp)
ENDIF(HAVE_G2O)

# Create svo library
ADD_LIBRARY(svo SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(svo ${LINK_LIBS} ${catkin_LIBRARIES}  ${Pangolin_LIBRARIES})

################################################################################
# TESTS

ADD_EXECUTABLE(test_pipeline test/test_pipeline.cpp)
TARGET_LINK_LIBRARIES(test_pipeline svo)

ADD_EXECUTABLE(svo_pose test/main.cpp)
TARGET_LINK_LIBRARIES(svo_pose svo)

ADD_EXECUTABLE(test_live_vo test/test_live_vo.cpp)
TARGET_LINK_LIBRARIES(test_live_vo svo)
