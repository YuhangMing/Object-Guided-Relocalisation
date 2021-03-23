cmake_minimum_required(VERSION 2.8)
project(fusion LANGUAGES CXX CUDA)

# defaults build to release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(fusion SHARED "")

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
# find_package(CUDA 8.0 REQUIRED)
# include(cmake/FindOpenNI2.cmake)
# include(cmake/FindAzure.cmake)
message("-- USING Camera: Linking Azure")

# find_package(Python3 COMPONENTS Development)
# message("-- Python Dir: " ${Python3_INCLUDE_DIRS})
# message("-- Python Lib: " ${Python3_LIBRARIES})
# message("-- OpenCV Lib: " ${OpenCV_LIBRARIES})
# message("-- Eigen Lib: " ${EIGEN3_VERSION})
message("-- Cuda Dir: " ${CUDA_INCLUDE_DIRS})

target_include_directories(fusion
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    /home/yohann/anaconda3/envs/NOCS/include/python3.7m
    /home/yohann/anaconda3/envs/NOCS/lib/python3.7/site-packages/numpy/core/include
    /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/include/
)

target_compile_features(fusion
PUBLIC
    cxx_std_11
)

set_target_properties(fusion
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_link_libraries(fusion
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Pangolin_LIBRARIES}
    /home/yohann/anaconda3/envs/NOCS/lib/libpython3.7m.so
    # ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/lib/libg2o_core.so
    # ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/lib/libg2o_solver_dense.so
    # ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/lib/libg2o_types_icp.so
    # ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/lib/libg2o_types_slam3d.so
    libg2o_core.so
    libg2o_solver_dense.so
    libg2o_types_icp.so
    libg2o_types_slam3d.so
    /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4a.so
    /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4arecord.so
    )

target_sources(fusion
PRIVATE
    src/system.cpp
    src/map_manager.cpp

    src/utils/settings.cpp
    
    src/data_struct/rgbd_frame.cpp
    src/data_struct/map_struct.cu
    src/data_struct/map_cuboid.cpp
    src/data_struct/map_object.cpp

    src/tracking/icp_tracker.cpp
    src/tracking/rgbd_odometry.cpp
    src/tracking/device_image.cpp
    src/tracking/cuda_imgproc.cu
    src/tracking/pose_estimator.cu
    src/tracking/build_pyramid.cpp

    src/relocalization/ransac_ao.cpp
    src/relocalization/relocalizer.cpp

    src/visualization/main_window.cpp

    src/voxel_hashing/voxel_hashing.cpp
    src/voxel_hashing/depth_fusion.cu
    src/voxel_hashing/ray_tracing.cu
    src/voxel_hashing/scene_mesh.cu

    src/input/tum_loader.cpp
    src/input/azure.cpp

    src/detection/detector.cpp

    src/optimizer/graph_optimizer.cpp
)

add_subdirectory(example)
include(cmake/InstallGLSL.cmake)
