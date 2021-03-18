option(WITH_AZURE ON)

if(${WITH_AZURE})
  message("-- USING Camera: Linking Azure")
  target_sources(fusion
  PRIVATE
    src/input/azure.cpp
  )

  target_include_directories(fusion
  PUBLIC
    # /usr/local/include/k4a
    /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/include/
  )

  if(NOT ${Pangolin_FOUND})
    target_link_directories(fusion
    PUBLIC
      # /usr/local/bin/libk4a.so
      /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin
      # /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4a.so
      # /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4arecord.so
    )

    target_link_libraries(fusion
    PUBLIC
      /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4a.so
      /home/yohann/SLAMs/cameras/Azure-Kinect-Sensor-SDK/build/bin/libk4arecord.so
    )
  endif()
endif()
