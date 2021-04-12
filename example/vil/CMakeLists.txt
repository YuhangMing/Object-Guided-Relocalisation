project(vil_example)

add_executable(vil main.cpp)

target_link_libraries(vil
PRIVATE
    fusion
)

add_executable(map visualiseMapReg.cpp)

target_link_libraries(map
PRIVATE
    fusion
)

