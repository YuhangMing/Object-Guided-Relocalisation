project(tum_example)

add_executable(tum main.cpp)

target_link_libraries(tum
PRIVATE
    fusion
)