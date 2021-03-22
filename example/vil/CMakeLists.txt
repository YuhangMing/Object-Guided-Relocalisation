project(vil_example)

add_executable(vil_const const_main.cpp)

target_link_libraries(vil_const
PRIVATE
    fusion
)

