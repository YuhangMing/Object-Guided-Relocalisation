project(vil_example)

add_executable(vil_const const_main.cpp)

target_link_libraries(vil_const
PRIVATE
    fusion
)

add_executable(vil_reloc reloc_main.cpp)

target_link_libraries(vil_reloc
PRIVATE
    fusion
)

add_executable(vil_vislz vislz_main.cpp)

target_link_libraries(vil_vislz
PRIVATE
    fusion
)


