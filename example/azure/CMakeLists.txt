project(azure_example)

add_executable(azure main.cpp)

target_link_libraries(azure
PRIVATE
    fusion
)
