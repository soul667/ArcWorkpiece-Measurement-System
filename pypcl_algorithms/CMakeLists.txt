cmake_minimum_required(VERSION 3.4...3.18)
project(pypcl_algorithms)

# Find PCL
find_package(PCL REQUIRED)

add_subdirectory(pybind11)

# Include PCL headers
include_directories(${PCL_INCLUDE_DIRS})

pybind11_add_module(pypcl_algorithms 
    src/main.cpp
    src/cylinder_fitting.cpp
)

# Link PCL libraries
target_link_libraries(pypcl_algorithms PRIVATE ${PCL_LIBRARIES})

# Add PCL compile definitions and flags
target_compile_definitions(pypcl_algorithms 
    PRIVATE 
    ${PCL_DEFINITIONS}
    VERSION_INFO="${EXAMPLE_VERSION_INFO}"
)
