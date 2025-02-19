#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// Forward declaration of the init function from cylinder_fitting.cpp
void init_cylinder_fitting(py::module& m);

PYBIND11_MODULE(pypcl_algorithms, m) {
    m.doc() = R"pbdoc(
        Python bindings for PCL algorithms
        
        This module provides Python bindings for various Point Cloud Library (PCL) 
        algorithms, making them easily accessible from Python.
    )pbdoc";

    // Initialize cylinder fitting module
    init_cylinder_fitting(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
