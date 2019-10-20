#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::array_t<double> calc_iou_mtx(py::array_t<bool> input1, py::array_t<bool> input2) {
    auto r1 = input1.unchecked<2>(), r2 = input2.unchecked<2>();

    if (r1.ndim() != 2 || r2.ndim() != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (r1.shape(1) != r2.shape(1))
        throw std::runtime_error("Size of shape 1 must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(std::vector<ptrdiff_t>{r1.shape(0), r2.shape(0)});

    auto res_mut = result.mutable_unchecked<2>();

    for (ssize_t i = 0; i < r1.shape(0); i++)
        for (ssize_t j = 0; j < r2.shape(0); j++) {
            long int_count = 0;
            long union_count = 0;
            for (ssize_t k = 0; k < r1.shape(1); k++) {
                int_count += r1(i,k) && r2(j,k);
                union_count += r1(i,k) || r2(j,k);
            }
        if (union_count == 0) {
            res_mut(i,j) = 0;
            //std::cout << "union count was zero..."<<std::endl;
        } else {
            res_mut(i,j) = int_count/(double)union_count;
            //std::cout << "Calculated res mute: "<<res_mut(i,j)<<std::endl;
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_iou_mtx", &calc_iou_mtx, "Calculates the iou matrix");
}