#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>


using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::MatrixXd svd(Eigen::Ref<RowMatrixXd> mat) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // std::cout << svd.singularValues() << std::endl;
    return svd.matrixU();
}

Eigen::VectorXd lsq(Eigen::Ref<RowMatrixXd> mat, Eigen::VectorXd b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // std::cout << svd.singularValues() << std::endl;
    return svd.solve(b);
}

Eigen::VectorXd ridge(Eigen::Ref<RowMatrixXd> mat, Eigen::VectorXd b, double alpha) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get singular values.
    Eigen::VectorXd s = svd.singularValues();
    // std::cout << "inverse ingular_value:\n" << 1.0/s.array() << std::endl;

    // Get singular values grater than 1e-15
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> indices = (s.array() > 1e-15);
    int survived_singular_value_num = indices.cast<int>().sum();
    Eigen::VectorXd s_nnz = s.topRows(survived_singular_value_num);

    // Initialize d matrix which become relaxed inverse singular value vector.
    Eigen::VectorXd d = Eigen::VectorXd::Zero(s.size());
    // relaxed inverse singular value vector d e,g.
    // 	    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d.head(survived_singular_value_num) = s_nnz.array() / (s_nnz.array().pow(2.0) + alpha).array();
    // std::cout << "alpha: " << alpha << "\n relaxed inverse siugular values:\n" << d << std::endl;

    // compute lsq coefficient using psuedo inverse matrix of x.
    return svd.matrixV() * d.asDiagonal() * (svd.matrixU().adjoint()) * b;
}

namespace py = pybind11;
PYBIND11_PLUGIN(_svd_lsq) {
    py::module m("_svd_lsq", "lsq of svd using eigen and pybind11");

    m.def("_svd", &svd,
        py::return_value_policy::reference_internal,
        "A function do svd");

    m.def("_lsq", &lsq,
        py::return_value_policy::reference_internal,
        "solve least square");

    m.def("_ridge", &ridge,
        py::return_value_policy::reference_internal,
        "ridge");

    return m.ptr();

}
