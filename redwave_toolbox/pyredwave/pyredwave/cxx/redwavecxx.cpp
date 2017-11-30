// redwavecxx.cpp - This file is part of the pyredwave toolbox.
// This software aims at performing redundant wavelet transformations.
// Copyright 2014 CEA
// Contributor : Jeremy Rapin (jeremy.rapin.math@gmail.com)
// Created on 15/12/2014, last modified on 15/12/2014
// 
// This software is governed by the CeCILL license under French law and
// abiding by the rules of distribution of free software. You can use,
// modify and/ or redistribute the software under the terms of the CeCILL
// license as circulated by CEA, CNRS and INRIA at the following URL
// "http://www.cecill.info".
// 
// As a counterpart to the access to the source code and rights to copy,
// modify and redistribute granted by the license, users are provided only
// with a limited warranty and the software's author,  the holder of the
// economic rights,  and the successive licensors have only limited
// liability.
// 
// In this respect, the user's attention is drawn to the risks associated
// with loading,  using,  modifying and/or developing or reproducing the
// software by the user in light of its specific status of free software,
// that may mean that it is complicated to manipulate,  and that also
// therefore means that it is reserved for developers and experienced
// professionals having in-depth computer knowledge. Users are therefore
// encouraged to load and test the software's suitability as regards their
// requirements in conditions enabling the security of their systems and/or
// data to be ensured and,  more generally, to use and operate it in the
// same conditions as regards security.
// 
// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL license and that you accept its terms.

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <iostream>
#include <stdexcept>
#include "redWaveTools.hpp"

//#define __PARALLELIZED__

namespace bp = boost::python;
namespace np = boost::numpy;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

using namespace bp;


class pyRedWave
{
public:
    pyRedWave(const np::ndarray& input, int wave_dimension, const np::ndarray& filter, const int number_scales=3, const bool isometric=false);
    pyRedWave(const np::ndarray& input, bp::tuple wave_dimensions, const np::ndarray& filter, const int number_scales=3, const bool isometric=false);
    ~pyRedWave();
    np::ndarray forward(const np::ndarray& input);
    np::ndarray backward(const np::ndarray& input);
    void transform(np::ndarray& x, np::ndarray& wx, const int direction);
    np::ndarray sparse_proximal(const np::ndarray& input, const np::ndarray& lambdas, const int number_of_iterations=24);
    np::ndarray sparse_inversion(const np::ndarray& x0, const np::ndarray& AtA, const np::ndarray& AtY, const np::ndarray& lambdas, const int number_of_iterations=24, const bool non_negative=false);

    //np::ndarray divideByTwo(const np::ndarray& arr);
    //bp::tuple createGridArray(int rows, int cols);
private:
    vectorData<int>* _wave_dims;
    vectorData<double> _filter;
    RedWave* _redwave;
    void _initialize(const np::ndarray& input, const np::ndarray& filter, const int number_scales=3, const bool isometric=false);
    bool _check_dimensions(const NdimData* x, const NdimData* wx);
};


void pyRedWave::transform(np::ndarray& x, np::ndarray& wx, const int direction)
{
    if (this->_redwave)
    {
        NdimData xdata = narray_to_ndimdata(x);
        NdimData wxdata = narray_to_ndimdata(wx);
        this->_redwave->wavelets(xdata, wxdata, 1);
    }
    else
        std::cout << "RedWave was not properly initialized, no computation performed.\n";
}



np::ndarray pyRedWave::forward(const np::ndarray& input)
{
    // get input
    NdimData inputdata = narray_to_ndimdata(input);

    // make output
    if (this->_redwave)
    {
        bp::list shapelist;
        const int* wx_sizes = this->_redwave->get_wx_size();
        for (int i = 0; i < inputdata.getNdim(); ++i)
           shapelist.append(wx_sizes[i]? wx_sizes[i]: inputdata.getSize(i));
        bp::tuple shape = bp::tuple(shapelist);
        np::ndarray output = np::zeros(shape, np::dtype::get_builtin<double>());
        NdimData outputdata = narray_to_ndimdata(output);
    
        this->_redwave->wavelets(inputdata, outputdata, 1);
        return output;
    }
    else
    {
        std::cout << "RedWave was not properly initialized, returning null array.\n";
        return np::zeros(bp::make_tuple(0,0), np::dtype::get_builtin<double>());
    }
}



np::ndarray pyRedWave::backward(const np::ndarray& input)
{
    // get input
    NdimData inputdata = narray_to_ndimdata(input);

    // make output
    if (this->_redwave)
    {
        bp::list shapelist;
        const int* x_sizes = this->_redwave->get_x_size();
        for (int i = 0; i < inputdata.getNdim(); ++i)
           shapelist.append(x_sizes[i]? x_sizes[i]: inputdata.getSize(i));
        bp::tuple shape = bp::tuple(shapelist);
        np::ndarray output = np::zeros(shape, np::dtype::get_builtin<double>());
        NdimData outputdata = narray_to_ndimdata(output);
    
        this->_redwave->wavelets(outputdata, inputdata, -1);
        return output;
    }
    else
    {
        std::cout << "RedWave was not properly initialized, returning null array.\n";
        return np::zeros(bp::make_tuple(0,0), np::dtype::get_builtin<double>());
    }
}



np::ndarray pyRedWave::sparse_proximal(const np::ndarray& input, const np::ndarray& lambdas, const int number_of_iterations)
{
    // get inputs
    NdimData inputdata = narray_to_ndimdata(input);
    NdimData lambdasdata = narray_to_ndimdata(lambdas);

    // make output
    if (this->_redwave)
    {
        bp::list shapelist;
        for (int i = 0; i < inputdata.getNdim(); ++i)
           shapelist.append(inputdata.getSize(i));
        bp::tuple shape = bp::tuple(shapelist);
        np::ndarray output = np::zeros(shape, np::dtype::get_builtin<double>());
        NdimData outputdata = narray_to_ndimdata(output);
    
        this->_redwave->analysisThresholding(inputdata, outputdata, lambdasdata, number_of_iterations);
        return output;
    }
    else
    {
        std::cout << "RedWave was not properly initialized, returning null array.\n";
        return np::zeros(bp::make_tuple(0,0), np::dtype::get_builtin<double>());
    }
}



np::ndarray pyRedWave::sparse_inversion(const np::ndarray& x0, const np::ndarray& AtA, const np::ndarray& AtY, const np::ndarray& lambdas, const int number_of_iterations, const bool non_negative)
{
    // make output
    if (this->_redwave)
    {
        // get inputs
        np::ndarray x = x0.copy();
        NdimData xdata = narray_to_ndimdata(x);
        NdimData lambdasdata = narray_to_ndimdata(lambdas);
        NdimData AtAdata = narray_to_ndimdata(AtA);
        NdimData AtYdata = narray_to_ndimdata(AtY);
    
        this->_redwave->analysisInversion(xdata, AtAdata, AtYdata, lambdasdata, number_of_iterations, non_negative);
        return x;
    }
    else
    {
        std::cout << "RedWave was not properly initialized, returning null array.\n";
        return np::zeros(bp::make_tuple(0,0), np::dtype::get_builtin<double>());
    }
}


pyRedWave::pyRedWave(const np::ndarray& input, const int wave_dimension, const np::ndarray& filter, const int number_scales, const bool isometric)
{
    // set pointers to NULL
    this->_wave_dims = NULL;

    // make dimension vector
    this->_wave_dims = new vectorData<int>(1);
    this->_wave_dims->at(0) = wave_dimension;
    
    // initialization of other variables
    this->_initialize(input, filter, number_scales, isometric);
}



pyRedWave::pyRedWave(const np::ndarray& input, bp::tuple wave_dimensions, const np::ndarray& filter, const int number_scales, const bool isometric)
{
    this->_wave_dims = NULL;
    // make dimension vector
    int num_dims = bp::len(wave_dimensions);
    this->_wave_dims = new vectorData<int>(num_dims);
    for (int k = 0; k < num_dims; ++k)
        this->_wave_dims->at(k) = extract<int>(wave_dimensions[k]);

    // initialization of other variables
    this->_initialize(input, filter, number_scales, isometric);
}



void pyRedWave::_initialize(const np::ndarray& input, const np::ndarray& filter, const int number_scales, const bool isometric)
{
    // initializes all parameters apart from _wave_dims
    // (the constructors deal with the polymorphism)
    this->_redwave = NULL;

    // array
    NdimData inputdata = narray_to_ndimdata(input);

    //get filter
    NdimData filtdata = narray_to_ndimdata(filter);
    this->_filter = filtdata.getVector(0, 0);

    // create the RedWave instance
    this->_redwave = new RedWave(*(this->_wave_dims), this->_filter, number_scales, isometric);
    // initialize internal memory
    this->_redwave->setSizes(inputdata, 1, true);
}


pyRedWave::~pyRedWave()
{
    if (this->_redwave)
        delete this->_redwave;
    if (this->_wave_dims)
        delete this->_wave_dims;
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(sparse_proximal_overloads, pyRedWave::sparse_proximal, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(sparse_inversion_overloads, pyRedWave::sparse_inversion, 4, 6)

BOOST_PYTHON_MODULE(redwavecxx)
{
    np::initialize();
    class_<pyRedWave>("RedWave", bp::init<np::ndarray, int, np::ndarray, int, bool>())
        .def(init<np::ndarray, int, np::ndarray, int>())
        .def(init<np::ndarray, int, np::ndarray>())
        .def(init<np::ndarray, bp::tuple, np::ndarray, int, bool>())
        .def(init<np::ndarray, bp::tuple, np::ndarray, int>())
        .def(init<np::ndarray, bp::tuple, np::ndarray>())
        .def("forward", &pyRedWave::forward)
        .def("backward", &pyRedWave::backward)
        .def("transform", &pyRedWave::transform)
        .def("sparse_proximal", &pyRedWave::sparse_proximal, sparse_proximal_overloads())
        .def("sparse_inversion", &pyRedWave::sparse_inversion, sparse_inversion_overloads())

    ;
}
