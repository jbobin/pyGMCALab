// Copyright (c) 2014-2015, CosmicPy Developers
// Licensed under CeCILL 2.1 - see LICENSE.rst
#pragma once

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <stdexcept>
#include <sstream>

namespace bp = boost::python;
namespace np = boost::numpy;

// Helper class for fast access to array elements
template<typename T> class NumPyArrayData
{
	char* m_data;
	const Py_intptr_t* m_strides;
    
public:
	NumPyArrayData<T>(const np::ndarray &arr)
	{
		np::dtype dtype = arr.get_dtype();
		np::dtype dtype_expected = np::dtype::get_builtin<T>();
        
		if (dtype != dtype_expected)
		{
			std::stringstream ss;
			ss << "NumPyArrayData: Unexpected data type (" << bp::extract<const char*>(dtype.attr("__str__")()) << ") received. ";
			ss << "Expected " << bp::extract<const char*>(dtype_expected.attr("__str__")());
			throw std::runtime_error(ss.str().c_str());
		}
		
		m_data = arr.get_data();
		m_strides = arr.get_strides();
	}
    
	T* data()
	{
		return reinterpret_cast<T*>(m_data);
	}
    
	const Py_intptr_t* strides()
	{
		return m_strides;
	}
    
	// 1D array access
	inline T& operator()(int i)
	{
		return *reinterpret_cast<T*>(m_data + i*m_strides[0]);
	}
    
	// 2D array access
	inline T& operator()(int i, int j)
	{
		return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1]);
	}
    
	// 3D array access
	inline T& operator()(int i, int j, int k)
	{
		return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1] + k*m_strides[2]);
	}
    
	// 4D array access
	inline T& operator()(int i, int j, int k, int l)
	{
		return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1] + k*m_strides[2] + l*m_strides[3]);
	}
};