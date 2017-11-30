// redWaveTools.hpp - This file is part of the pyredwave toolbox.
// This software aims at performing redundant wavelet transformations.
// Copyright 2014 CEA
// Contributor : Jeremy Rapin (jeremy.rapin.math@gmail.com)
// Created on 14/7/2014, last modified on 15/12/2014
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



#ifndef _RED_WAVE_TOOLBOX_HPP_
#define _RED_WAVE_TOOLBOX_HPP_

//libraries
#include <math.h>
#include <string.h>
#include <stdio.h>

//math macros
#define max(A, B) (A > B ? A : B)
#define ABS(x) (x<0?-x:x)
#define isint(x) ((x - floor(x)) > 0.0 ? 0 : 1)

//tensor dimension limit
#define MAXNDIMDATA 8

//PARALLELIZATION
#define __PARALLELIZED__ //remove this line for unparallelized functions
//if defined, the compilation will need specific settings
//in Matlab, this means modifying the mexopts.bat file


#ifdef __PARALLELIZED__
#include <omp.h>
#endif

// MATLAB
#ifdef RW_MATLAB_INTERFACE
#include "mex.h"
#include "matrix.h"
#define RW_PRINT mexPrintf
#else
#define RW_PRINT printf
#endif



// Python
#ifdef RW_PYTHON_INTERFACE
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <stdexcept>
#include <sstream>
namespace bp = boost::python;
namespace np = boost::numpy;
#endif


/************************
 *   vectorData Class   *
 ************************/


template <class T> class vectorData {
public:
    vectorData():ptr(NULL), step(0), length(0), _owner(0){;}
    vectorData(const vectorData &data){RW_PRINT("Copy constructor not implemented\n");}
    ~vectorData(){if (_owner){delete[] ptr;};}
    
    inline T& at(const int &ind) { return ptr[ind * step]; }
    inline const T get(const int &ind) const { return ptr[ind * step]; }
    inline const int getLength() const { return length; }
    inline T& operator[](const int &ind) {return ptr[ind*step];}
    
	
	
	vectorData(const int &vector_length, T* pointer = NULL, const int &increment_step = 1): ptr(pointer), step(increment_step), length(vector_length) {
		if (ptr==NULL) {
			ptr=new T[vector_length]; //(T *)malloc(vector_length*sizeof(T)); 
			_owner=1;
		}
		else
			_owner=0;
	}
	
	vectorData<T> operator=(const vectorData<T>& other) {
		if (_owner) {
			delete[] ptr;
			RW_PRINT("BEWARE: Freeing memory during copy.\n");
		}
		
		ptr=other.ptr;
		step=other.step;
		length=other.length;
		
		_owner=0;
		if (other._owner!=0)
			RW_PRINT("BEWARE: Copying owner vectorData is dangerous.\n");
		return vectorData(length, ptr, step);
	}
	
	
	vectorData<T> getSubvector(const int &offset, const int &new_length){
		if (offset + new_length > length)
		{
			RW_PRINT("ERROR: subvector is not contained in the initial vector.\n");
			return vectorData(length, ptr, step);
		}
		else
			return vectorData(new_length, ptr + offset * step, step);
	}
    
    T* ptr;
    int step;
    int length;
private:
    bool _owner;
};





class NdimDataOperations;
class NdimData;
// MATLAB
#ifdef RW_MATLAB_INTERFACE
const NdimData matlabArrayToNdimData(const mxArray *array);
#endif
// Python
#ifdef RW_PYTHON_INTERFACE
NdimData narray_to_ndimdata(const np::ndarray& array);
#endif






/**********************
 *   NdimData Class   *
 **********************/


class NdimData{
    friend class NdimDataOperations;

public:
    NdimData();
    NdimData(const NdimData &data){RW_PRINT("Copy constructor not implemented\n");} 
    NdimData& operator=(const NdimData& other);
    NdimData(const int &Ndim, const int* sizes, double* pointer = NULL, const int* steps = NULL);
    NdimData(const int &size0, const int &size1, double* pointer = NULL);

    ~NdimData(){if (_owner){delete[] _ptr;};}
    
    const int getNumVectors(const int &face){return _numel/_sizes[face];};
    vectorData<double> getVector(const int &face, const int &ind);
    const vectorData<double> getVector(const int &face, const int &ind) const;
    
    int getNumSlices(const vectorData<int> &faceDims) const;
    NdimData get_slice(const vectorData<int> &faceDims, const int &ind);
    const NdimData get_slice(const vectorData<int> &faceDims, const int &ind) const;
    NdimData getSubdata(const int &dimension, const int &offset,  const int &new_length);
    
    const int getSize(const int &dimension) const {return _sizes[dimension];}
    const int* getSizes() const {return _sizes;}
    const int* getSteps() const {return _steps;}
    const int getNdim() const {return _Ndim;}
    const int getNumel() const {return _numel;}
    double* getPtr(){return _ptr;}
    
    void printInfo(const char* str) const;
    
    //for convenience, but dangerous for non-contiguous data
    inline const double get(const int &ind) const { return _ptr[ind]; }
    inline double& operator[](const int &ind) {return _ptr[ind];}
    
private:
    double* _ptr;
    int _Ndim;
    int _owner;
    int _numel;
    int _sizes[MAXNDIMDATA];
    int _steps[MAXNDIMDATA];
    
};


#define MAX_OPERATION_NUMBER 8
class NdimDataOperations{
    friend inline NdimDataOperations& operator*(const double &scalar, NdimDataOperations &ope);
    
    
public:
    NdimDataOperations():_num_ope(0){;}
    NdimDataOperations(const double &scalar,const NdimData &data):_num_ope(1){_scalars[0]=scalar;_data_ptrs[0] = &data;}
    
    NdimDataOperations(const NdimDataOperations &ope); 
    
    NdimData& computeTo(NdimData& output);
    NdimData& LinfProjectTo(NdimData& output, const NdimData& inf_bounds, const double &tau = 1.0);
    NdimData& LinfProjectTo(NdimData& output, const NdimDataOperations &inf_bounds){if(inf_bounds._num_ope>1){RW_PRINT("Only first component of inf_bounds is used.\n");}LinfProjectTo(output,*inf_bounds._data_ptrs[0],inf_bounds._scalars[0]);};
    NdimData& negativePartTo(NdimData& output);
    NdimData& positivePartTo(NdimData& output);
    
    void add(const double &scalar,const NdimData &data);
    
    inline NdimDataOperations& operator+(const NdimData& data){this->add( 1, data); return *this; };
    inline NdimDataOperations& operator-(const NdimData& data){this->add(-1, data); return *this; };
    NdimDataOperations& operator+(NdimDataOperations ope);
    inline NdimDataOperations& operator-(NdimDataOperations ope){return *this+(-1.0)*ope;}
    
    
   
private:
    int _num_ope;
    double _scalars[MAX_OPERATION_NUMBER];
    const NdimData* _data_ptrs[MAX_OPERATION_NUMBER];

};

//scalar to operator
inline NdimDataOperations& operator*(const double &scalar, NdimDataOperations& ope){for (int k = 0; k<ope._num_ope; ++k){ope._scalars[k]*=scalar;}return ope;}
// data/scalor to data
inline NdimDataOperations operator+(const NdimData& data1, const NdimData& data2){NdimDataOperations ope(1.0,data1); ope.add( 1,data2); return ope;}
inline NdimDataOperations operator-(const NdimData& data1, const NdimData& data2){NdimDataOperations ope(1.0,data1); ope.add(-1,data2); return ope;}
inline NdimDataOperations operator*(const double &scalar, const NdimData& data){return NdimDataOperations(scalar,data);}
//affects to
inline NdimData& operator<<(NdimData& data,NdimDataOperations &ope){ope.computeTo(data);return data;}






class RedWave{
public:
    RedWave(const vectorData<int> &waveDims, const vectorData<double> &filter, const int &L, const bool &isometric = 0);
    ~RedWave();

    int setSizes(const NdimData &x, const int &directDomain = 1, const bool &use_internal_wx = 1);
    const int* get_x_size() const {return _x_sizes;}
    const int* get_wx_size() const {return _wx_sizes;}
    bool check_dimensions(const NdimData* x, const NdimData* wx);
    
    int wavelets(NdimData &x, NdimData &wx,  const int &forward);
    int analysisInversion(NdimData &x, const NdimData &AtA, const NdimData &AtY, const NdimData &thresholds, const int &num_iter, const bool nonNegative = true);
    int analysisThresholding(NdimData &input, NdimData &output, const NdimData &thresholds, const int &num_iter);
    
private:
    
    const int* get_internal_wx_size() const;
    const int* get_internal_wx_steps() const;
    const vectorData<int>& get_wave_dimensions() const {return *_wave_dims;}
    inline int get_thread_id() const;
    NdimData get_current_x_slice();
    NdimData get_current_wx_slice();
    
    NdimData& sliceWavelets_1D(NdimData &x, NdimData &wx, const int &forward);
    NdimData& sliceWavelets_2D(NdimData &x, NdimData &wx, const int &forward);
    
    //handles
    inline NdimData& sliceForward(NdimData &x, NdimData &wx){return (_wave_dims->getLength() > 1 ? sliceWavelets_2D(x, wx, 1) : sliceWavelets_1D(x, wx, 1));}
    inline NdimData& sliceBackward(NdimData &x, NdimData &wx){return (_wave_dims->getLength() > 1 ? sliceWavelets_2D(x, wx, -1) : sliceWavelets_1D(x, wx, -1));}
    
    inline void forwardConvolution(const vectorData<double> &in, vectorData<double> &low, vectorData<double> &high, const int &scale);
    inline void backwardConvolution(vectorData<double> &out, const vectorData<double> &low, const vectorData<double> &high, const int &scale);
    
    int _L;
    bool _sizes_are_initialized ;
    vectorData<int>* _wave_dims;
    int _x_sizes[MAXNDIMDATA];
    int _wx_sizes[MAXNDIMDATA];
    int _wx_slice_sizes[MAXNDIMDATA];
    NdimData* _inner_x;
    NdimData* _inner_wx;
    NdimData* _temp_data;
    
    vectorData<double>* _hfilt_forw;
    vectorData<double>* _lfilt_forw;
    vectorData<double>* _hfilt_back;
    vectorData<double>* _lfilt_back;
    
    void createMemory(const bool wx_buffer = false);
    int checkConsistency(const NdimData &input, const int &forward);
};



void gaussInversion(NdimData* output,NdimData* input = NULL);//inversion of matrices

#endif


