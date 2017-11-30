// redWaveTools.cpp - This file is part of the pyredwave toolbox.
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


#include "redWaveTools.hpp"





/*********************
 *   redWave Class   *
 *********************/


bool RedWave::check_dimensions(const NdimData* x, const NdimData* wx)
{
    
    if (!_sizes_are_initialized)
    {
        RW_PRINT("C CODE ERROR: Sizes must be initialized before using wavelets");
        return false;
    }

    bool is_ok = true;
    if (x)
    {
        const int* dimensions = this->get_x_size();
        for (int k = 0; k < _wave_dims->getLength(); ++k)
        {
            int ind = _wave_dims->get(k);
            if (dimensions[ind] != x->getSize(ind))
                is_ok = false;
        }
    }
    if (wx)
    {
        // check wavelet dimensions for wx
        const int* dimensions = this->get_wx_size();
        for (int k = 0; k < _wave_dims->getLength(); ++k)
        {
            int ind = _wave_dims->get(k);
            if (dimensions[ind] != wx->getSize(ind))
                is_ok = false;
        }
        // check if sizes are coherent
        if (x)
            for (int k = 0; k < MAXNDIMDATA; ++k)
                if (!dimensions[k])
                    if (wx->getSize(k) != x->getSize(k))
                        is_ok = false;
    }

    //error message
    if (!is_ok)
        RW_PRINT("Error: operation not performed because dimensions are incoherent.\n");

    return is_ok;
}




RedWave::RedWave(const vectorData<int> &waveDims, const vectorData<double> &filter, const int &L, const bool &isometric)
{
    
    ////////////////////////
    //    copying data    //
    ////////////////////////
    
    //_memory     = NULL;
    _inner_x    = NULL;
    _inner_wx   = NULL;
    _temp_data  = NULL;
    _hfilt_back = NULL;
    _lfilt_back = NULL;
    _hfilt_forw = NULL;
    _lfilt_forw = NULL;
    _wave_dims  = NULL;
    
    for (int k = 0; k<MAXNDIMDATA ;  ++k)
    {
        _x_sizes[k] = 0;
        _wx_sizes[k] = 0;
        //_memory_pattern[k] = NULL;
    }
    //_memory_pattern[MAXNDIMDATA] = NULL;
    
    _L = L;
    _sizes_are_initialized  = false;


    ///////////////////////////////////////////////////////////
    // dimensions on which to perform the wavelet transforms //
    ///////////////////////////////////////////////////////////
    
    _wave_dims = new vectorData<int>(waveDims.getLength());
    for (int k = 0; k<waveDims.getLength() ;  ++k)
        _wave_dims->at(k) = waveDims.get(k);
    
    
    
    
    
    ////////////////////////////////////////
    //    lowpass and highpass filters    //
    ////////////////////////////////////////
    ///////////////////
    //forward filters//
    _hfilt_forw = new vectorData<double>(filter.getLength());
    _lfilt_forw = new vectorData<double>(filter.getLength());
    
    int forward = 1;
    //lowpass and highpass
    double normalization = (isometric ? sqrt(2.0) : (forward > 0 ? 1 : 2));
    for (int k = 0; k < filter.getLength(); k++){
        _hfilt_forw->at(k) = filter.get(filter.getLength()-k-1)/normalization;
        _lfilt_forw->at(k) = filter.get(k)/normalization;
    }
    for (int k = 1; k < filter.getLength(); k += 2)
        _hfilt_forw->at(k) = -_hfilt_forw->at(k);
    
    ////////////////////
    //backward filters//
    _hfilt_back = new vectorData<double>(filter.getLength());
    _lfilt_back = new vectorData<double>(filter.getLength());
    forward = -1;
    normalization = (isometric ? sqrt(2.0) : (forward > 0 ? 1 : 2));
    //lowpass and highpass
    for (int k=0; k<filter.getLength(); k++){
        _hfilt_back->at(k) = filter.get(filter.getLength() - k - 1) / normalization;
        _lfilt_back->at(k) = filter.get(k) / normalization;
    }
    for (int k=1; k<filter.getLength(); k += 2)
        _hfilt_back->at(k) = -_hfilt_back->at(k);
    
}





int RedWave::setSizes(const NdimData &x, const int &directDomain, const bool &use_internal_wx)
{
    _sizes_are_initialized = true;
    
    //check wavelet dimensions
    int checkWaveDims[MAXNDIMDATA];
    for (int k = 0; k < MAXNDIMDATA; ++k)     
        checkWaveDims[k] = 0;
    
    for (int k = 0; k < _wave_dims->getLength(); ++k)
        if ((_wave_dims->at(k) < 0) || (_wave_dims->at(k) >= MAXNDIMDATA))
        {
            RW_PRINT("ERROR: Wavelets can only be performed on dimensions 1 to %i.\n", MAXNDIMDATA);
            return 10;
        }
        else
        {
            checkWaveDims[_wave_dims->at(k)] += 1;
            //RW_PRINT("dimension %i.\n",(int)waveDims.at(k));
        }
    for (int k=0;k<MAXNDIMDATA;++k)
        if (checkWaveDims[k]>1) {
            RW_PRINT("ERROR: cannot perform more one wavelet transform on dimension %i.\n", k+1);
            return 10;
        }
    
    
    //check consistency of input
    int error = this->checkConsistency(x, directDomain);
    if (error)
    	return error;
    
    
//    if (_wave_dims->getLength() == 1)
//        if (directDomain > 0)
//            _wx_sizes[] *= _L + 1;
//        else
//            _x_sizes[_wave_dims->get(0)] /= _L + 1;
//    else
//        
//            int ind = _wave_dims->get(k);
//            if (directDomain > 0)
//                _wx_sizes[ind] =  x.getSize(k) * 2 * (k == 0 ? _L : 1);
//            else
//                _x_sizes[ind] =  x.getSize(k) / (2 * (k == 0 ? _L : 1));

    int num_dims = _wave_dims->getLength();
    for (int k = 0; k < num_dims; ++k)
    {
        int ind = _wave_dims->get(k);
        int scales = (num_dims == 1? _L + 1: 2 * (k == 0 ? _L : 1));
        if (directDomain > 0)
        {
            _wx_sizes[ind] =  x.getSize(ind) * scales;
            _x_sizes[ind] =  x.getSize(ind);
        }
        else
        {
            _x_sizes[ind] =  x.getSize(ind) / scales;
            _wx_sizes[ind] =  x.getSize(ind);
        }
    }
    // RW_PRINT("x_sizes: %i, %i, %i, %i\n",_x_sizes[0],_x_sizes[1],_x_sizes[2],_x_sizes[3]);
    // RW_PRINT("wx_sizes: %i, %i, %i, %i\n",_wx_sizes[0],_wx_sizes[1],_wx_sizes[2],_wx_sizes[3]);
    
    //memory
    createMemory(use_internal_wx);
    return 0;
}





NdimData& RedWave::sliceWavelets_1D(NdimData &x, NdimData &wx, const int &forward)
{
    
    int temp_slice = this->get_thread_id();
    int dim = _wave_dims->get(0);
    vectorData<double> x_vect = x.getVector(dim, 0);
    vectorData<double> wx_vect = wx.getVector(dim, 0);
    
    int len = x_vect.length;
    vectorData<double> low_vect = wx_vect.getSubvector(0, len);
    vectorData<double> temp_vect = _temp_data->getVector(dim, temp_slice).getSubvector(0, len);
    
    // use a temp vector in order to avoid writing on currently used data
    vectorData<double>* low[2];
    low[1] = &temp_vect;
    
    if (forward > 0)
    {
        // at the end, write into output data
        low[0] = &low_vect;
        
        for (int L = 1; L <= _L; ++L)//loop on all slices
        {
            vectorData<double> high = wx_vect.getSubvector(L * len, len);

            //launch forward or backward wavelets
            forwardConvolution((L == 1 ? x_vect: *low[(L + _L + 1) % 2]), *low[(L + _L) % 2], high, L);
        }
    }
    else
    {
         // at the end, write into output data
        low[0] = &x_vect;

        for (int L = _L; L > 0; --L)//loop on all slices
        {
            vectorData<double> high = wx_vect.getSubvector(L * len, len);

            //launch forward or backward wavelets
            backwardConvolution(*low[(L + 1) % 2], (L == _L ? low_vect : *low[L % 2]), high, L);
        }
    }
    return (forward > 0) ? wx : x;
}



NdimData& RedWave::sliceWavelets_2D(NdimData &x, NdimData &wx, const int &forward)
{
    if (_wave_dims->getLength() > 2)
        RW_PRINT("ERROR: Only wavelets up to 2 dimensions are implemented.\n");
    else
    {
        int dim0 = _wave_dims->get(0);
        int len0 = x.getSize(dim0);
        int dim1 = _wave_dims->get(1);
        int len1 = x.getSize(dim1);
        
        NdimData temp_slice = _temp_data->get_slice(*_wave_dims, get_thread_id());
        NdimData in, low, high, out;
        vectorData<double> in_v, low_v, high_v, out_v;
        
        //x.printInfo("x");
        if (forward > 0)
            for (int L = 1; L <= _L; ++L)//loop on all slices
            {
                
                // DIMENSIONS 1 //
                in = (L == 1 ? x.getSubdata(dim0, 0, len0) : wx.getSubdata(dim0, 2 * (_L - L + 1) * len0, len0).getSubdata(dim1, 0, len1));
                //in = (L == 1 ? x : wx.getSubdata(dim0, 2 * (_L - L + 1) * len0, len0).getSubdata(dim1, 0, len1));
                low = temp_slice.getSubdata(dim0, 0, len0);
                high = temp_slice.getSubdata(dim0, len0, len0);
                   
                int num_slices = in.getNumVectors(dim0);
                for (int k = 0; k < num_slices; ++k)
                {
                    in_v = in.getVector(dim0, k);
                    low_v = low.getVector(dim0, k);
                    high_v = high.getVector(dim0, k);
                    forwardConvolution(in_v, low_v, high_v, L);
                }
                
                // DIMENSIONS 2 //
                in = temp_slice.getSubdata(dim0, 0, 2 * len0);
                low = wx.getSubdata(dim0, 2 * (_L - L) * len0, 2 * len0).getSubdata(dim1, 0, len1);
                high = wx.getSubdata(dim0, 2 * (_L - L) * len0, 2 * len0).getSubdata(dim1, len1, len1);
                
                
                num_slices = in.getNumVectors(dim1);
                for (int k = 0; k < num_slices; ++k)
                {
                    in_v = in.getVector(dim1, k);
                    low_v = low.getVector(dim1, k);
                    high_v = high.getVector(dim1, k);
                    forwardConvolution(in_v, low_v, high_v, L);
                }
                 
            }
        else
        {
            for (int L = _L; L > 0; --L)//loop on all slices
            {
                // DIMENSIONS 2, 1/2 //
                out = temp_slice.getSubdata(dim0, 0, len0);
                low = (L == _L ? wx.getSubdata(dim0, 0, len0).getSubdata(dim1, 0, len1) : x.getSubdata(dim0, 0, len0));
                high = wx.getSubdata(dim0, 2 * (_L - L) * len0, len0).getSubdata(dim1, len1, len1);
                
                int num_slices = out.getNumVectors(dim1);
                for (int k = 0; k < num_slices; ++k)
                {
                    out_v = out.getVector(dim1, k);
                    low_v = low.getVector(dim1, k);
                    high_v = high.getVector(dim1, k);
                    backwardConvolution(out_v, low_v, high_v, L);
                }
                
                // DIMENSIONS 2, 2/2 //
                out = temp_slice.getSubdata(dim0, len0, len0);
                low = wx.getSubdata(dim0, (1 + 2 * (_L - L)) * len0, len0).getSubdata(dim1, 0, len1);
                high = wx.getSubdata(dim0, (1 + 2 * (_L - L)) * len0, len0).getSubdata(dim1, len1, len1);
                
                num_slices = out.getNumVectors(dim1);
                for (int k = 0; k < num_slices; ++k)
                {
                    out_v = out.getVector(dim1, k);
                    low_v = low.getVector(dim1, k);
                    high_v = high.getVector(dim1, k);
                    backwardConvolution(out_v, low_v, high_v, L);
                }
                
                // DIMENSIONS 1 //
                out = x.getSubdata(dim0, 0, len0);
                low = temp_slice.getSubdata(dim0, 0, len0);
                high = temp_slice.getSubdata(dim0, len0, len0);

                num_slices = out.getNumVectors(dim0);
                for (int k = 0; k < num_slices; ++k)
                {
                    out_v = out.getVector(dim0, k);
                    low_v = low.getVector(dim0, k);
                    high_v = high.getVector(dim0, k);
                    backwardConvolution(out_v, low_v, high_v, L);
                }
            }
        }
    }

    return (forward > 0) ? wx : x;
}





void RedWave::forwardConvolution(const vectorData<double> &in, vectorData<double> &low, vectorData<double> &high, const int &scale){
    
    if ((in.getLength() > low.getLength()) || (in.getLength() > high.getLength()))
        RW_PRINT("ERROR: incoherent sizes for the outputs of the convolution.\n");
    else
    {
        //changes with scale
        int trou = (int) pow(2.0, (double) scale - 1);

        //variables
        int ind, j, f;
        double sum_high, sum_low;

        //convolution
        for (j = 0; j < in.getLength(); j++)//loop on columns
        {
            sum_low = 0;
            sum_high=0;
            //jh=j+scale*x->length;

            for (f = 0; f < _lfilt_forw->getLength(); f++) {
                ind = (j + f * trou) % in.getLength();
                sum_low = sum_low + in.get(ind) * _lfilt_forw->ptr[f];//lf[f];//
                sum_high = sum_high + in.get(ind) * _hfilt_forw->ptr[f];//hf[f];//
            }
            //RW_PRINT("j = %i, l = %g, h = %g, step_l = %i, step_h = %i\n",j,sum_l,sum_h,step_l,step_h);
            low.at(j) = sum_low;
            high.at(j) = sum_high;
            //RW_PRINT("ind = %i\n",scale*x->length+j);
        }
    }
    
}






void RedWave::backwardConvolution(vectorData<double> &out, const vectorData<double> &low, const vectorData<double> &high, const int &scale){
    
    if ((out.getLength() > low.getLength()) || (out.getLength() > high.getLength()))
        RW_PRINT("ERROR: incoherent sizes for the inputs of the convolution.\n");
    else
    {
        //changes with scale
        int trou = (int) pow(2.0, (double) scale - 1);

        //variables
        int ind, j, f;
        double sum;

        //convolution
        for (j=0; j< out.getLength(); j++)//loop on columns
        {
            sum=0;
            for (f = 0; f < _lfilt_back->getLength(); f++) {
                ind = (j + (out.getLength() - f) * trou) % out.getLength();
                sum = sum + low.get(ind) * _lfilt_back->ptr[f] + high.get(ind) * _hfilt_back->ptr[f];//hf[f];//
                //RW_PRINT("(j,ind,f)=(%i,%i,%i) : %g * %g + %g *%g = %g\n",j,ind,f,V_GET(low, ind),lfilt->ptr[f],V_GET(wx, scale*x->length+ind),hfilt->ptr[f], V_GET(low, ind)*lfilt->ptr[f] + V_GET(wx, scale*x->length+ind)*hfilt->ptr[f]);
            }

            out.at(j) = sum;
        }
    }
}







int RedWave::checkConsistency(const NdimData &input, const int &forward) {
    
    int num_dims = _wave_dims->getLength();
    for (int k = 0; k < num_dims; ++k) {
        int waveSize = input.getSize(_wave_dims->at(k));
        if (forward==1)//forward
        {
            //check the number of rows
            if (!isint((double) waveSize / pow(2.0, (double) _L))) {
                RW_PRINT("ERROR: dimension %i must be a multiple of 2^(L).\n(%i not a multiple of %i)\n", (int)(_wave_dims->at(k)+1), (int) waveSize, (int) pow(2.0, (double) _L));
                return 1;
            }
            
        }
        else //backward
        {
            if (!isint(waveSize / (num_dims == 1 ? _L + 1 : 2 * (k == 0 ? _L : 1)))) {
                RW_PRINT("ERROR: dimension %i is not consistent with L-scaled backward transform.\n", (int)(_wave_dims->at(k)+1));
                return 3;
            }
            waveSize = waveSize / (num_dims == 1 ? _L + 1 : 2 * (k == 0 ? _L : 1));
            if (!isint((double) waveSize / pow(2.0, (double) _L))) {
                RW_PRINT("ERROR: dimension %i in the direct domain should have been a multiple of 2^(L).\n", (int)(_wave_dims->at(k)+1));
                return 4;
            }
            
        }
    }
    
    return 0;
}




RedWave::~RedWave()
{
    
    if (_temp_data)
    	delete _temp_data;
    
//     for (int k = 0; k<MAXNDIMDATA+1 ;  ++k)
//         if (_memory_pattern[k])
//             delete _memory_pattern[k];
    if (_inner_wx)
        delete _inner_wx;
    if (_inner_x)
    	delete _inner_x;
    
// 	if (_memory)
//         free(_memory);
    
    if (_hfilt_back)
    	_hfilt_back;
    
    if (_lfilt_back)
    	delete _lfilt_back;
    if (_hfilt_back)
    	delete _hfilt_back;
    if (_lfilt_forw)
    	delete _lfilt_forw;
    if (_hfilt_forw)
        delete _hfilt_forw;
    
    if (_wave_dims)   
    	delete _wave_dims;
    
}


void RedWave::createMemory(const bool wx_buffer)
{
//     //delete preexisting memory
//     if (wx_buffer)
//     {
// //         if (_memory)
// //             free(_memory);
// //         for (int k = 0; k < MAXNDIMDATA; ++k)
// //             if (_memory_pattern[k])
// //                 delete _memory_pattern[k];
// //         if (_temp_data)
// //             delete _temp_data;
//         if (_temp_data)
//              delete _temp_data;
//                 if (_temp_data)
//              delete _temp_data;
//     }
    
    
    
    //delete preexisting memory
    if (_temp_data)
        delete _temp_data;
    if (_inner_wx)
        delete _inner_wx;
    if (_inner_x)
        delete _inner_x;
    
    
    int num_threads = 1;
    #ifdef __PARALLELIZED__
            num_threads = omp_get_max_threads();
    #endif
    
    
    int num_dims = _wave_dims->getLength();

    ////////////////////
    //    temp data   //
    ////////////////////
    int memory_numel = 1;
    int memory_steps[MAXNDIMDATA];
    int memory_sizes[MAXNDIMDATA];
    for (int k = 0; k < MAXNDIMDATA; ++k)
    {//initialize sizes and steps
        memory_sizes[k] = 1;
        memory_steps[k] = 1;
    }

    
    for (int k = num_dims - 1; k >= 0; --k)
    {//set wave sizes
        int dim = _wave_dims->at(k);
        memory_sizes[dim] = _x_sizes[dim] * ( ((num_dims > 1) && (k == 0)) ? 2 : 1);
        memory_steps[dim] = memory_numel;
        memory_numel *= memory_sizes[dim];
    }
    

    // if (_num_slices > 1)
    {//set temp number of slices (depends on parallelization)
        int slice_index = 0;
        while (memory_sizes[slice_index] != 1)
            ++slice_index;
        memory_sizes[slice_index] = num_threads;
        memory_steps[slice_index] = memory_numel;
        memory_numel *= num_threads;
    }
    
    _temp_data = new NdimData(MAXNDIMDATA, memory_sizes, NULL, memory_steps);

    
    //////////////////
    //   inner wx   //
    //////////////////
    
    if (wx_buffer)
    {
        ////////
        // wx //
        ////////
        memory_numel = 1;
        for (int k = 0;k < MAXNDIMDATA; ++k)
        {//initialize sizes and steps
            memory_sizes[k]=1;
            memory_steps[k]=1;
        }


        for (int k = num_dims - 1; k >= 0; --k)
        {//set wave sizes
            int dim = _wave_dims->at(k);
            memory_sizes[dim] = _x_sizes[dim] * 2 * ( k == 0 ? _L : 1);
            memory_steps[dim] = memory_numel;
            memory_numel *= memory_sizes[dim];
        }


        // if (_num_slices > 1)
        {//set temp number of slices (depends on parallelization)
            int slice_index = 0;
            while (memory_sizes[slice_index] != 1)
                ++slice_index;
            memory_sizes[slice_index] = num_threads;
            memory_steps[slice_index] = memory_numel;
            memory_numel *= num_threads;
        }

        _inner_wx = new NdimData(MAXNDIMDATA, memory_sizes, NULL, memory_steps);
        
        ///////
        // x //
        ///////
        memory_numel = 1;
        for (int k = 0;k < MAXNDIMDATA; ++k)
        {//initialize sizes and steps
            memory_sizes[k]=1;
            memory_steps[k]=1;
        }


        for (int k = num_dims - 1; k >= 0; --k)
        {//set wave sizes
            int dim = _wave_dims->at(k);
            memory_sizes[dim] = _x_sizes[dim];
            memory_steps[dim] = memory_numel;
            memory_numel *= memory_sizes[dim];
        }


        //if (_num_slices > 1)
        {//set temp number of slices (depends on parallelization)
            int slice_index = 0;
            while (memory_sizes[slice_index] != 1)
                ++slice_index;
            memory_sizes[slice_index] = num_threads;
            memory_steps[slice_index] = memory_numel;
            memory_numel *= num_threads;
        }

        _inner_x = new NdimData(MAXNDIMDATA, memory_sizes, NULL, memory_steps);
        
        
    }
    
}

const int* RedWave::get_internal_wx_size() const
{
//     if (_memory_pattern[_wave_dims->getLength()])
//         return _memory_pattern[_wave_dims->getLength()]->getSizes();
    if (_inner_wx)
        return _inner_wx->getSizes();
    else
    {
        RW_PRINT("ERROR: internal wx not initialized\n");
        return NULL;
    }
}

const int* RedWave::get_internal_wx_steps() const
{
//     if (_memory_pattern[_wave_dims->getLength()])
//         return _memory_pattern[_wave_dims->getLength()]->getSteps();
    if (_inner_wx)
        return _inner_wx->getSteps();
    else
    {
        RW_PRINT("ERROR: internal wx not initialized\n");
        return NULL;
    }
}

inline int RedWave::get_thread_id() const
{
    int thread = 0;
    #ifdef __PARALLELIZED__
            thread = omp_get_thread_num();
    #endif
    return thread;
}

NdimData RedWave::get_current_x_slice()
{
//     if (_memory_pattern[0])
//     {
//         int temp_slice=0;
//         #ifdef __PARALLELIZED__
//                 temp_slice=omp_get_thread_num();
//         #endif
//         return _memory_pattern[0]->get_slice(*_wave_dims, temp_slice);
//     }
    if (_inner_x)
        return _inner_x->get_slice(*_wave_dims, get_thread_id());
    else
    {
        RW_PRINT("ERROR: internal x not initialized\n");
        return NdimData();
    }
}

NdimData RedWave::get_current_wx_slice()
{
    if (_inner_wx)
        return _inner_wx->get_slice(*_wave_dims, get_thread_id());
    else
    {
        RW_PRINT("ERROR: internal wx not initialized\n");
        return NdimData();
    }
}













// high level stuff
int RedWave::wavelets(NdimData &x, NdimData &wx, const int &forward)
{
    if (check_dimensions(&x, &wx))
    {
        int num_slices = x.getNumSlices(*_wave_dims);
        //x: direct domain matrix
        //wx: wavelet domain matrix
        //forward: forward (1) or backward (-1) transform
            
        /////////////////////////////////
        //    SLICE PARALLELIZATION    //
        /////////////////////////////////
                
        #ifdef __PARALLELIZED__
        #pragma omp parallel for
        #endif
        for (int s = 0; s < num_slices; s++)//loop on all slices
        {
    
            
            
            //RW_PRINT("Slice parallelization\n");
            NdimData x_slice = x.get_slice(this->get_wave_dimensions(), s);
            NdimData wx_slice = wx.get_slice(this->get_wave_dimensions(), s);
            if (_wave_dims->getLength() == 1)
                this->sliceWavelets_1D(x_slice, wx_slice, forward);
            else
                this->sliceWavelets_2D(x_slice, wx_slice, forward);
            
    
        }
        return 0;
    }
    else
        return 1;
}





int RedWave::analysisThresholding(NdimData &input, NdimData &output, const NdimData &thresholds, const int &num_iter)
{
    if (check_dimensions(&input, &thresholds) && check_dimensions(&output, &thresholds))
    {
        int num_slices = input.getNumSlices(*_wave_dims);
        //input: input matrix (to be transformed)
        //output: output matrix (transformed)
        //thresholds: threshold values for each element of the wavelet transform (must be of wavelet size)
        //num_iter: number of iterations of the algorithm
        
        double tau=1.0;//not tested
        bool accelerated=true;//not tested
        
    
        int wx_sliceNumel=this->get_current_wx_slice().getNumel();
        
        NdimData* previous_wx=NULL;
        NdimData* wx_copy=NULL;
        if (num_iter>1)
        {
            previous_wx = new NdimData(MAXNDIMDATA, this->get_internal_wx_size(),NULL,this->get_internal_wx_steps());
            wx_copy = new NdimData(MAXNDIMDATA, this->get_internal_wx_size(),NULL,this->get_internal_wx_steps());
        }
                    
                    
        #ifdef __PARALLELIZED__
        #pragma omp parallel for
        #endif
        for (int s = 0; s < num_slices; s++)//loop on all slices
        {
    
            int temp_slice = this->get_thread_id();
    
            NdimData u_slice=input.get_slice(this->get_wave_dimensions(),s);
            NdimData x_slice = output.get_slice(this->get_wave_dimensions(),s);
            NdimData wx_slice = this->get_current_wx_slice();
            NdimData thresholds_slice=thresholds.get_slice(this->get_wave_dimensions(),s);
    
            //initialize
            this->sliceForward(u_slice,wx_slice);
            
            //temporary slices
            NdimData previous_wx_slice, wx_copy_slice;
            if (num_iter>1)
            {
                wx_copy_slice=wx_copy->get_slice(this->get_wave_dimensions(),temp_slice);
                previous_wx_slice=previous_wx->get_slice(this->get_wave_dimensions(),temp_slice);
            }
    
    
    
            if (num_iter>1)
                memcpy(previous_wx_slice.getPtr(), wx_slice.getPtr(), wx_sliceNumel*sizeof(double) );
    
            //thresholding;
            (1.0 * wx_slice).LinfProjectTo(wx_slice,thresholds_slice);
            
    
            double t = 1.0;
            for (int i=0;i<num_iter-1;++i)//begin iterations
            {
                double t_next = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0;
                double alpha = (t - 1) / t_next;
                t = t_next;
                
                //reweight
                if (accelerated)
                { 
                    memcpy(wx_copy_slice.getPtr(), wx_slice.getPtr(), wx_sliceNumel*sizeof(double) );
                    //contiguousSliceWeightedSum(&wx_slice, &wx_slice, &previous_wx_slice, 1+(t-1)/t, (1-t)/t);
                    wx_slice << (1+alpha) * wx_slice - alpha * previous_wx_slice;
                    memcpy(previous_wx_slice.getPtr(), wx_copy_slice.getPtr(), wx_sliceNumel*sizeof(double) );
                }
                memcpy(wx_copy_slice.getPtr(), wx_slice.getPtr(), wx_sliceNumel*sizeof(double) );
    
    
    
                //gradient
                this->sliceBackward(x_slice,wx_slice);
                //sliceWeightedSum(&x_slice, &x_slice, &u_slice, 1, -1);
                x_slice << 1.0*x_slice - 1.0*u_slice;
                this->sliceForward(x_slice,wx_slice);
    
                //proximal
                (1.0*wx_copy_slice - tau*wx_slice).LinfProjectTo(wx_slice,thresholds_slice, tau);
                //sliceLinfProxOfWeightedSum(&wx_slice, &wx_copy_slice, &wx_slice, 1, -tau, &thresholds_slice, tau);
    
    
            }
    
    
            //get primal variable
            this->sliceBackward(x_slice,wx_slice);
            x_slice<< 1.0*u_slice-1.0*x_slice;
            //sliceWeightedSum(&x_slice, &u_slice, &x_slice, 1, -1);
    
    
    
    
    
                }// end of slice parallelization
    
    
    
    
        ///////////////////////////
        //    cleaning memory    //
        ///////////////////////////
        if (previous_wx)
            delete previous_wx;
        if (wx_copy)
            delete wx_copy;
        
        return 0;
    }
    else
        return 1;
}


int RedWave::analysisInversion(NdimData &x, const NdimData &AtA, const NdimData &AtY, const NdimData &thresholds, const int &num_iter, const bool nonNegative)
{
    if (check_dimensions(&x, &thresholds))
    {
        int num_slices = x.getNumSlices(*_wave_dims);
        
        /////////////////////////////////
        //    Chambolles parameters    //
        /////////////////////////////////
    
        //algorithm parameters
        double sigma = 0.675;//0.975
        double tau = sigma;
        double theta = 1;//0.5;
    
        //data fidelity term implicit gradient matrix
        NdimData AtAI1(AtA.getNdim(), AtA.getSizes());
        int num_sources = AtA.getSize(0);
        for (int i = 0; i < AtA.getNumel(); ++i)
            AtAI1[i]=tau*AtA.get(i);
        for (int i = 0; i < num_sources; ++i)
            AtAI1[i+i*num_sources]+=1;
        gaussInversion(&AtAI1);
    
        
    
        
        /////////////////////
        //    variables    //
        /////////////////////
        NdimData x1(MAXNDIMDATA, x.getSizes());
        NdimData x_interm(MAXNDIMDATA, x.getSizes());
        NdimData* y_pos;
        if (nonNegative)
            y_pos = new NdimData(MAXNDIMDATA, x.getSizes());
        NdimData y_L1(MAXNDIMDATA, thresholds.getSizes());
        //memcpy(previous_wx_slice.getPtr(), wx_slice->getPtr(), sliceNumel*sizeof(double) )
        
                
        /////////////////////////
        //    initialization   //
        /////////////////////////  
            
        memcpy(x1.getPtr(), x.getPtr(), x.getNumel() * sizeof(double) );
        if (nonNegative)
            memcpy(y_pos->getPtr(), x.getPtr(), x.getNumel() * sizeof(double) );
       
       
        #ifdef __PARALLELIZED__
        #pragma omp parallel for
        #endif
            for (int s = 0; s < num_slices; s++)//loop on all slices
            {
    
    
    
                NdimData y_L1_slice = y_L1.get_slice(this->get_wave_dimensions(), s);
                NdimData thresholds_slice = thresholds.get_slice(this->get_wave_dimensions(), s);
                NdimData x_slice = x.get_slice(this->get_wave_dimensions(), s);
                NdimData x_interm_slice = x_interm.get_slice(this->get_wave_dimensions(), s);
                NdimData x1_slice = x1.get_slice(this->get_wave_dimensions(), s);
                NdimData AtY_slice = AtY.get_slice(this->get_wave_dimensions(), s);
                NdimData x2_slice = this->get_current_x_slice();
                NdimData y_pos_slice;
                if (nonNegative)
                    y_pos_slice = y_pos->get_slice(this->get_wave_dimensions(), s);
    
                //initialize y_L1
                this->sliceForward(x_slice,y_L1_slice);
                ((1 + sigma) * y_L1_slice).LinfProjectTo(y_L1_slice, thresholds_slice);
                
                //initialize y_pos
                if (nonNegative)
                    (1.0 * y_pos_slice + sigma * x_slice).negativePartTo(y_pos_slice);
                
                //prepare matrix product
                this->sliceBackward(x2_slice, y_L1_slice);
                if (nonNegative)
                    x_interm_slice << 1.0 * x1_slice - tau * x2_slice - tau * y_pos_slice + tau * AtY_slice;
                else
                    x_interm_slice << 1.0 * x1_slice - tau * x2_slice + tau * AtY_slice;
            }
            
            // GATHER THE THREADS
        
            
            for (int k = 1; k < num_iter; ++k)
            {
                #ifdef __PARALLELIZED__
                #pragma omp parallel for
                #endif
                for (int s = 0; s < num_slices; s++)//loop on all slices
                {
    
    
    
                    NdimData x_slice = x.get_slice(this->get_wave_dimensions(), s);
                    NdimData x1_slice = x1.get_slice(this->get_wave_dimensions(), s);
                    NdimData x2_slice = this->get_current_x_slice();
                    NdimData x_interm_slice = x_interm.get_slice(this->get_wave_dimensions(), s);
    
                
                    vectorData<double> h = AtAI1.getVector(0, s);
                    
                    
                    //matrix product
                    {   NdimData x_interm_slice0 = x_interm.get_slice(this->get_wave_dimensions(), 0);
                        NdimData x_interm_slice1 = x_interm.get_slice(this->get_wave_dimensions(), 1);
                        x2_slice<< h.get(0) * x_interm_slice0 + h.get(1) * x_interm_slice1;}
    
                    for (int k = 2; k < num_slices; ++k)
                    {
                        NdimData x_interm_slice = x_interm.get_slice(this->get_wave_dimensions(), k);
    
                        x2_slice << 1.0 * x2_slice + h.get(k) * x_interm_slice;
                    }
    
    
                    x_slice << (1 + theta) * x2_slice - theta * x1_slice;
                    (1.0 * x2_slice).computeTo(x1_slice);
                }
                
    
                #ifdef __PARALLELIZED__
                #pragma omp parallel for
                #endif
                for (int s = 0; s < num_slices; s++)//loop on all slices
                {
                    NdimData x_slice = x.get_slice(this->get_wave_dimensions(), s);
                    NdimData x1_slice = x1.get_slice(this->get_wave_dimensions(), s);
                    NdimData x2_slice = this->get_current_x_slice();
                    NdimData x_interm_slice = x_interm.get_slice(this->get_wave_dimensions(), s);
                    NdimData AtY_slice = AtY.get_slice(this->get_wave_dimensions(), s);
                    
                    NdimData y_L1_slice = y_L1.get_slice(this->get_wave_dimensions(), s);
                    NdimData thresholds_slice = thresholds.get_slice(this->get_wave_dimensions(), s);
                    NdimData wx_temp_slice = this->get_current_wx_slice();
                    NdimData y_pos_slice;
                    if (nonNegative)
                    	y_pos_slice = y_pos->get_slice(this->get_wave_dimensions(), s);
                    
                //update y_L1
                this->sliceForward(x_slice, wx_temp_slice);
                (1.0 * y_L1_slice + sigma * wx_temp_slice).LinfProjectTo(y_L1_slice, thresholds_slice);
                
                //update y_pos
                if (nonNegative)
                    (1.0 * y_pos_slice + sigma * x_slice).negativePartTo(y_pos_slice);
                
                //prepare matrix product
                this->sliceBackward(x2_slice, y_L1_slice);
                if (nonNegative)
                    x_interm_slice << 1.0 * x1_slice - tau * x2_slice - tau * y_pos_slice + tau * AtY_slice;
                else
                    x_interm_slice << 1.0 * x1_slice - tau * x2_slice + tau * AtY_slice;
    
                }//GATHER THREADS
            }
        
        
            //TERMINATE
        
            #ifdef __PARALLELIZED__
            #pragma omp parallel for
            #endif
            for (int s = 0; s < num_slices; s++)//loop on all slices
            {
    
                    NdimData x_slice = x.get_slice(this->get_wave_dimensions(), s);
                    NdimData x1_slice = x1.get_slice(this->get_wave_dimensions(), s);
                    NdimData x2_slice = this->get_current_x_slice(); 
                    
                    vectorData<double> h = AtAI1.getVector(0,s);
                    
                    
    
                    {   NdimData x_interm_slice0 = x_interm.get_slice(this->get_wave_dimensions(), 0);
                        NdimData x_interm_slice1 = x_interm.get_slice(this->get_wave_dimensions(), 1);
                        x2_slice << h.get(0) * x_interm_slice0 + h.get(1) * x_interm_slice1;}
    
                    for (int k = 2; k < num_slices; ++k)
                    {
                        NdimData x_interm_slice = x_interm.get_slice(this->get_wave_dimensions(), k);
    
                        x2_slice << 1.0 * x2_slice+h.get(k) * x_interm_slice;
                    }
    
                    if (nonNegative)
                        ((1 + theta) * x2_slice - theta * x1_slice).positivePartTo(x_slice);
                    else
                        ((1 + theta) * x2_slice - theta * x1_slice).computeTo(x_slice);
            }
            
        if (nonNegative)
            delete y_pos;
    
        return 0;
    }
    else
        return 1;
}

        
        



//////////////////////////////////////////////////////////////
//                         Vectors and data                          //
//////////////////////////////////////////////////////////////



/************************
 *    NdimData Class    *
 ************************/

NdimData::NdimData() {
    _ptr = NULL;
    _owner = 0;
    _Ndim = 0;
    _numel = 0;
    for (int k = 0; k < MAXNDIMDATA; ++k) {
        _sizes[k] = 0;
        _steps[k] = 0;
    }
}

void NdimData::printInfo(const char* str) const
{
    RW_PRINT("id: %s\n", str);
    RW_PRINT("Sizes: ");
    for (int k = 0; k < MAXNDIMDATA - 1; ++k)
        RW_PRINT("%i, ", _sizes[k]);
    RW_PRINT("%i.\n", _sizes[MAXNDIMDATA - 1]);
    
    RW_PRINT("Steps: ");
    for (int k = 0; k < MAXNDIMDATA - 1; ++k)
        RW_PRINT("%i, ", _steps[k]);
    RW_PRINT("%i.\n", _steps[MAXNDIMDATA - 1]);
    
    RW_PRINT("owner: %i, numel: %i, pointer: %p.\n\n", _owner, _numel, _ptr);
}




NdimData::NdimData(const int &N, const int* sizes, double* pointer, const int* steps) {
    _Ndim = N;//dimension
    _ptr = pointer;
    _sizes[0] = sizes[0];
    _steps[0] = steps ? steps[0] : 1;
    _numel = sizes[0];
    
    for (int k = 1; k < _Ndim; ++k) {
        _sizes[k] = sizes[k];
        _numel *= sizes[k];
        _steps[k] = steps ? steps[k] : _steps[k-1] * _sizes[k-1];
    }
    
    for (int k = _Ndim; k < MAXNDIMDATA; ++k) {
        _sizes[k] = 1;
        _steps[k] = _steps[k-1] * _sizes[k-1];
    }
    
    if (_ptr == NULL) {
        //compute size
        int total_size = 1;
        for (int k = 0; k < _Ndim; ++k)
            total_size *= _sizes[k];
        //allocate
        _ptr = new double[total_size];
        _owner = 1;
    }
    else
        _owner = 0;
}


NdimData::NdimData(const int &size0, const int &size1, double* pointer) {
    _Ndim=2;//dimension
    _ptr=pointer;
    _sizes[0]=size0;_sizes[1]=size1;
    _steps[0]=1;_steps[1]=size0;
    _numel=size0*size1;
    
    for (int k=_Ndim;k<MAXNDIMDATA;++k) {
        _sizes[k]=1;
        _steps[k]=_steps[k-1]*_sizes[k-1];
    }
    
    if (_ptr==NULL) {
        //allocate
        _ptr= new double[size0 * size1];
        _owner=1;
    }
    else
        _owner=0;
    
}
// 


NdimData& NdimData::operator=(const NdimData& other) {
    if (_owner) {
        delete[] _ptr;
        RW_PRINT("BEWARE: Freeing memory during copy.\n");
    }
    
    _ptr=other._ptr;
    _Ndim=other._Ndim;
    _numel=other._numel;
    _owner=0;
    if (other._owner!=0)
        RW_PRINT("BEWARE: Copying owner NdimData is dangerous.\n");
    
    memcpy(_sizes, other._sizes, MAXNDIMDATA*sizeof(int) );
    memcpy(_steps, other._steps, MAXNDIMDATA*sizeof(int) );
    
    return *this;
}



vectorData<double> NdimData::getVector(const int &face, const int &ind) {
    
    double* ptr=_ptr;
    
    int size=1;
    for (int k=0;k<_Ndim;++k)
        if (k!=face) {
        ptr+= (int(ind/size)%_sizes[k])*_steps[k];
        size*=_sizes[k];
        }
    
    return vectorData<double>(_sizes[face], ptr, _steps[face]);
    
}

const vectorData<double> NdimData::getVector(const int &face, const int &ind) const{
    
    double* ptr=_ptr;
    
    int size=1;
    for (int k=0;k<_Ndim;++k)
        if (k!=face) {
        ptr+= (int(ind/size)%_sizes[k])*_steps[k];
        size*=_sizes[k];
        }
    
    return vectorData<double>(_sizes[face], ptr, _steps[face]);
    
}


 int NdimData::getNumSlices(const vectorData<int> &faceDims) const{
    int num=_numel;
    for (int k=0;k<faceDims.getLength();++k)
        num/=_sizes[faceDims.get(k)];
    return num;
}

NdimData NdimData::get_slice(const vectorData<int> &faceDims, const int &ind) {
    double* ptr=_ptr;
    int sliceSizes[MAXNDIMDATA];
    memcpy(sliceSizes, _sizes, MAXNDIMDATA*sizeof(int) );
    int size=1;
    for (int k=0;k<_Ndim;++k) {
        bool isface=false;
        //check if dim on face or on slice
        for (int i=0;i<faceDims.getLength();++i)
            if (faceDims.get(i)==k)
                isface=true;
        
        //if not, increment pointer
        if (!isface) {
            ptr+= (int(ind/size)%_sizes[k])*_steps[k];
            size*=_sizes[k];
            //if not on face, set size to 1
            sliceSizes[k]=1;
        }
    }
    return NdimData(_Ndim, sliceSizes, ptr, _steps);
}
const NdimData NdimData::get_slice(const vectorData<int> &faceDims, const int &ind) const {
    double* ptr=_ptr;
    int sliceSizes[MAXNDIMDATA];
    memcpy(sliceSizes, _sizes, MAXNDIMDATA*sizeof(int) );
    int size=1;
    for (int k=0;k<_Ndim;++k) {
        bool isface=false;
        //check if dim on face or on slice
        for (int i=0;i<faceDims.getLength();++i)
            if (faceDims.get(i)==k)
                isface=true;
        
        //if not, increment pointer
        if (!isface) {
            ptr+= (int(ind/size)%_sizes[k])*_steps[k];
            size*=_sizes[k];
            //if not on face, set size to 1
            sliceSizes[k]=1;
        }
    }
    return NdimData(_Ndim, sliceSizes, ptr, _steps);
}

NdimData NdimData::getSubdata(const int &dimension, const int &offset,  const int &new_length)
{

    if (_sizes[dimension] < offset + new_length)
    {
                        this->printInfo("inside");
                RW_PRINT("dim: %i, offset + length: %i\n", _sizes[dimension], offset + new_length);
        RW_PRINT("ERROR: subdata is not contained in the initial data.\n");
        return NdimData(_Ndim, _sizes, _ptr, _steps);
    }
    else
    {
        int subdataSizes[MAXNDIMDATA];
        memcpy(subdataSizes, _sizes, MAXNDIMDATA * sizeof(int));
        subdataSizes[dimension] = new_length;
        return NdimData(_Ndim, subdataSizes, _ptr + _steps[dimension] * offset, _steps);
        
    }

}









#define OUT_EL(i,j) output->getPtr()[i+n*j]
void gaussInversion(NdimData* output,NdimData* input)
{
    int n = output->getSize(0);
    if (input)
        memcpy(output->getPtr(), input->getPtr(), input->getNumel()*sizeof(double) );
    vectorData<int> perm(n);
    vectorData<double> temp(n);
    
    for (int k=0;k<n;++k)
        perm.at(k)=k;
    
    for (int k=0;k<n;++k)
        {
         //find pivot
         int imax=k;
         double maxval=OUT_EL(k,k);
         
         //pivot is NOT WORKING
         for (int i=k+1;i<n;++i)
         {
            
            //RW_PRINT("k=%i, i=%i, %f and %f\n",k,i,ABS(OUT_EL(i,k)),ABS(maxval));
            if (ABS(OUT_EL(i,k))>ABS(maxval))
            {
                maxval=OUT_EL(i,k);
                imax=i;
            }
        }
        if (imax!=k)
        {
            double temp=perm.at(imax);
            perm.at(imax)=perm.get(k);
            perm.at(k)=(int)temp;
            
            for (int i=0;i<n;++i)
            {
                temp=OUT_EL(imax,i);
                OUT_EL(imax,i)=OUT_EL(k,i);
                OUT_EL(k,i)=temp;
            }
        }
        //RW_PRINT("max : %f (ind %i)\n",maxval,imax);
        
        double d=1.0/maxval;
        for (int i=0;i<n;++i)
            temp.at(i)=OUT_EL(i,k);
        
        for (int j=0;j<n;++j)
        {
            double c = OUT_EL(k,j)*d;
            for (int i=0;i<n;++i)
                OUT_EL(i,j)=OUT_EL(i,j)-temp.get(i)*c;
            OUT_EL(k,j)=c;
        }
        for (int i=0;i<n;++i)
        	OUT_EL(i,k)=-temp.get(i)*d;
        OUT_EL(k,k)=d;
        
    }
    
//     for (int i=0;i<n;++i)
//         RW_PRINT("%i ",perm.get(i));
//        RW_PRINT("\n") ;
       
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<n;++j)
            temp.at(perm.get(j))=OUT_EL(i,j);
        for (int j=0;j<n;++j)
            OUT_EL(i,j)=temp.at(j);
    }
}








/******************
 *   Operations   *
 ******************/

void NdimDataOperations::add(const double &scalar,const NdimData &data)
{
    if (_num_ope<MAX_OPERATION_NUMBER)
    {
        _scalars[_num_ope]= 1;
        _data_ptrs[_num_ope] = &data;
        ++_num_ope;
        //RW_PRINT("test\n");
    }
    else
    {
        RW_PRINT("ERROR: too many operations in NdimDataOperations.\n");
    }
}




NdimDataOperations& NdimDataOperations::operator+(NdimDataOperations ope)
{
    if (MAX_OPERATION_NUMBER>this->_num_ope+ope._num_ope)
    {
        memcpy(this->_scalars+this->_num_ope, ope._scalars, ope._num_ope*sizeof(double));
        memcpy(this->_data_ptrs+this->_num_ope, ope._data_ptrs, ope._num_ope*sizeof(NdimData*));
        this->_num_ope+=ope._num_ope;
    }
    else
        RW_PRINT("ERROR: too many operations in NdimDataOperations.\n");
    
    return *this;
}

NdimData& NdimDataOperations::computeTo(NdimData& output)
{
        //choose vector dimension
    int dim=-1;
    int length=-1;
    for (int k=0;k<output.getNdim();++k)
        if (output.getSize(k)>length)
        {
            dim=k;
            length=output.getSize(k);
        }
    
    //compute on each vector
    int num_vectors=output.getNumVectors(dim);
    vectorData<double> vcts[MAX_OPERATION_NUMBER];
    vectorData<double> out_vct;
    
    for (int k=0;k<num_vectors;++k)
    {
        out_vct = output.getVector(dim,k);
        for (int i = 0; i<_num_ope; ++i)
            vcts[i]= _data_ptrs[i]->getVector(dim,k);
        
           
           for (int el=0;el<length;el++)
           {
//                 out_vct[el]=_scalars[0]*_vcts[0].get(el);
//                 for (int i=1; i<_num_ope; ++i)
//                     out_vct[el]+=_scalars[i]*_vcts[i].get(el);
                double val = 0;
                for (int i=0; i<_num_ope; ++i)
                     val+=_scalars[i]*vcts[i].get(el);
                out_vct[el]=val;
           }
	}
    
    return output;
}

NdimData& NdimDataOperations::negativePartTo(NdimData& output)
{
        //choose vector dimension
    int dim=-1;
    int length=-1;
    for (int k=0;k<output.getNdim();++k)
        if (output.getSize(k)>length)
        {
            dim=k;
            length=output.getSize(k);
        }
    
    //compute on each vector
    int num_vectors=output.getNumVectors(dim);
    vectorData<double> vcts[MAX_OPERATION_NUMBER];
    vectorData<double> out_vct;
    
    for (int k=0;k<num_vectors;++k)
    {
        out_vct = output.getVector(dim,k);
        for (int i = 0; i<_num_ope; ++i)
            vcts[i]= _data_ptrs[i]->getVector(dim,k);
        
           
           for (int el=0;el<length;el++)
           {

                double val = 0;
                for (int i=0; i<_num_ope; ++i)
                     val+=_scalars[i]*vcts[i].get(el);
                out_vct[el]=(val<0?val:0);
           }
	}
    
    return output;
}

NdimData& NdimDataOperations::positivePartTo(NdimData& output)
{
        //choose vector dimension
    int dim=-1;
    int length=-1;
    for (int k=0;k<output.getNdim();++k)
        if (output.getSize(k)>length)
        {
            dim=k;
            length=output.getSize(k);
        }
    
    //compute on each vector
    int num_vectors=output.getNumVectors(dim);
    vectorData<double> vcts[MAX_OPERATION_NUMBER];
    vectorData<double> out_vct;
    
    for (int k=0;k<num_vectors;++k)
    {
        out_vct = output.getVector(dim,k);
        for (int i = 0; i<_num_ope; ++i)
            vcts[i]= _data_ptrs[i]->getVector(dim,k);
        
           
           for (int el=0;el<length;el++)
           {

                double val = 0;
                for (int i=0; i<_num_ope; ++i)
                     val+=_scalars[i]*vcts[i].get(el);
                out_vct[el]=(val>0?val:0);
           }
	}
    
    return output;
}


NdimData& NdimDataOperations::LinfProjectTo(NdimData& output, const NdimData& inf_bounds, const double &tau)
{
	//choose vector dimension
    int dim=-1;
    int length=-1;
    for (int k=0;k<output.getNdim();++k)
        if (output.getSize(k)>length)
        {
            dim=k;
            length=output.getSize(k);
        }
    
    //compute on each vector
    int num_vectors=output.getNumVectors(dim);
    vectorData<double> vcts[MAX_OPERATION_NUMBER];
    vectorData<double> out_vct;
    vectorData<double> inf_vct;
    
    for (int k=0;k<num_vectors;++k)
    {
        out_vct = output.getVector(dim,k);
        inf_vct = inf_bounds.getVector(dim,k);
        for (int i = 0; i<_num_ope; ++i)
            vcts[i]= _data_ptrs[i]->getVector(dim,k);
        
           
           for (int el=0;el<length;el++)
           {
                double val = 0;
                for (int i=0; i<_num_ope; ++i)
                     val+=_scalars[i]*vcts[i].get(el);
                
                double t = inf_vct.get(el)*tau;
                val=(val>t?t:val);
                out_vct[el]=(val<-t?-t:val);
           }
	}
    
    return output;
}



NdimDataOperations::NdimDataOperations(const NdimDataOperations &ope)
{
    _num_ope=ope._num_ope;
    memcpy(_data_ptrs,ope._data_ptrs,_num_ope*sizeof(NdimData*));
    memcpy(_scalars,ope._scalars,_num_ope*sizeof(double));
    
}



/******************
 *   Interface    *
 ******************/

#ifdef RW_MATLAB_INTERFACE
const NdimData matlabArrayToNdimData(const mxArray *array)
{
    int ndims = mxGetNumberOfDimensions(array);
    if (ndims>MAXNDIMDATA) {
        RW_PRINT("ERROR: only data up to %i dimensions are supported.\n", ndims);
        return NdimData();
    }
    return NdimData(ndims, mxGetDimensions(array), mxGetPr(array));
}
#endif



#ifdef RW_PYTHON_INTERFACE
NdimData narray_to_ndimdata(const np::ndarray& array)
{
	np::dtype dtype = array.get_dtype();
	np::dtype dtype_expected = np::dtype::get_builtin<double>();
	int ndims = array.get_nd();
	
	if (dtype != dtype_expected)
	{
		std::stringstream ss;
		ss << "RedWave: Unexpected data type (" << bp::extract<const char*>(dtype.attr("__str__")()) << ") received. ";
		ss << "Expected " << bp::extract<const char*>(dtype_expected.attr("__str__")());
		throw std::runtime_error(ss.str().c_str());
	}
	if (ndims > MAXNDIMDATA)
	{
		std::stringstream ss;
		ss << "RedWave: Too many dimensions, got " << ndims << " while maximum is " << MAXNDIMDATA << " (this can be changed by modifying constant MAXNDIMDATA in redWaveTools.h).";
		throw std::runtime_error(ss.str().c_str());
	}
	
	double* data = reinterpret_cast<double*>(array.get_data());
	const Py_intptr_t* strides = array.get_strides();
	const Py_intptr_t* shape = array.get_shape();
	int sizes[MAXNDIMDATA];
	int steps[MAXNDIMDATA];
	for (int k = 0; k < ndims; ++k)
	{
		steps[k] = strides[k] * sizeof(char) / sizeof(double);
		sizes[k] = shape[k];
	}
	return NdimData(ndims, sizes, data, steps);		
}
#endif

