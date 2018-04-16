/*
 * cln.h - This file is part of MRS3D
 * Created on 16/05/11
 * Contributor : François Lanusse (francois.lanusse@gmail.com)
 *
 * Copyright 2012 CEA
 *
 * This software is a computer program whose purpose is to apply mutli-
 * resolution signal processing algorithms on spherical 3D data.
 *
 * This software is governed by the CeCILL  license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

#ifndef STARLET2D_H
#define STARLET2D_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "omp.h"
#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::numpy;

class Starlet2D
{
    
public:
  Starlet2D(int Nx, int Ny, int Nz, int nscales,int lh);

    void filter_1d(double* xin,double* xout,int N,double* h,int j);
    void transform(double* In, double* Wt_Out, double* CR_Out, double* xIn, double* yIn, double* xOut, double* yOut, double* h);
    void transform1d(double* In, double* Wt_Out, double* CR_Out, double* xIn,  double* xOut,  double* h);
    
    np::ndarray transform_numpy(np::ndarray &In, np::ndarray &Filter){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
	NumPyArrayData<double> F_data(Filter);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny,J+1), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
        double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*(J+1));
	double *pt_WT = (double *) malloc(sizeof(double)*Nx*Ny*J);
	double *pt_CR = (double *) malloc(sizeof(double)*Nx*Ny);
	double *pt_Row = (double *) malloc(sizeof(double)*Ny);
	double *pt_Col = (double *) malloc(sizeof(double)*Nx);
	double *pt_Ro = (double *) malloc(sizeof(double)*Ny);
	double *pt_Co = (double *) malloc(sizeof(double)*Nx);
	double *pt_F = (double *) malloc(sizeof(double)*Lh);

	for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }

	for (int k =0; k< Nz; k++){
	
	    for (int x=0; x < Nx; x++){
	      for (int y =0; y< Ny; y++){
		    pt_In[x + y*Nx] = In_data(k,x,y);
		}
	      }

	
	    // Transform
    
	    transform(pt_In, pt_WT, pt_CR, pt_Row, pt_Col, pt_Ro, pt_Co, pt_F);
        
	    for (int x=0; x < Nx; x++) {
	      for (int y =0; y< Ny; y++) {
		for (int j =0; j< J; j++) {
		   Out_data(k,x,y,j) = pt_WT[x + y*Nx + j*Nx*Ny];
		}
		Out_data(k,x,y,J) = pt_CR[x + y*Nx];
	      }
	    }

	  }
        
        free(pt_In);
        free(pt_Out);
	free(pt_WT);
	free(pt_CR);
	free(pt_Row);
	free(pt_Col);
	free(pt_Ro);
	free(pt_Co);
	free(pt_F);
        
        return Out;
    }

    np::ndarray transform_omp_numpy(np::ndarray &In, np::ndarray &Filter){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
	NumPyArrayData<double> F_data(Filter);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny,J+1), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
	double *pt_F = (double *) malloc(sizeof(double)*Lh);

	for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }


	int k;

#pragma omp parallel for shared(In_data,Out_data, pt_F, k)
	for (k =0; k< Nz; k++){

	    double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
	    double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*(J+1));
	    double *pt_WT = (double *) malloc(sizeof(double)*Nx*Ny*J);
	    double *pt_CR = (double *) malloc(sizeof(double)*Nx*Ny);
	    double *pt_Row = (double *) malloc(sizeof(double)*Ny);
	    double *pt_Col = (double *) malloc(sizeof(double)*Nx);
	    double *pt_Ro = (double *) malloc(sizeof(double)*Ny);
	    double *pt_Co = (double *) malloc(sizeof(double)*Nx);
	
	    for (int x=0; x < Nx; x++){
	      for (int y =0; y< Ny; y++){
		    pt_In[x + y*Nx] = In_data(k,x,y);
		}
	      }

	
	    // Transform
    
	    transform(pt_In, pt_WT, pt_CR, pt_Row, pt_Col, pt_Ro, pt_Co, pt_F);
        
	    for (int x=0; x < Nx; x++) {
	      for (int y =0; y< Ny; y++) {
		for (int j =0; j< J; j++) {
		   Out_data(k,x,y,j) = pt_WT[x + y*Nx + j*Nx*Ny];
		}
		Out_data(k,x,y,J) = pt_CR[x + y*Nx];
	      }
	    }

	    free(pt_In);
	    free(pt_Out);
	    free(pt_WT);
	    free(pt_CR);
	    free(pt_Row);
	    free(pt_Col);
	    free(pt_Ro);
	    free(pt_Co);

	  }
        
        
	free(pt_F);
        
        return Out;
    }

    
    void reconstruct(double* CR, double* WT );
    np::ndarray reconstruct_numpy(np::ndarray &In){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_WT = (double *) malloc (sizeof(double)*Nx*Ny*J);
        double *pt_CR = (double *) malloc (sizeof(double)*Nx*Ny);
	double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
        
        for (int k =0; k< Nz; k++){

	  for (int x=0; x < Nx; x++) {
	      for (int y =0; y< Ny; y++) {
		for (int j =0; j< J; j++) {
		    pt_WT[x + y*Nx + j*Nx*Ny] = In_data(k,x,y,j);
		}
		pt_CR[x + y*Nx] = In_data(k,x,y,J);
	      }
	    }

	    //Reconstruction
    
	  reconstruct(pt_CR, pt_WT);
	  
	    for (int x=0; x < Nx; x++){
	      for (int y =0; y< Ny; y++){
		Out_data(k,x,y) = pt_CR[x + y*Nx];
		}
	      }

	  }
        
        
        return Out;
    }

    np::ndarray reconstruct_omp_numpy(np::ndarray &In){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

	int k;

#pragma omp parallel for shared(In_data,Out_data)
        for (k =0; k< Nz; k++){

	  double *pt_WT = (double *) malloc (sizeof(double)*Nx*Ny*J);
        double *pt_CR = (double *) malloc (sizeof(double)*Nx*Ny);
	double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);

	  for (int x=0; x < Nx; x++) {
	      for (int y =0; y< Ny; y++) {
		for (int j =0; j< J; j++) {
		    pt_WT[x + y*Nx + j*Nx*Ny] = In_data(k,x,y,j);
		}
		pt_CR[x + y*Nx] = In_data(k,x,y,J);
	      }
	    }

	    //Reconstruction
    
	  reconstruct(pt_CR, pt_WT);
	  
	    for (int x=0; x < Nx; x++){
	      for (int y =0; y< Ny; y++){
		Out_data(k,x,y) = pt_CR[x + y*Nx];
		}
	      }

	    free(pt_CR);
	    free(pt_WT);
	    free(pt_Out);

	  }
        
        
        return Out;
    }
    
    
    // 1D TRANSFORM
    
    
    np::ndarray transform1d_omp_numpy(np::ndarray &In, np::ndarray &Filter){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
	NumPyArrayData<double> F_data(Filter);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,J+1), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
	double *pt_F = (double *) malloc(sizeof(double)*Lh);

	for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }


	int k;

#pragma omp parallel for shared(In_data,Out_data, pt_F, k)
	for (k =0; k< Nz; k++){

	    double *pt_In = (double *) malloc(sizeof(double)*Nx);
	    double *pt_Out = (double *) malloc(sizeof(double)*Nx*(J+1));
	    double *pt_WT = (double *) malloc(sizeof(double)*Nx*J);
	    double *pt_CR = (double *) malloc(sizeof(double)*Nx);
	    double *pt_Col = (double *) malloc(sizeof(double)*Nx);
	    double *pt_Co = (double *) malloc(sizeof(double)*Nx);
	
	    for (int x=0; x < Nx; x++){
		    pt_In[x] = In_data(k,x);
		}
	
	    // Transform
    
	    //transform(pt_In, pt_WT, pt_CR, pt_Row, pt_Col, pt_Ro, pt_Co, pt_F);
	    transform1d(pt_In, pt_WT, pt_CR, pt_Col, pt_Co, pt_F);
        
	    for (int x=0; x < Nx; x++) {
		for (int j =0; j< J; j++) {
		   Out_data(k,x,j) = pt_WT[x  + j*Nx];
		}
		Out_data(k,x,J) = pt_CR[x];
	    }

	    free(pt_In);
	    free(pt_Out);
	    free(pt_WT);
	    free(pt_CR);
	    free(pt_Col);
	    free(pt_Co);

	  }
        
        
	free(pt_F);
        
        return Out;
    }
    
    // ADJOINT 1D
    void adjoint1d(double* CR, double* WT , double* xIn,  double* xOut,  double* xTemp,  double* h);
    
    np::ndarray adjoint1d_omp_numpy(np::ndarray &In, np::ndarray &Filter){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        NumPyArrayData<double> F_data(Filter);
        
        double *pt_F = (double *) malloc(sizeof(double)*Lh);

		for (int x=0; x < Lh; x++)
	  	{
	    	pt_F[x] = F_data(x);
	  	}

		int k;

		#pragma omp parallel for shared(In_data,Out_data, pt_F, k)
        for (k =0; k< Nz; k++){

	  		double *pt_WT = (double *) malloc (sizeof(double)*Nx*J);
        	double *pt_CR = (double *) malloc (sizeof(double)*Nx);
			double *pt_Out = (double *) malloc (sizeof(double)*Nx);
			double *pt_In = (double *) malloc(sizeof(double)*Nx);
			double *pt_Temp = (double *) malloc(sizeof(double)*Nx);
	    	
	    	
	    	for (int x=0; x < Nx; x++){
	    		for (int j =0; j< J; j++) {
		    		pt_WT[x  + j*Nx] = In_data(k,x,j);
				}
		    	pt_CR[x] = In_data(k,x,J);
			}
			

	    	//Apply the adjoint
    
	  		adjoint1d(pt_CR, pt_WT,pt_In,pt_Out,pt_Temp,pt_F);
	  
	    	for (int x=0; x < Nx; x++){
				Out_data(k,x) = pt_CR[x];
	      	}

	    	free(pt_CR);
	    	free(pt_WT);
	    	free(pt_Out);
	    	free(pt_In);
	    	free(pt_Temp);

	  	}
        
        
        return Out;
    }

    
    // PRIVATE
    
private:
 
    int Nx, Ny, Nz, J,Lh;
};

#endif // STARLET2D_H
