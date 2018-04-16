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


#include <iostream>
#include "starlet2d.h"

Starlet2D::Starlet2D(int nx, int ny,int nz, int Nscales,int lh)
{
    Nx = nx;
    Ny = ny;
    Nz = nz;
    J = Nscales;
    Lh = lh;
}

// 1D Filtering routine with mirror border option
// Input, Output, Filter, Filter length

void Starlet2D::filter_1d(double* xin,double* xout,int N,double* h,int scale)
{
  int i = 0;
  int j = 0;
  int m2 = floor(Lh/2);
  double val = 0;
  int Lindix = 0;
  int Tindix = 0;

  for (i=0;i<N;i++)
    {
      val = 0;
      for (j=0;j<Lh;j++) 
        {
	  Lindix = i + pow(2,scale)*(j - m2);
	  Tindix = Lindix;
	  if (Lindix < 0){
      	    Lindix = -Tindix;
	  }
	  if (Lindix > N-1){
	    Lindix = 2*(N-1) - Tindix;
	  }
	  val = val + h[j]*xin[Lindix];
	}
      xout[i] = val;
    }
}

// Forward starlet transform

void Starlet2D::transform(double* In, double* Wt_Out, double* CR_Out, double* xIn, double* yIn, double* xOut, double* yOut, double* h)
{
  int i = 0;
  int j = 0;
  int k = 0;

  for (j=0;j<J;j++) // Looping over the scales
    {

      for (i=0;i<Nx;i++)  // Looping over the rows
	{
	  for (k=0;k<Ny;k++)
	    {
	      yIn[k] = In[i + k*Nx];
	    }

	  filter_1d(yIn,yOut,Ny,h,j);
	  
	  for (k=0;k<Ny;k++)
	    {
	      CR_Out[i + k*Nx] = yOut[k];
	    }

	}

      for (i=0;i<Ny;i++)  // Looping over the columns
	{
	  for (k=0;k<Nx;k++)
	    {
	      xIn[k] = CR_Out[k+i*Nx];
	    }

	  filter_1d(xIn,xOut,Nx,h,j);
	  
	  for (k=0;k<Nx;k++)
	    {
	      CR_Out[k+i*Nx] = xOut[k];
	    }
	}

      // Updating the wavelet coefficients

      for (i=0;i<Ny;i++)
	{
	  for (k=0;k<Nx;k++)
	    {
	      Wt_Out[k+i*Nx+j*Nx*Ny] = In[k + i*Nx] - CR_Out[k + i*Nx];
	      In[k + i*Nx] =  CR_Out[k + i*Nx];
	    }
	}
    }
}

// Forward starlet transform

void Starlet2D::transform1d(double* In, double* Wt_Out, double* CR_Out, double* xIn,  double* xOut, double* h)
{
  int k = 0;
  int j = 0;

  for (j=0;j<J;j++) // Looping over the scales
    {

	  for (k=0;k<Nx;k++)
	    {
	      xIn[k] = In[k];;
	    }

	  filter_1d(xIn,xOut,Nx,h,j);
	  
	  for (k=0;k<Nx;k++)
	    {
	      CR_Out[k] = xOut[k];
	    }
	
      // Updating the wavelet coefficients

	  for (k=0;k<Nx;k++)
	    {
	      Wt_Out[k+j*Nx] = In[k] - CR_Out[k];
	      In[k] =  CR_Out[k];
	    }
    }
}


void Starlet2D::reconstruct(double* CR, double* WT )
{
  int i=0;
  int j=0;
  int k=0;

  for (j=0;j<J;j++)
    {
      for (i=0;i<Ny;i++)
	{
	  for (k=0;k<Nx;k++)
	    {
	      CR[k + i*Nx] += WT[k+i*Nx+j*Nx*Ny];
	    }
	}
    }
}

void Starlet2D::adjoint1d(double* CR, double* WT, double* xIn,  double* xOut,  double* xTemp, double* h)
{
  int j=0;
  int k=0;
  
  
  for (j=J-1;j>-1;j--) 
   {
     
	 // Filter the wt
	 
	 for (k=0;k<Nx;k++)
  	  {
	      xTemp[k] = WT[k+j*Nx];
      }
	 
	 filter_1d(xTemp,xOut,Nx,h,j);
	  
	  for (k=0;k<Nx;k++)
	    {
	      xTemp[k] -= xOut[k];
	    }
	    
	  // Filter the coarse scale

	  filter_1d(CR,xOut,Nx,h,j);
	  
	  for (k=0;k<Nx;k++)
	    {
	      CR[k] = xOut[k] + xTemp[k];
	    }      
    }
    
}




