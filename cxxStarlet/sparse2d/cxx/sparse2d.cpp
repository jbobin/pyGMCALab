#include "starlet2d.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(sparse2d)
{
	np::initialize();
	
	class_< Starlet2D >("Starlet2D", init<int, int, int,int,int >())
	  .def("forward", &Starlet2D::transform_numpy)
	  .def("forward_omp", &Starlet2D::transform_omp_numpy)
	  .def("forward1d_omp", &Starlet2D::transform1d_omp_numpy)
	  .def("backward", &Starlet2D::reconstruct_numpy)
	  .def("adjoint1d",&Starlet2D:: adjoint1d_omp_numpy)
	  .def("backward_omp", &Starlet2D::reconstruct_omp_numpy);
}
