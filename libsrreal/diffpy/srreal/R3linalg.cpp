/***********************************************************************
* Short Title: linear algebra functions on R3
*
* Comments: defininitions of linear algebra functions for
*     blitz::TinyVector  and  blitz::TinyMatrix
*
* $Id$
*
* <license text>
***********************************************************************/

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <diffpy/srreal/R3linalg.hpp>

using namespace diffpy::srreal;

double R3::determinant(const R3::Matrix& A)
{
    gsl_matrix* gA = gsl_matrix_alloc(R3::Ndim, R3::Ndim);
    for (int i = 0; i != R3::Ndim; ++i)
    {
	for (int j = 0; j != R3::Ndim; ++j)
	{
	    gsl_matrix_set(gA, i, j, A(i,j));
	}
    }
    gsl_permutation* gP = gsl_permutation_alloc(R3::Ndim);
    int signum;
    gsl_linalg_LU_decomp(gA, gP, &signum);
    double det = gsl_linalg_LU_det(gA, signum);
    gsl_permutation_free(gP);
    gsl_matrix_free(gA);
    return det;
}


R3::Matrix R3::inverse(const R3::Matrix& A)
{
    gsl_matrix* gA = gsl_matrix_alloc(R3::Ndim, R3::Ndim);
    for (int i = 0; i != R3::Ndim; ++i)
    {
	for (int j = 0; j != R3::Ndim; ++j)
	{
	    gsl_matrix_set(gA, i, j, A(i,j));
	}
    }
    gsl_permutation* gP = gsl_permutation_alloc(R3::Ndim);
    int signum;
    gsl_linalg_LU_decomp(gA, gP, &signum);
    R3::Matrix B;
    gsl_matrix_view gB = gsl_matrix_view_array(B.data(), R3::Ndim, R3::Ndim);
    gsl_linalg_LU_invert(gA, gP, &gB.matrix);
    gsl_permutation_free(gP);
    gsl_matrix_free(gA);
    return B;
}


R3::Matrix R3::transpose(const R3::Matrix& A)
{
    R3::Matrix res;
    res = A(0,0), A(1,0), A(2,0),
          A(0,1), A(1,1), A(2,1),
          A(0,2), A(1,2), A(2,2);
    return res;
}


// End of file
