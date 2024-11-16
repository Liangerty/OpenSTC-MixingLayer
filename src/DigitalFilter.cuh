#pragma once

#include <curand_kernel.h>
#include "Define.h"
#include "Parameter.h"
#include "Mesh.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct DBoundCond;

void initialize_digital_filter(Parameter &parameter, cfd::Mesh &mesh, DBoundCond &dBoundCond);

// Steps to initiate the digital filter

// 1. Compute the Lund matrix for the digital filter
void get_digital_filter_lund_matrix(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                    std::vector<std::vector<real>> &scaled_y);

void assume_gaussian_reynolds_stress(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                     std::vector<std::vector<real>> &y_scaled);

__global__ void
compute_lundMat_with_assumed_gaussian_reynolds_stress(const real *Rij,
                                                      ggxl::VectorField1D<real> *df_lundMatrix_hPtr,
                                                      int i_face, const real *y_scaled, int my, int ngg);

// 2. Compute the length scale for the digital filter
// Currently ignore this part. For mixing layers, the value is just the initial vorticity thickness

// 3. Compute the convolution kernel for the digital filter
void get_digital_filter_convolution_kernel(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                           std::vector<std::vector<real>> &y_scaled, real dz);

__global__ void compute_convolution_kernel(const real *y_scaled, ggxl::VectorField2D<real> *df_by,
                                           ggxl::VectorField2D<real> *df_bz, int iFace, int my, real dz_scaled,
                                           int ngg);

// 4. Initialize the random number states for the digital filter
void initialize_digital_filter_random_number_states(cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                                    std::vector<int> &N2, int ngg);

// 5. Compute the velocity fluctuations with convolution kernel
void
apply_convolution(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                  std::vector<int> &N2, int ngg);

// 7. compute the fluctuations!
void compute_fluctuations(cfd::DBoundCond &dBoundCond, std::vector<int> &N1, std::vector<int> &N2, int ngg);


__global__ void
compute_fluctuations(ggxl::VectorField3D<real> *fluctuation_dPtr, ggxl::VectorField1D<real> *lundMatrix_dPtr,
                     ggxl::VectorField2D<real> *velFluc_dPtr, int iFace, int my, int mz, int ngg);

}