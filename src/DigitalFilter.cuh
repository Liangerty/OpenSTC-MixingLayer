#pragma once

#include <curand_kernel.h>
#include "Define.h"
#include "Parameter.h"
#include "Mesh.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct DBoundCond;

struct DParameter;

struct DZone;

struct Inflow;

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

__global__ void compute_convolution_kernel(const real *y_scaled, ggxl::VectorField2D<real> *df_by,
                                           ggxl::VectorField2D<real> *df_bz, int iFace, int my, real dz_scaled,
                                           int ngg);

__global__ void
compute_fluctuations_first_step(ggxl::VectorField3D<real> *fluctuation_dPtr, ggxl::VectorField1D<real> *lundMatrix_dPtr,
                                ggxl::VectorField2D<real> *df_velFluc_old_dPtr, int iFace, int my, int mz, int ngg);

__global__ void
generate_random_numbers_kernel(ggxl::VectorField2D<curandState> *rng_states, int iFace, int my,
                               int mz, ggxl::VectorField2D<real> *random_numbers, int ngg);

__global__ void
remove_mean_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg);

__global__ void
apply_periodic_in_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg);

__global__ void perform_convolution_y(ggxl::VectorField2D<real> *random_numbers, ggxl::VectorField2D<real> *df_by,
                                      ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, int ngg);

__global__ void
perform_convolution_z(ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, ggxl::VectorField2D<real> *velFluc,
                      ggxl::VectorField2D<real> *df_bz, int ngg);

__global__ void
Castro_time_correlation_and_fluc_computation(ggxl::VectorField2D<real> *velFluc_old,
                                             ggxl::VectorField2D<real> *velFluc_new, int iFace, int my, int mz,
                                             int ngg, DParameter *param, DZone *zone, Inflow *inflow,
                                             ggxl::VectorField3D<real> *fluctuation_dPtr,
                                             ggxl::VectorField1D<real> *lundMatrix_dPtr);

// 2. Compute the length scale for the digital filter
// Currently ignore this part. For mixing layers, the value is just the initial vorticity thickness

// 3. Compute the convolution kernel for the digital filter
void get_digital_filter_convolution_kernel(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                           std::vector<std::vector<real>> &y_scaled, real dz);

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