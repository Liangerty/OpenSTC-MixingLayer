#include "TurbInflow.cuh"
#include "BoundCond.cuh"
#include <ctime>

void cfd::initialize_digital_filter(Parameter &parameter, Mesh &mesh, DBoundCond &dBoundCond) {
  if (parameter.get_int("problem_type") != 1) {
    // The current implementation is only for mixing layers
    return;
  }

  int n_df = 0;
  std::vector<int> df_label;
  std::vector<std::string> df_related_boundary = {};
  auto &bcs = parameter.get_string_array("boundary_conditions");
  dBoundCond.df_label.resize(dBoundCond.n_inflow, -1);
  int i_inflow = 0;
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto &type = std::get<std::string>(bc.at("type"));

    if (type == "inflow") {
      int fluc = std::get<int>(bc.at("fluctuation_type"));
      if (fluc == 11) {
        ++n_df;
        df_label.push_back(std::get<int>(bc.at("label")));
        df_related_boundary.push_back(bc_name);
        dBoundCond.df_label[i_inflow] = i_inflow;
      }
      ++i_inflow;
    }
  }
  parameter.update_parameter("df_label", df_label);

  if (n_df == 0) {
    return;
  }

  // Careful! The current implementation is based on the Touber's implementation, the z direction is assumed to be periodic.
  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    printf("\tInitialize digital filter for turbulence inflow generation.\n");
    printf("\tThe z direction is assumed to be periodic, and the inflow is assumed to be in the yz(const x) plane!\n");
  }

  // Although the implementation below is for multiple boundaries, the current implementation only supports one boundary.
  std::vector<int> N1, N2, iBlock;
  std::vector<std::vector<real>> scaled_y;
  real y_ref{0}, dz{0};
  const int ngg{parameter.get_int("ngg")};
  if (parameter.get_int("problem_type") == 1) {
    y_ref = parameter.get_real("delta_omega");
  }
  for (int type = 0; type < n_df; ++type) {
    int label = df_label[type];
    const auto &boundary_name = df_related_boundary[type];
    // Find the corresponding boundary faces
    for (auto blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (auto &b: bs) {
        if (label == b.type_label) {
          ++dBoundCond.n_df_face;
          iBlock.push_back(blk);

          int n1{mesh[blk].my}, n2{mesh[blk].mz};
          if (b.face == 1) {
            printf("Boundary %s is on the xz plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
            // n1 = mesh[blk].mx;
          } else if (b.face == 2) {
            printf("Boundary %s is on the xy plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
            // n1 = mesh[blk].mx;
            // n2 = mesh[blk].my;
          }

          int i = b.direction == 1 ? mesh[blk].mx - 1 : 0;
          std::vector<real> scaled_y_temp(mesh[blk].my + 2 * ngg);
          for (int j = -ngg; j < mesh[blk].my + ngg; ++j) {
            scaled_y_temp[j + ngg] = mesh[blk].y(i, j, 0) / y_ref;
          }
          scaled_y.push_back(scaled_y_temp);
          for (int k = 1; k < n2; ++k) {
            dz += mesh[blk].z(i, 0, k) - mesh[blk].z(i, 0, k - 1);
          }
          dz /= (n2 - 1);

          printf("\tInflow boundary %s with (%d, %d) grid points for digital filter on  process %d, dz = %f.\n",
                 boundary_name.c_str(), n1, n2, myid, dz);

          N1.push_back(n1);
          N2.push_back(n2);
        }
      }
    }
  }
  if (dBoundCond.n_df_face == 0) {
    // When parallel, there may be only one inflow plane in a single processor
    return;
  }

  dBoundCond.initialize_df_memory(mesh, parameter, N1, N2);
  if (myid == 0)
    printf("\tThe memory for the digital filter is allocated.\n");

  // compute lund matrix
  get_digital_filter_lund_matrix(parameter, dBoundCond, N1, scaled_y);
  if (myid == 0)
    printf("\tThe Lund matrix for the digital filter is computed.\n");

  // compute convolution kernel
  get_digital_filter_convolution_kernel(parameter, dBoundCond, N1, scaled_y, dz);
  if (myid == 0)
    printf("\tThe convolution kernel for the digital filter is computed.\n");

  // initialize the random number states
  initialize_digital_filter_random_number_states(dBoundCond, N1, N2, ngg);
  if (myid == 0)
    printf("\tThe random number states for the digital filter are initialized.\n");

  // Do the same for a new velocity fluctuation
  for (int iFace = 0; iFace < dBoundCond.n_df_face; ++iFace) {
    dBoundCond.generate_random_numbers(N1[iFace], N2[iFace], ngg, iFace);
    dBoundCond.apply_convolution(N1[iFace], N2[iFace], ngg, iFace);
    int sz = N1[iFace] * N2[iFace] * 3;
    cudaMemcpy(dBoundCond.df_velFluc_old_hPtr[iFace].data(), dBoundCond.df_velFluc_new_hPtr[iFace].data(),
               sz * sizeof(real), cudaMemcpyDeviceToDevice);
  }

  // generate the fluctuation profile
  std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr(dBoundCond.n_df_face);
  for (int iFace = 0; iFace < dBoundCond.n_df_face; ++iFace) {
    // This assumes a constant i face
    fluctuation_hPtr[iFace].allocate_memory(1, N1[iFace], N2[iFace], 3, mesh.ngg);
  }
  cudaMalloc(&dBoundCond.fluctuation_dPtr, dBoundCond.n_df_face * sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(dBoundCond.fluctuation_dPtr, fluctuation_hPtr.data(),
             dBoundCond.n_df_face * sizeof(ggxl::VectorField3D<real>),
             cudaMemcpyHostToDevice);
  compute_fluctuations(dBoundCond, N1, N2, mesh.ngg);
  if (myid == 0)
    printf("\tThe velocity fluctuations are computed.\n");

  parameter.update_parameter("update_df", false);
}

void
cfd::get_digital_filter_lund_matrix(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                    std::vector<std::vector<real>> &scaled_y) {
  int method = parameter.get_int("reynolds_stress_supplier");
  if (method == 2) {
//    for (int iFace = 0; iFace < iBlock.size(); ++iFace) {
    // Assume the Reynolds stress is Gaussian
    assume_gaussian_reynolds_stress(parameter, dBoundCond, N1, scaled_y);
//    }
  } else {
    printf("The method %d is not supported for the Reynolds stress supplier.\n", method);
    MpiParallel::exit();
  }
}

void
cfd::assume_gaussian_reynolds_stress(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                     std::vector<std::vector<real>> &y_scaled) {
  auto Rij = parameter.get_real_array("df_reynolds_gaussian_peak");

  real *Rij_dPtr;
  cudaMalloc(&Rij_dPtr, 6 * sizeof(real));
  cudaMemcpy(Rij_dPtr, Rij.data(), 6 * sizeof(real), cudaMemcpyHostToDevice);

  // Assume the Reynolds stress is Gaussian
  const int ngg = parameter.get_int("ngg");
  for (int i = 0; i < dBoundCond.n_df_face; ++i) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[i].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[i].data(), y_scaled[i].size() * sizeof(real), cudaMemcpyHostToDevice);

    int TPB = 512;
    int BPG = (N1[i] + 2 * ngg + TPB - 1) / TPB;
    compute_lundMat_with_assumed_gaussian_reynolds_stress<<<BPG, TPB>>>(Rij_dPtr, dBoundCond.df_lundMatrix_dPtr, i,
                                                                        y_scaled_dPtr, N1[i], ngg);

    cudaFree(y_scaled_dPtr);
  }
  cudaFree(Rij_dPtr);
}

__global__ void
cfd::compute_lundMat_with_assumed_gaussian_reynolds_stress(const real *Rij,
                                                           ggxl::VectorField1D<real> *df_lundMatrix_hPtr,
                                                           int i_face, const real *y_scaled, int my, int ngg) {
  int j = (int) (blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) {
    return;
  }

  auto &mat = df_lundMatrix_hPtr[i_face];
  real y_ = y_scaled[j + ngg]; // The first element is the ghost cell, the last element is the ghost cell

  // Assume the Reynolds stress is Gaussian, Rij[0] is the peak value, sigma = 1/2
  real gaussian_y = exp(-y_ * y_ * 0.5 * 4);

  real R11_y = Rij[0] * gaussian_y;
  real R12_y = Rij[1] * gaussian_y;
  real R22_y = Rij[2] * gaussian_y;
  real R13_y = Rij[3] * gaussian_y;
  real R23_y = Rij[4] * gaussian_y;
  real R33_y = Rij[5] * gaussian_y;
//  printf("R(%d, 0:5) = (%e, %e, %e, %e, %e, %e)\n", j, R11_y, R12_y, R22_y, R13_y, R23_y, R33_y);

  mat(j, 0) = sqrt(abs(R11_y)); // a(1, 1)
  mat(j, 1) = mat(j, 0) < 1e-40 ? 0 : R12_y / mat(j, 0); // a(2, 1)
  mat(j, 2) = sqrt(abs(R22_y - mat(j, 1) * mat(j, 1))); // a(2, 2)
  mat(j, 3) = mat(j, 0) < 1e-40 ? 0 : R13_y / mat(j, 0); // a(3, 1)
  mat(j, 4) = mat(j, 2) < 1e-40 ? 0 : (R23_y - mat(j, 3) * mat(j, 1)) / mat(j, 2); // a(3, 2)
  mat(j, 5) = sqrt(abs(R33_y - mat(j, 3) * mat(j, 3) - mat(j, 4) * mat(j, 4))); // a(3, 3)
//  printf("Lund(%d, 0:5) = (%e, %e, %e, %e, %e, %e)\n", j, mat(j, 0), mat(j, 1), mat(j, 2), mat(j, 3), mat(j, 4),
//         mat(j, 5));
}

void
cfd::get_digital_filter_convolution_kernel(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                           std::vector<std::vector<real>> &y_scaled, real dz) {
  // For mixing layer, the length scale is the initial vorticity thickness for 3 directions
  const real DF_IntegralLength = parameter.get_real("delta_omega");

  const int ngg = parameter.get_int("ngg");

  for (int iFace = 0; iFace < dBoundCond.n_df_face; ++iFace) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[iFace].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[iFace].data(), y_scaled[iFace].size() * sizeof(real), cudaMemcpyHostToDevice);

    int TPB = 512;
    int BPG = (N1[iFace] + 2 * ngg + TPB - 1) / TPB;
    compute_convolution_kernel<<<BPG, TPB>>>(y_scaled_dPtr, dBoundCond.df_by_dPtr, dBoundCond.df_bz_dPtr, iFace,
                                             N1[iFace], dz / DF_IntegralLength, ngg);
    cudaFree(y_scaled_dPtr);
  }
}

__global__ void cfd::compute_convolution_kernel(const real *y_scaled, ggxl::VectorField2D<real> *df_by,
                                                ggxl::VectorField2D<real> *df_bz, int iFace, int my,
                                                real dz_scaled, int ngg) {
  int j = int(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) {
    return;
  }

  real dy;
  if (j == -ngg) {
    dy = y_scaled[2 * ngg + 1] - y_scaled[2 * ngg];
  } else {
    dy = y_scaled[j + ngg] - y_scaled[j + ngg - 1];
  }
  // y filter coefficients
  // The length scales for 3 directions are the same, which is the initial vorticity thickness
  // Therefore, here we only compute 1 direction, and assign the same value to the other 2 directions
  real sum = 0;
  // the y has been scaled by the initial vorticity thickness; therefore, the dy/delta_omega here is just dy
  real expMulti = -pi * dy/* / DF_IntegralLength*/;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    real expValue = exp(expMulti * abs(jf));
    df_by[iFace](j, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    real value = df_by[iFace](j, jf, 0) / sum;
    df_by[iFace](j, jf, 0) = value;
    df_by[iFace](j, jf, 1) = value;
    df_by[iFace](j, jf, 2) = value;
  }

  // z filter coefficients
  sum = 0;
  expMulti = -pi * dz_scaled;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    real expValue = exp(expMulti * abs(jf));
    df_bz[iFace](j, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    real value = df_bz[iFace](j, jf, 0) / sum;
    df_bz[iFace](j, jf, 0) = value;
    df_bz[iFace](j, jf, 1) = value;
    df_bz[iFace](j, jf, 2) = value;
  }
}

void cfd::initialize_digital_filter_random_number_states(cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                                                         std::vector<int> &N2, int ngg) {
  for (int i = 0; i < dBoundCond.n_df_face; ++i) {
    int sz = (N1[i] + 2 * DBoundCond::DF_N + 2 * ngg) * (N2[i] + 2 * DBoundCond::DF_N + 2 * ngg) * 3;
    dim3 TPB = {1024, 1, 1};
    dim3 BPG = {(sz - 1) / TPB.x + 1, 1, 1};
    // Get the current time
    time_t time_curr;
    initialize_rng<<<BPG, TPB>>>(dBoundCond.rng_states_hPtr[i].data(), sz, time(&time_curr));
  }
}

void cfd::time_correlation(cfd::Parameter &parameter, cfd::DBoundCond &dBoundCond, std::vector<int> &N1,
                           std::vector<int> &N2,
                           real dt) {

}

void cfd::compute_fluctuations(cfd::DBoundCond &dBoundCond, std::vector<int> &N1, std::vector<int> &N2, int ngg) {
  for (int iFace = 0; iFace < dBoundCond.n_df_face; ++iFace) {
    int my = N1[iFace], mz = N2[iFace];
    dim3 TPB(32, 32);
    dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
    compute_fluctuations<<<BPG, TPB>>>(dBoundCond.fluctuation_dPtr, dBoundCond.df_lundMatrix_dPtr,
                                       dBoundCond.df_velFluc_new_dPtr, iFace, my, mz, ngg);
  }
}

__global__ void
cfd::compute_fluctuations(ggxl::VectorField3D<real> *fluctuation_dPtr, ggxl::VectorField1D<real> *lundMatrix_dPtr,
                          ggxl::VectorField2D<real> *velFluc_dPtr, int iFace, int my, int mz, int ngg) {
  int j = int(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  int k = int(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) {
    return;
  }

  auto &fluc = fluctuation_dPtr[iFace];
  auto &lund = lundMatrix_dPtr[iFace];
  auto &velGen = velFluc_dPtr[iFace];

  fluc(0, j, k, 0) = lund(j, 0) * velGen(j, k, 0);
  fluc(0, j, k, 1) = lund(j, 1) * velGen(j, k, 0) + lund(j, 2) * velGen(j, k, 1);
  fluc(0, j, k, 2) = lund(j, 3) * velGen(j, k, 0) + lund(j, 4) * velGen(j, k, 1) + lund(j, 5) * velGen(j, k, 2);
//  if (j == 91)
//    printf("vFluc(%d, %d, u:w) = (%e, %e, %e)\n", j, k, fluc(0, j, k, 0), fluc(0, j, k, 1), fluc(0, j, k, 2));
  __syncthreads();
}

