#include "DualTimeStepping.cuh"

__global__ void cfd::compute_qn_star(cfd::DZone *zone, integer n_var, real dt_global) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real factor = 0.5 * zone->jac(i, j, k) / dt_global;
  for (integer l = 0; l < n_var; ++l) {
    zone->qn_star(i, j, k, l) = factor * (4 * zone->cv(i, j, k, l) - zone->qn1(i, j, k, l));
  }
}

__global__ void
cfd::compute_modified_rhs(cfd::DZone *zone, integer n_var, real dt_global) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &dq = zone->dq;
  auto &q_star = zone->qn_star;
  const real factor = 1.5 * zone->jac(i, j, k) / dt_global;
  for (integer l = 0; l < n_var; ++l) {
    dq(i, j, k, l) += q_star(i, j, k, l) - factor * zone->cv(i, j, k, l);
  }
}

bool cfd::inner_converged(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, integer iter,
                          std::array<real, 4> &res_scale, integer myid, integer step, integer &inner_iter) {
  const integer n_block = mesh.n_block;

  std::array<real, 4> res{0, 0, 0, 0};
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    // compute the square of the difference of the basic variables
    compute_square_of_dbv_wrt_last_inner_iter<<<bpg, tpb>>>(field[b].d_ptr);
  }
  constexpr integer TPB{128};
  constexpr integer n_res_var{4};
  real res_block[n_res_var];
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
                                                TPB * sizeof(real) * n_res_var);
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB, TPB * sizeof(real) * n_res_var>>>(
        field[b].h_ptr->in_last_step.data(), size);
    reduction_of_dv_squared<n_res_var> <<<1, TPB, TPB * sizeof(real) * n_res_var>>>(
        field[b].h_ptr->in_last_step.data(), n_blocks);
    cudaMemcpy(res_block, field[b].h_ptr->in_last_step.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
    for (integer l = 0; l < n_res_var; ++l) {
      res[l] += res_block[l];
    }
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    static std::array<double, 4> res_temp;
    for (int i = 0; i < 4; ++i) {
      res_temp[i] = res[i];
    }
    MPI_Allreduce(res_temp.data(), res.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  for (auto &e: res) {
    e = std::sqrt(e / mesh.n_grid_total);
  }

  if (iter == 1) {
    for (integer i = 0; i < n_res_var; ++i) {
      res_scale[i] = res[i];
      if (res_scale[i] < 1e-20) {
        res_scale[i] = 1e-20;
      }
    }
    if (myid == 0) {
      printf("******************************\nStep %d.\n", step);
    }
  }

  for (integer i = 0; i < 4; ++i) {
    res[i] /= res_scale[i];
  }

  // Find the maximum error of the 4 errors
  real err_max = res[0];
  for (integer i = 1; i < 4; ++i) {
    if (std::abs(res[i]) > err_max) {
      err_max = res[i];
    }
  }

  if (myid == 0) {
    if (isnan(err_max)) {
      printf("Nan occurred in iter %d of step %d. Stop simulation.\n", iter, step);
      cfd::MpiParallel::exit();
    }
    if (iter % 5 == 0 || err_max < 1e-3)
      printf("iter %d, err_max=%e\n", iter, err_max);
  }

  if (iter == inner_iter) {
    const integer INNER_ITER_MAX = parameter.get_int("max_inner_iteration");
    if (err_max > 1e-3 && inner_iter < INNER_ITER_MAX) {
      inner_iter += 5;
      inner_iter = std::min(inner_iter, INNER_ITER_MAX);
      if (myid == 0) {
        printf("Inner iteration step is increased to %d.\n", inner_iter);
      }
    }
  }

  if (err_max < 1e-3) {
    if (auto minus = inner_iter - iter;minus > 0) {
      inner_iter -= minus;
      if (myid == 0) {
        printf("Inner iteration step is decreased to %d.\n", inner_iter);
      }
    }
    return true;
  }

  return false;
}

__global__ void cfd::compute_square_of_dbv_wrt_last_inner_iter(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->in_last_step;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  bv_last(i, j, k, 1) = (zone->vel(i, j, k) - bv_last(i, j, k, 1)) * (zone->vel(i, j, k) - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

__global__ void cfd::store_last_iter(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  zone->in_last_step(i, j, k, 0) = zone->bv(i, j, k, 0);
  zone->in_last_step(i, j, k, 1) = zone->vel(i, j, k);
  zone->in_last_step(i, j, k, 2) = zone->bv(i, j, k, 4);
  zone->in_last_step(i, j, k, 3) = zone->bv(i, j, k, 5);
}
