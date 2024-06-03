#include "FieldOperation.cuh"

__device__ void
cfd::compute_temperature_and_pressure(int i, int j, int k, const DParameter *param, DZone *zone, real total_energy) {
  const int n_spec = param->n_spec;
  auto &Y = zone->sv;
  auto &bv = zone->bv;

  real mw{0};
  for (int l = 0; l < n_spec; ++l) {
    mw += Y(i, j, k, l) / param->mw[l];
  }
  mw = 1 / mw;
  const real gas_const = R_u / mw;
  const real e =
      total_energy / bv(i, j, k, 0) - 0.5 * (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
                                             bv(i, j, k, 3) * bv(i, j, k, 3));

  real err{1}, t{bv(i, j, k, 5)};
  constexpr int max_iter{1000};
  constexpr real eps{1e-3};
  int iter = 0;

  real h_i[MAX_SPEC_NUMBER], cp_i[MAX_SPEC_NUMBER];
  while (err > eps && iter++ < max_iter) {
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cp_tot{0}, h{0};
    for (int l = 0; l < n_spec; ++l) {
      cp_tot += cp_i[l] * Y(i, j, k, l);
      h += h_i[l] * Y(i, j, k, l);
    }
    const real e_t = h - gas_const * t;
    const real cv = cp_tot - gas_const;
    const real t1 = t - (e_t - e) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  bv(i, j, k, 5) = t;
  bv(i, j, k, 4) = bv(i, j, k, 0) * t * gas_const;
}

__global__ void cfd::compute_velocity(cfd::DZone *zone) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const auto &bv = zone->bv;

  zone->vel(i, j, k) = std::sqrt(
      bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
}
