#include "InviscidScheme.cuh"
#include "Mesh.h"
#include "Parameter.h"
#include "AWENO.cuh"
#include "Field.h"
#include "DParameter.cuh"
#include "Reconstruction.cuh"
#include "RiemannSolver.cuh"
#include "Parallel.h"

namespace cfd {
template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};
  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 3 + 1)) * sizeof(real); // pv[n_var]+metric[3]+jacobian
  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need also the mass fractions of species, which is not included in the n_var
    shared_mem += n_computation_per_block * parameter.get_int("n_spec") * sizeof(real);
  }

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const int tid = threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2];
  const int block_dim = blockDim.x * blockDim.y * blockDim.z;
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      riemannSolver_laxFriedrich<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      riemannSolver_hllc<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, const int idx_shared, DParameter *param) {
  auto n_var = param->n_var;
  if constexpr (mix_model == MixtureModel::FL) {
    n_var += param->n_spec;
  }
  switch (param->reconstruction) {
    case 2:
      MUSCL_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    case 3:
      NND2_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    default:
      first_order_reconstruct(pv, pv_l, pv_r, idx_shared, n_var);
  }
  if constexpr (mix_model != MixtureModel::Air) {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    const auto n_spec = param->n_spec;
    real mw_inv_l{0.0}, mw_inv_r{0.0};
    for (int l = 0; l < n_spec; ++l) {
      mw_inv_l += pv_l[5 + l] / param->mw[l];
      mw_inv_r += pv_r[5 + l] / param->mw[l];
    }
    const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
    const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

    real hl[MAX_SPEC_NUMBER], hr[MAX_SPEC_NUMBER], cpl_i[MAX_SPEC_NUMBER], cpr_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
    compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
    real cpl{0}, cpr{0}, cvl{0}, cvr{0};
    for (auto l = 0; l < n_spec; ++l) {
      cpl += cpl_i[l] * pv_l[l + 5];
      cpr += cpr_i[l] * pv_r[l + 5];
      cvl += pv_l[l + 5] * (cpl_i[l] - R_u / param->mw[l]);
      cvr += pv_r[l + 5] * (cpr_i[l] - R_u / param->mw[l]);
      el += hl[l] * pv_l[l + 5];
      er += hr[l] * pv_r[l + 5];
    }
    pv_l[n_var] = pv_l[0] * el - pv_l[4]; //total energy
    pv_r[n_var] = pv_r[0] * er - pv_r[4];

    pv_l[n_var + 1] = cpl / cvl; //specific heat ratio
    pv_r[n_var + 1] = cpr / cvr;
  } else {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    pv_l[n_var] = el * pv_l[0] + pv_l[4] / (gamma_air - 1);
    pv_r[n_var] = er * pv_r[0] + pv_r[4] / (gamma_air - 1);
  }
}

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                   const Parameter &parameter) {
  // The implementation of AWENO is based on Fig.9 of (Ye, C-C, Zhang, P-J-Y, Wan, Z-H, and Sun, D-J (2022)
  // An alternative formulation of targeted ENO scheme for hyperbolic conservation laws. Computers & Fluids, 238, 105368.
  // doi:10.1016/j.compfluid.2022.105368.)

  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real); // metric[3]
  auto shared_cds = block_dim * n_var * sizeof(real); // f_i

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);

    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 2 * block.ngg) + 1;

    dim3 BPG2(bpg[0], bpg[1], bpg[2]);
    CDSPart1D<mix_model><<<BPG2, TPB, shared_cds>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    constexpr int tpb[3]{1, 1, 64};
    int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);

    dim3 BPG2(extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 2 * block.ngg) + 1);
    CDSPart1D<mix_model><<<BPG2, TPB, shared_cds>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_aweno_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = (int) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = (int) (blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var + 2};
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];


  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
  cv[i_shared * n_reconstruct + n_var + 1] = zone->bv(idx[0], idx[1], idx[2], 5);
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  AWENO_interpolation<mix_model>(cv, pv_l, pv_r, i_shared, n_var, metric, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      riemannSolver_laxFriedrich<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      riemannSolver_hllc<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, const int n_var,
                               const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};

  // Compute the entropy fix delta
  dim3 thread_per_block{8, 8, 4};
  if (extent[2] == 1) {
    thread_per_block = {16, 16, 1};
  }
  dim3 block_per_grid{(extent[0] + 1) / thread_per_block.x + 1,
                      (extent[1] + 1) / thread_per_block.y + 1,
                      (extent[2] + 1) / thread_per_block.z + 1};
  compute_entropy_fix_delta<mix_model><<<block_per_grid, thread_per_block>>>(zone, param);

  constexpr int block_dim = 128;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 1)) * sizeof(real) // pv[n_var]+jacobian
                    + n_computation_per_block * 3 * sizeof(real) // metric[3]
                    + n_computation_per_block * sizeof(real); // entropy fix delta
  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need also the mass fractions of species, which is not included in the n_var
    shared_mem += n_computation_per_block * parameter.get_int("n_spec") * sizeof(real);
  }

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void compute_entropy_fix_delta(cfd::DZone *zone, DParameter *param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  const auto &bv{zone->bv};
  const auto &metric{zone->metric(i, j, k)};

  const real U = abs(bv(i, j, k, 1) * metric(1, 1) + bv(i, j, k, 2) * metric(1, 2) + bv(i, j, k, 3) * metric(1, 3));
  const real V = abs(bv(i, j, k, 1) * metric(2, 1) + bv(i, j, k, 2) * metric(2, 2) + bv(i, j, k, 3) * metric(2, 3));
  const real W = abs(bv(i, j, k, 1) * metric(3, 1) + bv(i, j, k, 2) * metric(3, 2) + bv(i, j, k, 3) * metric(3, 3));

  const real kx = sqrt(metric(1, 1) * metric(1, 1) + metric(1, 2) * metric(1, 2) + metric(1, 3) * metric(1, 3));
  const real ky = sqrt(metric(2, 1) * metric(2, 1) + metric(2, 2) * metric(2, 2) + metric(2, 3) * metric(2, 3));
  const real kz = sqrt(metric(3, 1) * metric(3, 1) + metric(3, 2) * metric(3, 2) + metric(3, 3) * metric(3, 3));

  if (param->dim == 2) {
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + zone->acoustic_speed(i, j, k) * 0.5 * (kx + ky));
  } else {
    // 3D
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + W + zone->acoustic_speed(i, j, k) * (kx + ky + kz) / 3.0);
  }
}

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = (int) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = (int) (blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *entropy_fix_delta = &jac[n_point];
  real *fc = &entropy_fix_delta[n_point];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
  entropy_fix_delta[i_shared] = zone->entropy_fix_delta(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    entropy_fix_delta[tid + ngg] = zone->entropy_fix_delta(idx[0] + labels[0], idx[1] + labels[1], idx[2] + labels[2]);
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  riemannSolver_Roe<mix_model>(zone, pv, tid, param, fc, metric, jac, entropy_fix_delta,
                               direction);
  __syncthreads();


  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
void compute_convective_term_ep(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                const Parameter &parameter) {
  // Implementation of the 6th-order EP scheme.
  // PIROZZOLI S. Stabilized non-dissipative approximations of Euler equations in generalized curvilinear coordinates[J/OL].
  // Journal of Computational Physics, 2011, 230(8): 2997-3014. DOI:10.1016/j.jcp.2011.01.001.

  const int extent[3]{block.mx, block.my, block.mz};
  const int ngg{block.ngg};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (3 + 1 + 1 + 1) // metric[3], jac, uk_jac, totalEnthalpy
                     + (block_dim + ngg - 1) * 3 * n_var // tilde operations
                    ) * sizeof(real);

  for (int dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_ep_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    constexpr int tpb[3]{1, 1, 64};
    int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_ep_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void compute_convective_term_ep_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const int tid = threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2];
  const int block_dim = blockDim.x * blockDim.y * blockDim.z;
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  auto &pv = zone->bv;
  const auto n_var{param->n_var};

  extern __shared__ real s[];
  real *metric = s;
  real *jac = &metric[n_point * 3];
  real *uk_jac = &jac[n_point];
  real *totalEnthalpy = &uk_jac[n_point];
  real *fc = &totalEnthalpy[n_point];
  real *tilde_op = &fc[block_dim * n_var];
  const int i_shared = tid - 1 + ngg;

  // compute the contra-variant velocity
  metric[i_shared * 3] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 1);
  metric[i_shared * 3 + 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 2);
  metric[i_shared * 3 + 2] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 3);
  uk_jac[i_shared] = metric[i_shared * 3] * pv(idx[0], idx[1], idx[2], 1)
                     + metric[i_shared * 3 + 1] * pv(idx[0], idx[1], idx[2], 2)
                     + metric[i_shared * 3 + 2] * pv(idx[0], idx[1], idx[2], 3);
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
  uk_jac[i_shared] *= jac[i_shared];
  totalEnthalpy[i_shared] =
      (zone->cv(idx[0], idx[1], idx[2], 4) + pv(idx[0], idx[1], idx[2], 4)) / pv(idx[0], idx[1], idx[2], 0);
  // ghost cells
  constexpr int max_additional_ghost_point_loaded = 3; // This is for 6th-order ep, with 3 ghost points on each side.
  int ig_shared[max_additional_ghost_point_loaded];
  int g_idx[max_additional_ghost_point_loaded][3];
  int additional_loaded{0};
  if (tid < ngg - 1) {
    ig_shared[additional_loaded] = tid;
    g_idx[additional_loaded][0] = idx[0] - (ngg - 1) * labels[0];
    g_idx[additional_loaded][1] = idx[1] - (ngg - 1) * labels[1];
    g_idx[additional_loaded][2] = idx[2] - (ngg - 1) * labels[2];
    ++additional_loaded;
  }
  if (tid > block_dim - ngg - 1 || idx[direction] > max_extent - ngg - 1) {
    int igShared = tid + 2 * ngg - 1;
    int rGhIdx[3]{idx[0] + ngg * labels[0], idx[1] + ngg * labels[1], idx[2] + ngg * labels[2]};
    metric[igShared * 3] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 1);
    metric[igShared * 3 + 1] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 2);
    metric[igShared * 3 + 2] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 3);
    uk_jac[igShared] = metric[igShared * 3] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 1)
                       + metric[igShared * 3 + 1] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 2)
                       + metric[igShared * 3 + 2] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 3);
    jac[igShared] = zone->jac(rGhIdx[0], rGhIdx[1], rGhIdx[2]);
    uk_jac[igShared] *= jac[igShared];
    totalEnthalpy[igShared] =
        (zone->cv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 4) + pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 4)) /
        pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 0);
  }
  if (idx[direction] == max_extent - 1 && tid < ngg - 1) {
    int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      ig_shared[additional_loaded] = tid + m + 1;
      g_idx[additional_loaded][0] = idx[0] - (ngg - 1 - m - 1) * labels[0];
      g_idx[additional_loaded][1] = idx[1] - (ngg - 1 - m - 1) * labels[1];
      g_idx[additional_loaded][2] = idx[2] - (ngg - 1 - m - 1) * labels[2];
      ++additional_loaded;
    }
    int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      int igShared = i_shared + m + 1;
      int rGhIdx[3]{idx[0] + (m + 1) * labels[0], idx[1] + (m + 1) * labels[1], idx[2] + (m + 1) * labels[2]};
      metric[igShared * 3] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 1);
      metric[igShared * 3 + 1] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 2);
      metric[igShared * 3 + 2] = zone->metric(rGhIdx[0], rGhIdx[1], rGhIdx[2])(direction + 1, 3);
      uk_jac[igShared] = metric[igShared * 3] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 1)
                         + metric[igShared * 3 + 1] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 2)
                         + metric[igShared * 3 + 2] * pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 3);
      jac[igShared] = zone->jac(rGhIdx[0], rGhIdx[1], rGhIdx[2]);
      uk_jac[igShared] *= jac[igShared];
      totalEnthalpy[igShared] =
          (zone->cv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 4) + pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 4)) /
          pv(rGhIdx[0], rGhIdx[1], rGhIdx[2], 0);
    }
  }
  for (int g = 0; g < additional_loaded; ++g) {
    metric[ig_shared[g] * 3] = zone->metric(g_idx[g][0], g_idx[g][1], g_idx[g][2])(direction + 1, 1);
    metric[ig_shared[g] * 3 + 1] = zone->metric(g_idx[g][0], g_idx[g][1], g_idx[g][2])(direction + 1, 2);
    metric[ig_shared[g] * 3 + 2] = zone->metric(g_idx[g][0], g_idx[g][1], g_idx[g][2])(direction + 1, 3);
    uk_jac[ig_shared[g]] = metric[ig_shared[g] * 3] * pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 1)
                           + metric[ig_shared[g] * 3 + 1] * pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 2)
                           + metric[ig_shared[g] * 3 + 2] * pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 3);
    jac[ig_shared[g]] = zone->jac(g_idx[g][0], g_idx[g][1], g_idx[g][2]);
    uk_jac[ig_shared[g]] *= jac[ig_shared[g]];
    totalEnthalpy[ig_shared[g]] =
        (zone->cv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 4) + pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 4)) /
        pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 0);
  }
  __syncthreads();

  for (int l = 1; l <= 3; ++l) {
    int l_idx[3]{idx[0], idx[1], idx[2]};
    l_idx[direction] += l;
    const real weight = 0.125 * (uk_jac[i_shared] + uk_jac[i_shared + l]) *
                        (pv(idx[0], idx[1], idx[2], 0) + pv(l_idx[0], l_idx[1], l_idx[2], 0));
    const real pWeight = 0.25 * (pv(idx[0], idx[1], idx[2], 4) + pv(l_idx[0], l_idx[1], l_idx[2], 4));
    tilde_op[i_shared * n_var * 3 + (l - 1) * n_var] = 2 * weight;
    tilde_op[i_shared * n_var * 3 + (l - 1) * n_var + 1] =
        weight * (pv(idx[0], idx[1], idx[2], 1) + pv(l_idx[0], l_idx[1], l_idx[2], 1)) +
        pWeight * (metric[i_shared * 3] * jac[i_shared] + metric[(i_shared + l) * 3] * jac[i_shared + l]);
    tilde_op[i_shared * n_var * 3 + (l - 1) * n_var + 2] =
        weight * (pv(idx[0], idx[1], idx[2], 2) + pv(l_idx[0], l_idx[1], l_idx[2], 2)) +
        pWeight * (metric[i_shared * 3 + 1] * jac[i_shared] + metric[(i_shared + l) * 3 + 1] * jac[i_shared + l]);
    tilde_op[i_shared * n_var * 3 + (l - 1) * n_var + 3] =
        weight * (pv(idx[0], idx[1], idx[2], 3) + pv(l_idx[0], l_idx[1], l_idx[2], 3)) +
        pWeight * (metric[i_shared * 3 + 2] * jac[i_shared] + metric[(i_shared + l) * 3 + 2] * jac[i_shared + l]);
    tilde_op[i_shared * n_var * 3 + (l - 1) * n_var + 4] =
        weight * (totalEnthalpy[i_shared] + totalEnthalpy[i_shared + l]);
  }
  for (int g = 0; g < additional_loaded; ++g) {
    for (int l = 1; l <= 3; ++l) {
      int l_idx[3]{g_idx[g][0], g_idx[g][1], g_idx[g][2]};
      l_idx[direction] += l;
      const real weight = 0.125 * (uk_jac[ig_shared[g]] + uk_jac[ig_shared[g] + l]) *
                          (pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 0) + pv(l_idx[0], l_idx[1], l_idx[2], 0));
      const real pWeight = 0.25 * (pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 4) + pv(l_idx[0], l_idx[1], l_idx[2], 4));
      tilde_op[ig_shared[g] * n_var * 3 + (l - 1) * n_var] = 2 * weight;
      tilde_op[ig_shared[g] * n_var * 3 + (l - 1) * n_var + 1] =
          weight * (pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 1) + pv(l_idx[0], l_idx[1], l_idx[2], 1)) +
          pWeight *
          (metric[ig_shared[g] * 3] * jac[ig_shared[g]] + metric[(ig_shared[g] + l) * 3] * jac[ig_shared[g] + l]);
      tilde_op[ig_shared[g] * n_var * 3 + (l - 1) * n_var + 2] =
          weight * (pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 2) + pv(l_idx[0], l_idx[1], l_idx[2], 2)) +
          pWeight * (metric[ig_shared[g] * 3 + 1] * jac[ig_shared[g]] +
                     metric[(ig_shared[g] + l) * 3 + 1] * jac[ig_shared[g] + l]);
      tilde_op[ig_shared[g] * n_var * 3 + (l - 1) * n_var + 3] =
          weight * (pv(g_idx[g][0], g_idx[g][1], g_idx[g][2], 3) + pv(l_idx[0], l_idx[1], l_idx[2], 3)) +
          pWeight * (metric[ig_shared[g] * 3 + 2] * jac[ig_shared[g]] +
                     metric[(ig_shared[g] + l) * 3 + 2] * jac[ig_shared[g] + l]);
      tilde_op[ig_shared[g] * n_var * 3 + (l - 1) * n_var + 4] =
          weight * (totalEnthalpy[ig_shared[g]] + totalEnthalpy[ig_shared[g] + l]);
    }
  }
  __syncthreads();

  constexpr real central_1[3]{0.75, -3.0 / 20, 1.0 / 60};
  for (int l = 0; l < n_var; ++l) {
    real dql = 2 * (
        central_1[0] * tilde_op[i_shared * n_var * 3 + l]
        + central_1[1] * (tilde_op[i_shared * n_var * 3 + n_var + l] + tilde_op[(i_shared - 1) * n_var * 3 + n_var + l])
        + central_1[2] * (tilde_op[i_shared * n_var * 3 + 2 * n_var + l] +
                          tilde_op[(i_shared - 1) * n_var * 3 + 2 * n_var + l] +
                          tilde_op[(i_shared - 2) * n_var * 3 + 2 * n_var + l]));
    fc[tid * n_var + l] = dql;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

// template instantiation
template void
compute_convective_term_pv<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                              const Parameter &parameter);

template void
compute_convective_term_pv<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                  int n_var, const Parameter &parameter);

template void
compute_convective_term_pv<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                          int n_var, const Parameter &parameter);

template void
compute_convective_term_pv<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                             const Parameter &parameter);

template void
compute_convective_term_pv<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                             const Parameter &parameter);

template void
compute_convective_term_aweno<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                                 const Parameter &parameter);

template void
compute_convective_term_aweno<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                     int n_var, const Parameter &parameter);

template void
compute_convective_term_aweno<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                             int n_var, const Parameter &parameter);

template void
compute_convective_term_aweno<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                                const Parameter &parameter);

template void
compute_convective_term_aweno<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                                const Parameter &parameter);

template
void Roe_compute_inviscid_flux<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                  const int n_var, const Parameter &parameter);

template
void Roe_compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                      const int n_var, const Parameter &parameter);

template
void Roe_compute_inviscid_flux<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                 const int n_var, const Parameter &parameter);

template<>
void Roe_compute_inviscid_flux<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                 const int n_var, const Parameter &parameter) {
  printf("Roe_compute_inviscid_flux<MixtureModel::FL> is not implemented yet.\n");
  MpiParallel::exit();
}

template<>
void Roe_compute_inviscid_flux<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                              const int n_var, const Parameter &parameter) {
  printf("Roe_compute_inviscid_flux<MixtureModel::MixtureFraction> is not implemented yet.\n");
  MpiParallel::exit();
}

template void
compute_convective_term_ep<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                              const Parameter &parameter);

template void
compute_convective_term_ep<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                  int n_var, const Parameter &parameter);

template void
compute_convective_term_ep<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                          int n_var, const Parameter &parameter);

template void
compute_convective_term_ep<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                             const Parameter &parameter);

template void
compute_convective_term_ep<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                             const Parameter &parameter);
}