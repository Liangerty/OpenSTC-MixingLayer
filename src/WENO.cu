#include "InviscidScheme.cuh"
#include "Constants.h"
#include "Parallel.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Field.h"

namespace cfd {

template<MixtureModel mix_model>
void compute_convective_term_weno(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                  const Parameter &parameter) {
  // The implementation of classic WENO.
  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * n_var * 2 // F+/F-
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real); // metric[3]
  if (parameter.get_bool("positive_preserving")) {
    shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
  }

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_weno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    constexpr int tpb[3]{1, 1, 64};
    int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_weno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_weno_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
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
  real *Fp = &jac[n_point];
  real *Fm = &Fp[n_point * n_var];
  real *fc = &Fm[n_point * n_var];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
  cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(idx[0], idx[1], idx[2]);
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
  int ig_shared[max_additional_ghost_point_loaded];
  int additional_loaded{0};
  if (tid < ngg - 1) {
    ig_shared[additional_loaded] = tid;
    const int g_idx[3]{idx[0] - (ngg - 1) * labels[0], idx[1] - (ngg - 1) * labels[1],
                           idx[2] - (ngg - 1) * labels[2]};

    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
    cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
    for (auto l = 1; l < 4; ++l) {
      metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
    }
    jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    ++additional_loaded;
  }
  if (tid > block_dim - ngg - 1 || idx[direction] > max_extent - ngg - 1) {
    ig_shared[additional_loaded] = tid + 2 * ngg - 1;
    const int g_idx[3]{idx[0] + ngg * labels[0], idx[1] + ngg * labels[1], idx[2] + ngg * labels[2]};
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
    cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
    for (auto l = 1; l < 4; ++l) {
      metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
    }
    jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    ++additional_loaded;
  }
  if (idx[direction] == max_extent - 1 && tid < ngg - 1) {
    int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      ig_shared[additional_loaded] = tid + m + 1;
      const int g_idx[3]{idx[0] - (ngg - 1 - m - 1) * labels[0], idx[1] - (ngg - 1 - m - 1) * labels[1],
                             idx[2] - (ngg - 1 - m - 1) * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
      ++additional_loaded;
    }
    int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      ig_shared[additional_loaded] = i_shared + m + 1;
      const int g_idx[3]{idx[0] + (m + 1) * labels[0], idx[1] + (m + 1) * labels[1], idx[2] + (m + 1) * labels[2]};
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
      ++additional_loaded;
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (auto sch = param->inviscid_scheme; sch == 51) {
    compute_weno_flux_cp<mix_model>(cv, param, tid, metric, jac, fc, i_shared, Fp, Fm, ig_shared, additional_loaded,
                                    f_1st);
  } else if (sch == 52) {
    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, Fp, Fm, ig_shared, additional_loaded,
                                    f_1st);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(idx[0], idx[1], idx[2]);
    positive_preserving_limiter<mix_model>(f_1st, n_var, tid, fc, param, i_shared, dt, idx[direction], max_extent, cv,
                                           jac);
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
compute_flux(const real *Q, DParameter *param, const real *metric, real jac, real *Fk) {
  const int n_var = param->n_var;
  real jacUk{jac * (metric[0] * Q[1] + metric[1] * Q[2] + metric[2] * Q[3]) / Q[0]};
  real pk{Q[n_var]};

  Fk[0] = jacUk * Q[0];
  Fk[1] = jacUk * Q[1] + jac * pk * metric[0];
  Fk[2] = jacUk * Q[2] + jac * pk * metric[1];
  Fk[3] = jacUk * Q[3] + jac * pk * metric[2];
  Fk[4] = jacUk * (Q[4] + pk);

  for (int l = 5; l < n_var; ++l) {
    Fk[l] = jacUk * Q[l];
  }
}

template<MixtureModel mix_model>
__device__ void compute_flux(const real *Q, DParameter *param, const real *metric, real jac, real *Fp, real *Fm) {
  const int n_var = param->n_var;
  const real Uk{(Q[1] * metric[0] + Q[2] * metric[1] + Q[3] * metric[2]) / Q[0]};
  real pk{Q[n_var]};
  const real cGradK = Q[n_var + 1] * sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
  const real lambda0 = abs(Uk + cGradK);

  Fp[0] = 0.5 * jac * (Uk * Q[0] + lambda0 * Q[0]);
  Fp[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] + lambda0 * Q[1]);
  Fp[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] + lambda0 * Q[2]);
  Fp[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] + lambda0 * Q[3]);
  Fp[4] = 0.5 * jac * (Uk * (Q[4] + pk) + lambda0 * Q[4]);

  Fm[0] = 0.5 * jac * (Uk * Q[0] - lambda0 * Q[0]);
  Fm[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] - lambda0 * Q[1]);
  Fm[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] - lambda0 * Q[2]);
  Fm[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] - lambda0 * Q[3]);
  Fm[4] = 0.5 * jac * (Uk * (Q[4] + pk) - lambda0 * Q[4]);

  for (int l = 5; l < n_var; ++l) {
    Fp[l] = 0.5 * jac * (Uk * Q[l] + lambda0 * Q[l]);
    Fm[l] = 0.5 * jac * (Uk * Q[l] - lambda0 * Q[l]);
  }
}

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_ch(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st) {
  const int n_var = param->n_var;

  // 0: acans; 1: li xinliang(own flux splitting); 2: my(same spectral radius)
  constexpr int method = 1;

  if constexpr (method == 1) {
    compute_flux<mix_model>(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared],
                            &Fp[i_shared * n_var], &Fm[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux<mix_model>(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3],
                              jac[ig_shared[i]], &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
    }
  } else if constexpr (method == 2) {
    compute_flux<mix_model>(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared],
                            &Fp[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux<mix_model>(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3],
                              jac[ig_shared[i]], &Fp[ig_shared[i] * n_var]);
    }
  }

  // The first n_var in the cv array is conservative vars, followed by p and cm.
  const real *cvl{&cv[i_shared * (n_var + 2)]};
  const real *cvr{&cv[(i_shared + 1) * (n_var + 2)]};
  // First, compute the Roe average of the half-point variables.
  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
  const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
  const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
  const real hl{(cvl[4] + cvl[n_var]) / cvl[0]};
  const real hr{(cvr[4] + cvr[n_var]) / cvr[0]};
  const real hm{rlc * hl + rrc * hr};

  real svm[MAX_SPEC_NUMBER];
  memset(svm, 0, MAX_SPEC_NUMBER * sizeof(real));
  for (int l = 0; l < n_var - 5; ++l) {
    svm[l] = (rlc * cvl[l + 5] / cvl[0] + rrc * cvr[l + 5] / cvr[0]);
  }

  const int n_spec{param->n_spec};
  real mw_inv = 0;
  for (int l = 0; l < n_spec; ++l) {
    mw_inv += svm[l] / param->mw[l];
  }

  const real tl{cvl[n_var] / cvl[0]};
  const real tr{cvr[n_var] / cvr[0]};
  const real tm = (rlc * tl + rrc * tr) / (R_u * mw_inv);

  real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
  compute_enthalpy_and_cp(tm, h_i, cp_i, param);
  real cp{0}, cv_tot{0};
  for (int l = 0; l < n_spec; ++l) {
    cp += svm[l] * cp_i[l];
    cv_tot += svm[l] * (cp_i[l] - R_u / param->mw[l]);
  }
  const real gamma = cp / cv_tot;
  const real cm = sqrt(gamma * R_u * mw_inv * tm);
  const real gm1{gamma - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
//  real kx{0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3])};
//  real ky{0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1])};
//  real kz{0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2])};
  real kx{(jac[i_shared] * metric[i_shared * 3] + jac[i_shared + 1] * metric[(i_shared + 1) * 3]) /
          (jac[i_shared] + jac[i_shared + 1])};
  real ky{(jac[i_shared] * metric[i_shared * 3 + 1] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1]) /
          (jac[i_shared] + jac[i_shared + 1])};
  real kz{(jac[i_shared] * metric[i_shared * 3 + 2] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2]) /
          (jac[i_shared] + jac[i_shared + 1])};
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  const real cm2_inv{1.0 / (cm * cm)};
  gxl::Matrix<real, 5, 5> LR;
  LR(0, 0) = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
  LR(0, 1) = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
  LR(0, 2) = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
  LR(0, 3) = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
  LR(0, 4) = gm1 * cm2_inv * 0.5;
  LR(1, 0) = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
  LR(1, 1) = kx * gm1 * um * cm2_inv;
  LR(1, 2) = (kx * gm1 * vm + kz * cm) * cm2_inv;
  LR(1, 3) = (kx * gm1 * wm - ky * cm) * cm2_inv;
  LR(1, 4) = -kx * gm1 * cm2_inv;
  LR(2, 0) = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
  LR(2, 1) = (ky * gm1 * um - kz * cm) * cm2_inv;
  LR(2, 2) = ky * gm1 * vm * cm2_inv;
  LR(2, 3) = (ky * gm1 * wm + kx * cm) * cm2_inv;
  LR(2, 4) = -ky * gm1 * cm2_inv;
  LR(3, 0) = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
  LR(3, 1) = (kz * gm1 * um + ky * cm) * cm2_inv;
  LR(3, 2) = (kz * gm1 * vm - kx * cm) * cm2_inv;
  LR(3, 3) = kz * gm1 * wm * cm2_inv;
  LR(3, 4) = -kz * gm1 * cm2_inv;
  LR(4, 0) = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
  LR(4, 1) = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
  LR(4, 2) = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
  LR(4, 3) = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
  LR(4, 4) = gm1 * cm2_inv * 0.5;

  // Compute the characteristic flux with L.
  real fChar[5 + MAX_SPEC_NUMBER];
  constexpr real eps{1e-40};
  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 *
                          ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
                           (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));

  real alpha_l[MAX_SPEC_NUMBER];
  // compute the partial derivative of pressure to species density
  for (int l = 0; l < n_spec; ++l) {
    alpha_l[l] = gamma * R_u / param->mw[l] * tm - (gamma - 1) * h_i[l];
    // The computations including this alpha_l are all combined with a division by cm2.
    alpha_l[l] *= cm2_inv;
  }

  if constexpr (method == 0) {
    real ap[3], an[3];
    const auto max_spec_rad = abs(Uk_bar) + cm;
    ap[0] = 0.5 * (Uk_bar - cm + max_spec_rad) * gradK;
    ap[1] = 0.5 * (Uk_bar + max_spec_rad) * gradK;
    ap[2] = 0.5 * (Uk_bar + cm + max_spec_rad) * gradK;
    an[0] = 0.5 * (Uk_bar - cm - max_spec_rad) * gradK;
    an[1] = 0.5 * (Uk_bar - max_spec_rad) * gradK;
    an[2] = 0.5 * (Uk_bar + cm - max_spec_rad) * gradK;

    printf("Not implemented.\n");
  } else if constexpr (method == 1) {
    // Li Xinliang's flux splitting
    bool pp_limiter{param->positive_preserving};
    if (pp_limiter) {
      real spectralRadThis = abs((metric[i_shared * 3] * cv[i_shared * (n_var + 2) + 1] +
                                  metric[i_shared * 3 + 1] * cv[i_shared * (n_var + 2) + 2] +
                                  metric[i_shared * 3 + 2] * cv[i_shared * (n_var + 2) + 3]) /
                                 cv[i_shared * (n_var + 2)] + cv[i_shared * (n_var + 2) + n_var + 1] * sqrt(
          metric[(i_shared) * 3] * metric[(i_shared) * 3] + metric[(i_shared) * 3 + 1] * metric[(i_shared) * 3 + 1] +
          metric[(i_shared) * 3 + 2] * metric[(i_shared) * 3 + 2]));
      real spectralRadNext = abs((metric[(i_shared + 1) * 3] * cv[(i_shared + 1) * (n_var + 2) + 1] +
                                  metric[(i_shared + 1) * 3 + 1] * cv[(i_shared + 1) * (n_var + 2) + 2] +
                                  metric[(i_shared + 1) * 3 + 2] * cv[(i_shared + 1) * (n_var + 2) + 3]) /
                                 cv[(i_shared + 1) * (n_var + 2)] + cv[(i_shared + 1) * (n_var + 2) + n_var + 1] * sqrt(
          metric[(i_shared + 1) * 3] * metric[(i_shared + 1) * 3] +
          metric[(i_shared + 1) * 3 + 1] * metric[(i_shared + 1) * 3 + 1] +
          metric[(i_shared + 1) * 3 + 2] * metric[(i_shared + 1) * 3 + 2]));
      for (int l = 0; l < n_var - 5; ++l) {
        f_1st[tid * (n_var - 5) + l] =
            0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cv[i_shared * (n_var + 2) + l + 5] * jac1) +
            0.5 *
            (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cv[(i_shared + 1) * (n_var + 2) + l + 5] * jac2);
      }
    }

    for (int l = 0; l < 5; ++l) {
      real coeff_alpha_s{0.5};
      if (l == 1) {
        coeff_alpha_s = -kx;
      } else if (l == 2) {
        coeff_alpha_s = -ky;
      } else if (l == 3) {
        coeff_alpha_s = -kz;
      }

      real v[5];
      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        for (int n = 0; n < 5; ++n) {
          v[m] += LR(l, n) * Fp[(i_shared - 2 + m) * n_var + n];
        }
        for (int n = 0; n < n_spec; ++n) {
          v[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 2 + m) * n_var + 5 + n];
        }
      }
      // Reconstruct fPlusChar with WENO-Z-5.
      constexpr real one6th{1.0 / 6};
      real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
      real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
      real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
      constexpr real thirteen12th{13.0 / 12};
      real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                   0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                   0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                   0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
      constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
      real a0{three10th + three10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0))};
      real a1{six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1))};
      real a2{one10th + one10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2))};
      const real fPlusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        for (int n = 0; n < 5; ++n) {
          v[m] += LR(l, n) * Fm[(i_shared - 1 + m) * n_var + n];
        }
        for (int n = 0; n < n_spec; ++n) {
          v[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 1 + m) * n_var + 5 + n];
        }
      }

      // Reconstruct fMinusChar with WENO-Z-5.
      v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
      v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
      v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
      beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
              0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
              0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
              0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      tau5sqr = (beta0 - beta2) * (beta0 - beta2);
      a0 = one10th + one10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0));
      a1 = six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1));
      a2 = three10th + three10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2));
      const real fMinusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      fChar[l] = fPlusChar + fMinusChar;
    }
    for (int l = 0; l < n_spec; ++l) {
      real v[5];
      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        v[m] = -svm[l] * Fp[(i_shared - 2 + m) * n_var] + Fp[(i_shared - 2 + m) * n_var + 5 + l];
      }
      // Reconstruct fPlusChar with WENO-Z-5.
      constexpr real one6th{1.0 / 6};
      real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
      real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
      real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
      constexpr real thirteen12th{13.0 / 12};
      real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                   0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                   0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                   0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
      constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
      real a0{three10th + three10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0))};
      real a1{six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1))};
      real a2{one10th + one10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2))};
      const real fPlusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        v[m] = -svm[l] * Fm[(i_shared - 1 + m) * n_var] + Fm[(i_shared - 1 + m) * n_var + 5 + l];
      }
      // Reconstruct fMinusChar with WENO-Z-5.
      v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
      v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
      v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
      beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
              0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
              0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
              0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      tau5sqr = (beta0 - beta2) * (beta0 - beta2);
      a0 = one10th + one10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0));
      a1 = six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1));
      a2 = three10th + three10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2));
      const real fMinusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      fChar[l + 5] = fPlusChar + fMinusChar;
    }
  } else {
    // My method
    real spec_rad[3], spectralRadThis, spectralRadNext;
    memset(spec_rad, 0, 3 * sizeof(real));
    bool pp_limiter{param->positive_preserving};
    for (int l = -2; l < 4; ++l) {
      const real *Q = &cv[(i_shared + l) * (n_var + 2)];
      real c = Q[n_var + 1];
      real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
                         metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
                         metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
      real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
                 metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
      real ukPc = abs(Uk + c * grad_k);
      real ukMc = abs(Uk - c * grad_k);
      spec_rad[0] = max(spec_rad[0], ukMc);
      spec_rad[1] = max(spec_rad[1], abs(Uk));
      spec_rad[2] = max(spec_rad[2], ukPc);
      if (pp_limiter && l == 0)
        spectralRadThis = ukPc;
      if (pp_limiter && l == 1)
        spectralRadNext = ukPc;
    }
    spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
    spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
    spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));

    if (pp_limiter) {
      for (int l = 0; l < n_var - 5; ++l) {
        f_1st[tid * (n_var - 5) + l] =
            0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cv[i_shared * (n_var + 2) + l + 5] * jac1) +
            0.5 *
            (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cv[(i_shared + 1) * (n_var + 2) + l + 5] * jac2);
      }
    }

    for (int l = 0; l < 5; ++l) {
      real lambda_l{spec_rad[1]};
      if (l == 0) {
        lambda_l = spec_rad[0];
      } else if (l == 4) {
        lambda_l = spec_rad[2];
      }
      real coeff_alpha_s{0.5};
      if (l == 1) {
        coeff_alpha_s = -kx;
      } else if (l == 2) {
        coeff_alpha_s = -ky;
      } else if (l == 3) {
        coeff_alpha_s = -kz;
      }

      real v[5];
      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        for (int n = 0; n < 5; ++n) {
          v[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * n_var + n] + lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + n] *
                                                                   jac[i_shared - 2 + m]);
        }
        for (int n = 0; n < n_spec; ++n) {
          v[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 2 + m) * n_var + 5 + n] +
                                                lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + n + 5] *
                                                jac[i_shared - 2 + m]);
        }
        v[m] *= 0.5;
      }
      // Reconstruct fPlusChar with WENO-Z-5.
      constexpr real one6th{1.0 / 6};
      real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
      real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
      real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
      constexpr real thirteen12th{13.0 / 12};
      real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                   0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                   0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                   0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
      constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
      real a0{three10th + three10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0))};
      real a1{six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1))};
      real a2{one10th + one10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2))};
      const real fPlusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        for (int n = 0; n < 5; ++n) {
          v[m] += LR(l, n) * (Fp[(i_shared - 1 + m) * n_var + n] -
                              lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + n] * jac[i_shared - 1 + m]);
        }
        for (int n = 0; n < n_spec; ++n) {
          v[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 1 + m) * n_var + 5 + n] -
                                                lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + n + 5] *
                                                jac[i_shared - 1 + m]);
        }
        v[m] *= 0.5;
      }

      // Reconstruct fMinusChar with WENO-Z-5.
      v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
      v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
      v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
      beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
              0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
              0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
              0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      tau5sqr = (beta0 - beta2) * (beta0 - beta2);
      a0 = one10th + one10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0));
      a1 = six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1));
      a2 = three10th + three10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2));
      const real fMinusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      fChar[l] = fPlusChar + fMinusChar;
    }
    for (int l = 0; l < n_spec; ++l) {
      const real lambda_l{spec_rad[1]};
      real v[5];
      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        v[m] = 0.5 * ((Fp[(i_shared - 2 + m) * n_var + 5 + l] +
                       lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 2 + m]) -
                      svm[l] * (Fp[(i_shared - 2 + m) * n_var] +
                                lambda_l * cv[(i_shared - 2 + m) * (n_var + 2)] * jac[i_shared - 2 + m]));
      }
      // Reconstruct fPlusChar with WENO-Z-5.
      constexpr real one6th{1.0 / 6};
      real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
      real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
      real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
      constexpr real thirteen12th{13.0 / 12};
      real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                   0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                   0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                   0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
      constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
      real a0{three10th + three10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0))};
      real a1{six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1))};
      real a2{one10th + one10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2))};
      const real fPlusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      memset(v, 0, 5 * sizeof(real));
      for (int m = 0; m < 5; ++m) {
        v[m] = 0.5 * ((Fp[(i_shared - 1 + m) * n_var + 5 + l] -
                       lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 1 + m]) -
                      svm[l] * (Fp[(i_shared - 1 + m) * n_var] -
                                lambda_l * cv[(i_shared - 1 + m) * (n_var + 2)] * jac[i_shared - 1 + m]));
      }
      // Reconstruct fMinusChar with WENO-Z-5.
      v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
      v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
      v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
      beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
              0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
      beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
              0.25 * (v[1] - v[3]) * (v[1] - v[3]);
      beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
              0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
      tau5sqr = (beta0 - beta2) * (beta0 - beta2);
      a0 = one10th + one10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0));
      a1 = six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1));
      a2 = three10th + three10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2));
      const real fMinusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

      fChar[l + 5] = fPlusChar + fMinusChar;
    }
  }

  // Compute the right characteristic matrix
  LR(0, 0) = 1.0;
  LR(0, 1) = kx;
  LR(0, 2) = ky;
  LR(0, 3) = kz;
  LR(0, 4) = 1.0;
  LR(1, 0) = um - kx * cm;
  LR(1, 1) = kx * um;
  LR(1, 2) = ky * um - kz * cm;
  LR(1, 3) = kz * um + ky * cm;
  LR(1, 4) = um + kx * cm;
  LR(2, 0) = vm - ky * cm;
  LR(2, 1) = kx * vm + kz * cm;
  LR(2, 2) = ky * vm;
  LR(2, 3) = kz * vm - kx * cm;
  LR(2, 4) = vm + ky * cm;
  LR(3, 0) = wm - kz * cm;
  LR(3, 1) = kx * wm - ky * cm;
  LR(3, 2) = ky * wm + kx * cm;
  LR(3, 3) = kz * wm;
  LR(3, 4) = wm + kz * cm;
  LR(4, 0) = hm - Uk_bar * cm;
  LR(4, 1) = kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm;
  LR(4, 2) = ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm;
  LR(4, 3) = kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm;
  LR(4, 4) = hm + Uk_bar * cm;

  // Project the flux back to physical space
  auto fci = &fc[tid * n_var];
  fci[0] = LR(0, 0) * fChar[0] + LR(0, 1) * fChar[1] + LR(0, 2) * fChar[2] + LR(0, 3) * fChar[3] + LR(0, 4) * fChar[4];
  fci[1] = LR(1, 0) * fChar[0] + LR(1, 1) * fChar[1] + LR(1, 2) * fChar[2] + LR(1, 3) * fChar[3] + LR(1, 4) * fChar[4];
  fci[2] = LR(2, 0) * fChar[0] + LR(2, 1) * fChar[1] + LR(2, 2) * fChar[2] + LR(2, 3) * fChar[3] + LR(2, 4) * fChar[4];
  fci[3] = LR(3, 0) * fChar[0] + LR(3, 1) * fChar[1] + LR(3, 2) * fChar[2] + LR(3, 3) * fChar[3] + LR(3, 4) * fChar[4];

  fci[4] = LR(4, 0) * fChar[0] + LR(4, 1) * fChar[1] + LR(4, 2) * fChar[2] + LR(4, 3) * fChar[3] + LR(4, 4) * fChar[4];
  real add{0};
  for (int l = 0; l < n_spec; ++l) {
    add += alpha_l[l] * fChar[l + 5];
  }
  fci[4] -= add * cm * cm / gm1;

  const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  for (int l = 0; l < n_spec; ++l) {
    fci[5 + l] = svm[l] * coeff_add + fChar[l + 5];
  }
}

// The above function can actually realize the following ability, but the speed is slower than the specific version.
// Thus, we keep the current version.
template<>
__device__ void
compute_weno_flux_ch<MixtureModel::Air>(const real *cv, DParameter *param, int tid, const real *metric,
                                        const real *jac, real *fc, int i_shared, real *Fp, real *Fm,
                                        const int *ig_shared, int n_add, [[maybe_unused]] real *f_1st) {

  const int n_var = param->n_var;

  // Li Xinliang
  compute_flux<MixtureModel::Air>(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared],
                                  &Fp[i_shared * 5], &Fm[i_shared * 5]);
  for (size_t i = 0; i < n_add; i++) {
    compute_flux<MixtureModel::Air>(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3],
                                    jac[ig_shared[i]], &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
  }

  // My version
//  compute_flux<MixtureModel::Air>(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared],
//                                  &Fk[i_shared * 5]);
//  for (size_t i = 0; i < n_add; i++) {
//    compute_flux<MixtureModel::Air>(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3],
//                                    jac[ig_shared[i]], &Fk[ig_shared[i] * n_var]);
//  }
  // The first n_var in the cv array is conservative vars, followed by p and T.
  const real *cvl{&cv[i_shared * (n_var + 2)]};
  const real *cvr{&cv[(i_shared + 1) * (n_var + 2)]};
  // First, compute the Roe average of the half-point variables.
  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
  const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
  const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
  constexpr real gm1{gamma_air - 1};
  const real hl{(cvl[4] + cvl[n_var]) / cvl[0]};
  const real hr{(cvr[4] + cvr[n_var]) / cvr[0]};
  const real hm{rlc * hl + rrc * hr};
  const real cm2{gm1 * (hm - ekm)};
  const real cm{sqrt(cm2)};

  // Next, we compute the left characteristic matrix at i+1/2.
//  real kx{0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3])};
//  real ky{0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1])};
//  real kz{0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2])};
  real kx{(jac[i_shared] * metric[i_shared * 3] + jac[i_shared + 1] * metric[(i_shared + 1) * 3]) /
          (jac[i_shared] + jac[i_shared + 1])};
  real ky{(jac[i_shared] * metric[i_shared * 3 + 1] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1]) /
          (jac[i_shared] + jac[i_shared + 1])};
  real kz{(jac[i_shared] * metric[i_shared * 3 + 2] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2]) /
          (jac[i_shared] + jac[i_shared + 1])};
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  // The method which contains turbulent variables is not implemented yet.
  gxl::Matrix<real, 5, 5> LR;
  LR(0, 0) = (alpha + Uk_bar * cm) / cm2 * 0.5;
  LR(0, 1) = -(gm1 * um + kx * cm) / cm2 * 0.5;
  LR(0, 2) = -(gm1 * vm + ky * cm) / cm2 * 0.5;
  LR(0, 3) = -(gm1 * wm + kz * cm) / cm2 * 0.5;
  LR(0, 4) = gm1 / cm2 * 0.5;
  LR(1, 0) = kx * (1 - alpha / cm2) - (kz * vm - ky * wm) / cm;
  LR(1, 1) = kx * gm1 * um / cm2;
  LR(1, 2) = (kx * gm1 * vm + kz * cm) / cm2;
  LR(1, 3) = (kx * gm1 * wm - ky * cm) / cm2;
  LR(1, 4) = -kx * gm1 / cm2;
  LR(2, 0) = ky * (1 - alpha / cm2) - (kx * wm - kz * um) / cm;
  LR(2, 1) = (ky * gm1 * um - kz * cm) / cm2;
  LR(2, 2) = ky * gm1 * vm / cm2;
  LR(2, 3) = (ky * gm1 * wm + kx * cm) / cm2;
  LR(2, 4) = -ky * gm1 / cm2;
  LR(3, 0) = kz * (1 - alpha / cm2) - (ky * um - kx * vm) / cm;
  LR(3, 1) = (kz * gm1 * um + ky * cm) / cm2;
  LR(3, 2) = (kz * gm1 * vm - kx * cm) / cm2;
  LR(3, 3) = kz * gm1 * wm / cm2;
  LR(3, 4) = -kz * gm1 / cm2;
  LR(4, 0) = (alpha - Uk_bar * cm) / cm2 * 0.5;
  LR(4, 1) = -(gm1 * um - kx * cm) / cm2 * 0.5;
  LR(4, 2) = -(gm1 * vm - ky * cm) / cm2 * 0.5;
  LR(4, 3) = -(gm1 * wm - kz * cm) / cm2 * 0.5;
  LR(4, 4) = gm1 / cm2 * 0.5;

  // Compute the characteristic flux with L.
  real fChar[5];
  constexpr real eps{1e-40};
  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
//  const real eps_scaled = 1e-6 * param->weno_eps_scale;
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 *
                          ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
                           (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
//  const real eps_scaled = eps * param->weno_eps_scale *(jac[i_shared] + jac[i_shared + 1]) * 0.5; //
//  real spec_rad[3];
//  memset(spec_rad, 0, 3 * sizeof(real));
//  for (int l = -2; l < 4; ++l) {
//    const real *Q = &cv[(i_shared + l) * (n_var + 2)];
//    real c = sqrt(gamma_air * Q[n_var] / Q[0]);
//    real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
//                       metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
//                       metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
//    real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
//               metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
//    real ukPc = abs(Uk + c * grad_k);
//    real ukMc = abs(Uk - c * grad_k);
//    spec_rad[0] = max(spec_rad[0], ukMc);
//    spec_rad[1] = max(spec_rad[1], abs(Uk));
//    spec_rad[2] = max(spec_rad[2], ukPc);
//  }
//  spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
//  spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
//  spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));
  real ap[3], an[3];
  const auto max_spec_rad = abs(Uk_bar) + cm;
  ap[0] = 0.5 * (Uk_bar - cm + max_spec_rad) * gradK;
  ap[1] = 0.5 * (Uk_bar + max_spec_rad) * gradK;
  ap[2] = 0.5 * (Uk_bar + cm + max_spec_rad) * gradK;
  an[0] = 0.5 * (Uk_bar - cm - max_spec_rad) * gradK;
  an[1] = 0.5 * (Uk_bar - max_spec_rad) * gradK;
  an[2] = 0.5 * (Uk_bar + cm - max_spec_rad) * gradK;

  for (int l = 0; l < 5; ++l) {
    real lambda_p{ap[1]}, lambda_n{an[1]};
    if (l == 0) {
      lambda_p = ap[0];
      lambda_n = an[0];
    } else if (l == 4) {
      lambda_p = ap[2];
      lambda_n = an[2];
    }

    real v[5];
    memset(v, 0, 5 * sizeof(real));
    for (int m = 0; m < 5; ++m) {
      // Li Xinliang version
      for (int n = 0; n < 5; ++n) {
        v[m] += LR(l, n) * Fp[(i_shared - 2 + m) * 5 + n];
      }
      // ACANS version
//      for (int n = 0; n < 5; ++n) {
//        v[m] +=
//            lambda_p * LR(l, n) * cv[(i_shared - 2 + m) * (n_var + 2) + n] * 0.5 * (jac[i_shared] + jac[i_shared + 1]);
//      }

//      for (int n = 0; n < 5; ++n) {
//        v[m] += LR(l, n) * (Fk[(i_shared - 2 + m) * 5 + n] + lambda_l * cv[(i_shared - 2 + m) * 7 + n] *
//                                                             jac[i_shared - 2 + m]);
//      }
//      v[m] *= 0.5;
    }

    // Reconstruct fPlusChar with WENO-Z-5.
    constexpr real one6th{1.0 / 6};
    real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
    real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
    real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
    constexpr real thirteen12th{13.0 / 12};
    real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                 0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
    real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                 0.25 * (v[1] - v[3]) * (v[1] - v[3]);
    real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                 0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
    real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
    constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
    real a0{three10th + three10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0))};
    real a1{six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1))};
    real a2{one10th + one10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2))};
    const real fPlusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    memset(v, 0, 5 * sizeof(real));
    for (int m = 0; m < 5; ++m) {
      for (int n = 0; n < 5; ++n) {
        v[m] += LR(l, n) * Fm[(i_shared - 1 + m) * 5 + n];
      }

//      for (int n = 0; n < 5; ++n) {
//        v[m] +=
//            lambda_n * LR(l, n) * cv[(i_shared - 1 + m) * (n_var + 2) + n] * 0.5 * (jac[i_shared] + jac[i_shared + 1]);
//      }

      //      for (int n = 0; n < 5; ++n) {
//        v[m] += LR(l, n) * (Fk[(i_shared - 1 + m) * 5 + n] - lambda_l * cv[(i_shared - 1 + m) * 7 + n] *
//                                                             jac[i_shared - 1 + m]);
//      }
//      v[m] *= 0.5;
    }

    // Reconstruct fMinusChar with WENO-Z-5.
    v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
    v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
    v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
    beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
            0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
    beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
            0.25 * (v[1] - v[3]) * (v[1] - v[3]);
    beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
            0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
    tau5sqr = (beta0 - beta2) * (beta0 - beta2);
    a0 = one10th + one10th * tau5sqr / ((eps_scaled + beta0) * (eps_scaled + beta0));
    a1 = six10th + six10th * tau5sqr / ((eps_scaled + beta1) * (eps_scaled + beta1));
    a2 = three10th + three10th * tau5sqr / ((eps_scaled + beta2) * (eps_scaled + beta2));
    const real fMinusChar{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    fChar[l] = fPlusChar + fMinusChar;
  }

  // Compute the right characteristic matrix
  LR(0, 0) = 1.0;
  LR(0, 1) = kx;
  LR(0, 2) = ky;
  LR(0, 3) = kz;
  LR(0, 4) = 1.0;
  LR(1, 0) = um - kx * cm;
  LR(1, 1) = kx * um;
  LR(1, 2) = ky * um - kz * cm;
  LR(1, 3) = kz * um + ky * cm;
  LR(1, 4) = um + kx * cm;
  LR(2, 0) = vm - ky * cm;
  LR(2, 1) = kx * vm + kz * cm;
  LR(2, 2) = ky * vm;
  LR(2, 3) = kz * vm - kx * cm;
  LR(2, 4) = vm + ky * cm;
  LR(3, 0) = wm - kz * cm;
  LR(3, 1) = kx * wm - ky * cm;
  LR(3, 2) = ky * wm + kx * cm;
  LR(3, 3) = kz * wm;
  LR(3, 4) = wm + kz * cm;
  LR(4, 0) = hm - Uk_bar * cm;
  LR(4, 1) = kx * alpha / gm1 + (kz * vm - ky * wm) * cm;
  LR(4, 2) = ky * alpha / gm1 + (kx * wm - kz * um) * cm;
  LR(4, 3) = kz * alpha / gm1 + (ky * um - kx * vm) * cm;
  LR(4, 4) = hm + Uk_bar * cm;

  // Project the flux back to physical space
  auto fci = &fc[tid * n_var];
  for (int l = 0; l < 5; ++l) {
    real f{0};
    for (int m = 0; m < 5; ++m) {
      f += LR(l, m) * fChar[m];
    }
    fci[l] = f;
  }
}

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_cp(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st) {
  const int n_var = param->n_var;

  compute_flux<mix_model>(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared],
                          &Fp[i_shared * n_var], &Fm[i_shared * n_var]);
  for (size_t i = 0; i < n_add; i++) {
    compute_flux<mix_model>(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3],
                            jac[ig_shared[i]], &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
  }
  __syncthreads();

  constexpr real eps{1e-40};
  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
  const real eps_ref = eps * param->weno_eps_scale * 0.25 *
                       ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
                        (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
  real eps_scaled[3];
  eps_scaled[0] = eps_ref;
  eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
  eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

  if (param->positive_preserving) {
    for (int l = 0; l < n_var - 5; ++l) {
      f_1st[tid * (n_var - 5) + l] = 0.5 * Fp[i_shared * n_var + l + 5] + 0.5 * Fm[(i_shared + 1) * n_var + l + 5];
    }
  }

  auto fci = &fc[tid * n_var];

  for (int l = 0; l < n_var; ++l) {
    real eps_here{eps_scaled[0]};
    if (l == 1 || l == 2 || l == 3) {
      eps_here = eps_scaled[1];
    } else if (l == 4) {
      eps_here = eps_scaled[2];
    }

    real v[5];
    v[0] = Fp[(i_shared - 2) * n_var + l];
    v[1] = Fp[(i_shared - 1) * n_var + l];
    v[2] = Fp[i_shared * n_var + l];
    v[3] = Fp[(i_shared + 1) * n_var + l];
    v[4] = Fp[(i_shared + 2) * n_var + l];

    constexpr real one6th{1.0 / 6};
    real v0{one6th * (2 * v[2] + 5 * v[3] - v[4])};
    real v1{one6th * (-v[1] + 5 * v[2] + 2 * v[3])};
    real v2{one6th * (2 * v[0] - 7 * v[1] + 11 * v[2])};
    constexpr real thirteen12th{13.0 / 12};
    real beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
                 0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
    real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
                 0.25 * (v[1] - v[3]) * (v[1] - v[3]);
    real beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
                 0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
    constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
    real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
    real a0{three10th + three10th * tau5sqr / ((eps_here + beta0) * (eps_here + beta0))};
    real a1{six10th + six10th * tau5sqr / ((eps_here + beta1) * (eps_here + beta1))};
    real a2{one10th + one10th * tau5sqr / ((eps_here + beta2) * (eps_here + beta2))};
    const real fPlusCp{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    v[0] = Fm[(i_shared - 1) * n_var + l];
    v[1] = Fm[i_shared * n_var + l];
    v[2] = Fm[(i_shared + 1) * n_var + l];
    v[3] = Fm[(i_shared + 2) * n_var + l];
    v[4] = Fm[(i_shared + 3) * n_var + l];

    v0 = one6th * (11 * v[2] - 7 * v[3] + 2 * v[4]);
    v1 = one6th * (2 * v[1] + 5 * v[2] - v[3]);
    v2 = one6th * (-v[0] + 5 * v[1] + 2 * v[2]);
    beta0 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
            0.25 * (3 * v[2] - 4 * v[3] + v[4]) * (3 * v[2] - 4 * v[3] + v[4]);
    beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
            0.25 * (v[1] - v[3]) * (v[1] - v[3]);
    beta2 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
            0.25 * (v[0] - 4 * v[1] + 3 * v[2]) * (v[0] - 4 * v[1] + 3 * v[2]);
    tau5sqr = (beta0 - beta2) * (beta0 - beta2);
    a0 = one10th + one10th * tau5sqr / ((eps_here + beta0) * (eps_here + beta0));
    a1 = six10th + six10th * tau5sqr / ((eps_here + beta1) * (eps_here + beta1));
    a2 = three10th + three10th * tau5sqr / ((eps_here + beta2) * (eps_here + beta2));
    const real fMinusCp{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    fci[l] = fPlusCp + fMinusCp;
  }
}

template<MixtureModel mix_model>
__device__ void
positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
                            real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac) {
  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;
  for (int l = 0; l < n_var - 5; ++l) {
    real theta_p = 1.0, theta_m = 1.0;
    if (idx_in_mesh > -1) {
      const real up = 0.5 * alpha * cv[i_shared * (n_var + 2) + 5 + l] * jac[i_shared] - dt * fc[tid * n_var + 5 + l];
      if (up < 0) {
        const real up_lf =
            0.5 * alpha * cv[i_shared * (n_var + 2) + 5 + l] * jac[i_shared] - dt * f_1st[tid * (n_var - 5) + l];
        if (abs(up - up_lf) > 1e-20) {
          theta_p = (0 - up_lf) / (up - up_lf);
          if (theta_p > 1)
            theta_p = 1.0;
          else if (theta_p < 0)
            theta_p = 0;
        }
      }
    }

    if (idx_in_mesh < max_extent - 1) {
      const real um =
          0.5 * alpha * cv[(i_shared + 1) * (n_var + 2) + 5 + l] * jac[i_shared + 1] + dt * fc[tid * n_var + 5 + l];
      if (um < 0) {
        const real um_lf = 0.5 * alpha * cv[(i_shared + 1) * (n_var + 2) + 5 + l] * jac[i_shared + 1] +
                           dt * f_1st[tid * (n_var - 5) + l];
        if (abs(um - um_lf) > 1e-20) {
          theta_m = (0 - um_lf) / (um - um_lf);
          if (theta_m > 1)
            theta_m = 1.0;
          else if (theta_m < 0)
            theta_m = 0;
        }
      }
    }

    fc[tid * n_var + 5 + l] =
        min(theta_p, theta_m) * (fc[tid * n_var + 5 + l] - f_1st[tid * (n_var - 5) + l]) + f_1st[tid * (n_var - 5) + l];
  }
}

template void
compute_convective_term_weno<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                                const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                    int n_var, const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                            int n_var, const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                               const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                               const Parameter &parameter);
}