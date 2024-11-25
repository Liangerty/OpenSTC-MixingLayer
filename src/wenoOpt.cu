//#include "InviscidScheme.cuh"
//#include "Constants.h"
//#include "Parallel.h"
//#include "DParameter.cuh"
//#include "Thermo.cuh"
//#include "Field.h"
//
//namespace cfd {
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_1D_padding(cfd::DZone *zone, int direction, int max_extent, DParameter *param);
//
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_1D_split_component(cfd::DZone *zone, int direction, int max_extent, DParameter *param);
//
//template<MixtureModel mix_model>
//__global__ void
//compute_half_point_flux_weno_x_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param);
//
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_y_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param);
//
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_z_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param);
//
//template<MixtureModel mix_model>
//__global__ void compute_fp_fm(cfd::DZone *zone, int max_extent, DParameter *param) {
//  const auto ngg{zone->ngg};
//  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
//  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
//  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (i >= max_extent + ngg) return;
//
//  const auto &cv = zone->cv;
//  const int n_var = param->n_var;
//  auto &Fp = zone->Fp;
//  auto &Fm = zone->Fm;
//  real metric[3]{zone->metric(i, j, k)(1, 1), zone->metric(i, j, k)(1, 2), zone->metric(i, j, k)(1, 3)};
//
//  const real rho{cv(i, j, k, 0)};
//  const real rhoU{cv(i, j, k, 1)}, rhoV{cv(i, j, k, 2)}, rhoW{cv(i, j, k, 3)};
//  const real Uk{(rhoU * metric[0] + rhoV * metric[1] + rhoW * metric[2]) / rho};
//  const real pk{zone->bv(i, j, k, 4)};
//  real cc{0};
//  if constexpr (mix_model == MixtureModel::Air) {
//    cc = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//  } else {
//    cc = zone->acoustic_speed(i, j, k);
//  }
//  const real cGradK = cc * norm3d(metric[0], metric[1], metric[2]);
//  const real lambda0 = abs(Uk) + cGradK;
//  const real jac = zone->jac(i, j, k);
//  const real rhoE{cv(i, j, k, 4)};
//
//  Fp(i, j, k, 0) = 0.5 * jac * (Uk * rho + lambda0 * rho);
//  Fp(i, j, k, 1) = 0.5 * jac * (Uk * rhoU + pk * metric[0] + lambda0 * rhoU);
//  Fp(i, j, k, 2) = 0.5 * jac * (Uk * rhoV + pk * metric[1] + lambda0 * rhoV);
//  Fp(i, j, k, 3) = 0.5 * jac * (Uk * rhoW + pk * metric[2] + lambda0 * rhoW);
//  Fp(i, j, k, 4) = 0.5 * jac * (Uk * (rhoE + pk) + lambda0 * rhoE);
//
//  Fm(i, j, k, 0) = 0.5 * jac * (Uk * rho - lambda0 * rho);
//  Fm(i, j, k, 1) = 0.5 * jac * (Uk * rhoU + pk * metric[0] - lambda0 * rhoU);
//  Fm(i, j, k, 2) = 0.5 * jac * (Uk * rhoV + pk * metric[1] - lambda0 * rhoV);
//  Fm(i, j, k, 3) = 0.5 * jac * (Uk * rhoW + pk * metric[2] - lambda0 * rhoW);
//  Fm(i, j, k, 4) = 0.5 * jac * (Uk * (rhoE + pk) - lambda0 * rhoE);
//
//  for (int l = 5; l < n_var; ++l) {
//    const real rhoYl{cv(i, j, k, l)};
//    Fp(i, j, k, l) = 0.5 * jac * (Uk * rhoYl + lambda0 * rhoYl);
//    Fm(i, j, k, l) = 0.5 * jac * (Uk * rhoYl - lambda0 * rhoYl);
//  }
////  for (int l = 0; l < n_var; ++l) {
////    if (isnan(abs(Fp(i, j, k, l)))) {
////      printf("fp(%d, %d, %d, %d) is nan, fp[0:4]={%f, %f, %f, %f, %f}, Uk=%f, lambda=%f, c=%f, gradK=%f, cGradK=%f\n",
////             i, j, k, l, Fp(i, j, k, 0), Fp(i, j, k, 1),
////             Fp(i, j, k, 2), Fp(i, j, k, 3), Fp(i, j, k, 4), Uk, lambda0, zone->acoustic_speed(i, j, k),
////             norm3d(metric[0], metric[1], metric[2]),
////             cGradK);
////      break;
////    }
////  }
//  __syncthreads();
//}
//
//template<MixtureModel mix_model>
//__global__ void compute_dfc_dx(cfd::DZone *zone, int max_extent, DParameter *param) {
//  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
//  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
//  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (i >= max_extent) return;
//
//  int nv = param->n_var;
//  auto &dq = zone->dq;
//  auto &fc = zone->flux;
//  for (int l = 0; l < nv; ++l) {
//    dq(i, j, k, l) -= fc(i, j, k, l) - fc(i - 1, j, k, l);
//  }
//}
//
//template<MixtureModel mix_model>
//void compute_convective_term_weno_opt(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
//                                      const Parameter &parameter) {
//  // 0 - padding version, 1 - split component version, 2 - no shared memory version
//  constexpr int version = 2;
//  // The implementation of classic WENO.
//  const int extent[3]{block.mx, block.my, block.mz};
//
//  // If we change the 64 to 128 here, then the sharedMem for dim = 2 will be different from that for dim = 0 and 1.
//  constexpr int block_dim = 64;
//  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
//
//  if constexpr (version == 0) {
//    // Padding version. worse performance.
//    const int pad_reconstruct = ((n_var + 2 + 15) / 16) * 16 + 1;
//    const int pad_var = ((n_var + 15) / 16) * 16 + 1;
//
//    auto mem_cv = (n_computation_per_block * pad_reconstruct + 1) / 2 * 2;
//    auto mem_flux = (n_computation_per_block * pad_var + 1) / 2 * 2;
//    auto mem_fc = (block_dim * pad_var + 1) / 2 * 2;
//
//    auto sharedMem = (
//                         mem_fc                         // Fc
//                         + mem_flux * 2                 // F+/F-
//                         + mem_cv                       // cv[n_var]+p+T
//                         + n_computation_per_block * 3  // metric[3]
//                         + n_computation_per_block      // jacobian
//                     ) * sizeof(real);
//    if (parameter.get_bool("positive_preserving")) {
//      sharedMem += block_dim * (n_var - 5) * sizeof(real); // f_1th
//    }
//
//    for (auto dir = 0; dir < 2; ++dir) {
//      int tpb[3]{1, 1, 1};
//      tpb[dir] = block_dim;
//      int bpg[3]{extent[0], extent[1], extent[2]};
//      bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;
//
//      dim3 TPB(tpb[0], tpb[1], tpb[2]);
//      dim3 BPG(bpg[0], bpg[1], bpg[2]);
//      compute_convective_term_weno_1D_padding<mix_model><<<BPG, TPB, sharedMem>>>(zone, dir, extent[dir], param);
//    }
//
//    if (extent[2] > 1) {
//      // 3D computation
//      // Number of threads in the 3rd direction cannot exceed 64
//      constexpr int tpb[3]{1, 1, 64};
//      int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};
//
//      dim3 TPB(tpb[0], tpb[1], tpb[2]);
//      dim3 BPG(bpg[0], bpg[1], bpg[2]);
//      compute_convective_term_weno_1D_padding<mix_model><<<BPG, TPB, sharedMem>>>(zone, 2, extent[2], param);
//    }
//  } else if constexpr (version == 1) {
//    // Split component version. Worse performance.
//    auto shared_mem = (block_dim * n_var // fc
//                       + n_computation_per_block * n_var * 2 // F+/F-
//                       + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
//                      + n_computation_per_block * 3 * sizeof(real); // metric[3]
//    if (parameter.get_bool("positive_preserving")) {
//      shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
//    }
//
//    for (auto dir = 0; dir < 2; ++dir) {
//      int tpb[3]{1, 1, 1};
//      tpb[dir] = block_dim;
//      int bpg[3]{extent[0], extent[1], extent[2]};
//      bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;
//
//      dim3 TPB(tpb[0], tpb[1], tpb[2]);
//      dim3 BPG(bpg[0], bpg[1], bpg[2]);
//      compute_convective_term_weno_1D_split_component<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir],
//                                                                                           param);
//    }
//
//    if (extent[2] > 1) {
//      // 3D computation
//      // Number of threads in the 3rd direction cannot exceed 64
//      constexpr int tpb[3]{1, 1, 64};
//      int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};
//
//      dim3 TPB(tpb[0], tpb[1], tpb[2]);
//      dim3 BPG(bpg[0], bpg[1], bpg[2]);
//      compute_convective_term_weno_1D_split_component<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
//    }
//  } else {
//    // no shared memory version.
//    dim3 TPB(block_dim, 1, 1);
//    dim3 BPG((extent[0] + 2 * block.ngg - 1) / block_dim + 1, extent[1], extent[2]);
//    compute_fp_fm<mix_model><<<BPG, TPB>>>(zone, extent[0], param);
//    // When computing half-point flux, we need to compute one more point.
//    BPG = dim3((extent[0] + 1 - 1) / block_dim + 1, extent[1], extent[2]);
//    compute_half_point_flux_weno_x_no_shared_mem<mix_model><<<BPG, TPB>>>(zone, extent[0], param);
//    // Finally, compute the df/dx term.
//    BPG = dim3((extent[0] - 1) / block_dim + 1, extent[1], extent[2]);
//    compute_dfc_dx<mix_model><<<BPG, TPB>>>(zone, extent[0], param);
//
//
//    auto shared_mem = (block_dim * n_var // fc
//                       + n_computation_per_block * n_var * 2 // F+/F-
//                       + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
//                      + n_computation_per_block * 3 * sizeof(real); // metric[3]
//    if (parameter.get_bool("positive_preserving")) {
//      shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
//    }
//
////    BPG = dim3((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
////    compute_convective_term_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, extent[0], param);
//
//    TPB = dim3(1, block_dim, 1);
//    BPG = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
//    compute_convective_term_weno_y<mix_model><<<BPG, TPB, shared_mem>>>(zone, extent[1], param);
//
//    if (extent[2] > 1) {
//      TPB = dim3(1, 1, 64);
//      BPG = dim3(extent[0], extent[1], (extent[2] - 1) / (64 - 1) + 1);
//      compute_convective_term_weno_z<mix_model><<<BPG, TPB, shared_mem>>>(zone, extent[2], param);
//    }
//  }
//}
//
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_1D_padding(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
//  int labels[3]{0, 0, 0};
//  labels[direction] = 1;
//  const auto tid = (int) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
//  const auto block_dim = (int) (blockDim.x * blockDim.y * blockDim.z);
//  const auto ngg{zone->ngg};
//  const int n_point = block_dim + 2 * ngg - 1;
//
//  int idx[3];
//  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
//  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
//  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
//  idx[direction] -= 1;
//  if (idx[direction] >= max_extent) return;
//
//  // load variables to shared memory
//  extern __shared__ real s[];
//  const auto n_var{param->n_var};
//  auto n_reconstruct{n_var + 2};
//
//  // Pad to 128 bytes (16 doubles) will cause all threads to access the same bank at the same time, after that we plus 1 to avoid bank conflicts
//  const int pad_reconstruct = ((n_reconstruct + 15) / 16) * 16 + 1;
//  const int pad_var = ((n_var + 15) / 16) * 16 + 1;
//
//  real *cv = s; // Each thread accesses cv[i_shared * pad_reconstruct + l]
//  int offset = ((n_point * pad_reconstruct + 1) / 2) * 2;
//  real *Fp = &cv[offset]; // Each thread accesses Fp[i_shared * pad_var + l]
//  offset = ((n_point * pad_var + 1) / 2) * 2;
//  real *Fm = &Fp[offset]; // Each thread accesses Fm[i_shared * pad_var + l]
//  real *metric = &Fm[offset]; // Each thread accesses metric[i_shared * 3 + l]
//  real *jac = &metric[n_point * 3];
//  real *fc = &jac[n_point];
//  real *f_1st = nullptr;
//  if (param->positive_preserving) {
//    offset = ((block_dim * pad_var + 1) / 2) * 2;
//    f_1st = &fc[offset];
//  }
//
//  const int i_shared = tid - 1 + ngg;
//  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//    cv[i_shared * pad_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
////    cv[i_shared * n_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
//  }
//  cv[i_shared * pad_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
////  cv[i_shared * n_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
//  if constexpr (mix_model != MixtureModel::Air)
//    cv[i_shared * pad_reconstruct + n_var + 1] = zone->acoustic_speed(idx[0], idx[1], idx[2]);
////    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(idx[0], idx[1], idx[2]);
//  else
//    cv[i_shared * pad_reconstruct + n_var + 1] = sqrt(gamma_air * R_u / mw_air * zone->bv(idx[0], idx[1], idx[2], 5));
////    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_u / mw_air * zone->bv(idx[0], idx[1], idx[2], 5));
//  for (auto l = 1; l < 4; ++l) {
//    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
//  }
//  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
//
//  // ghost cells
//  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
//  int ig_shared[max_additional_ghost_point_loaded];
//  int additional_loaded{0};
//  if (tid < ngg - 1) {
//    ig_shared[additional_loaded] = tid;
//    const int g_idx[3]{idx[0] - (ngg - 1) * labels[0], idx[1] - (ngg - 1) * labels[1],
//                       idx[2] - (ngg - 1) * labels[2]};
//
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
////      cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//      cv[ig_shared[additional_loaded] * pad_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//    }
////    cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    cv[ig_shared[additional_loaded] * pad_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
//                                                                                            g_idx[2]);
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
////                                                                                          g_idx[2]);
//    else
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = sqrt(
////          gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = sqrt(
//          gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//    for (auto l = 1; l < 4; ++l) {
//      metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
//    }
//    jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//    ++additional_loaded;
//  }
//  if (tid > block_dim - ngg - 1 || idx[direction] > max_extent - ngg - 1) {
//    ig_shared[additional_loaded] = tid + 2 * ngg - 1;
//    const int g_idx[3]{idx[0] + ngg * labels[0], idx[1] + ngg * labels[1], idx[2] + ngg * labels[2]};
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv[ig_shared[additional_loaded] * pad_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
////      cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//    }
////    cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    cv[ig_shared[additional_loaded] * pad_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    if constexpr (mix_model != MixtureModel::Air)
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
////                                                                                          g_idx[2]);
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
//                                                                                            g_idx[2]);
//    else
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = sqrt(
//          gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = sqrt(
////          gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//    for (auto l = 1; l < 4; ++l) {
//      metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
//    }
//    jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//    ++additional_loaded;
//  }
//  if (idx[direction] == max_extent - 1 && tid < ngg - 1) {
//    int n_more_left = ngg - 1 - tid - 1;
//    for (int m = 0; m < n_more_left; ++m) {
//      ig_shared[additional_loaded] = tid + m + 1;
//      const int g_idx[3]{idx[0] - (ngg - 1 - m - 1) * labels[0], idx[1] - (ngg - 1 - m - 1) * labels[1],
//                         idx[2] - (ngg - 1 - m - 1) * labels[2]};
//
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
////        cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//        cv[ig_shared[additional_loaded] * pad_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//      }
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//      if constexpr (mix_model != MixtureModel::Air)
////        cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
////                                                                                            g_idx[2]);
//        cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
//                                                                                              g_idx[2]);
//      else
////        cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = sqrt(
////            gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//        cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = sqrt(
//            gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//      for (auto l = 1; l < 4; ++l) {
//        metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
//      }
//      jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//      ++additional_loaded;
//    }
//    int n_more_right = ngg - 1 - tid;
//    for (int m = 0; m < n_more_right; ++m) {
//      ig_shared[additional_loaded] = i_shared + m + 1;
//      const int g_idx[3]{idx[0] + (m + 1) * labels[0], idx[1] + (m + 1) * labels[1], idx[2] + (m + 1) * labels[2]};
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv[ig_shared[additional_loaded] * pad_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
////        cv[ig_shared[additional_loaded] * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
//      }
//      cv[ig_shared[additional_loaded] * pad_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
////      cv[ig_shared[additional_loaded] * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
//                                                                                              g_idx[2]);
////        cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = zone->acoustic_speed(g_idx[0], g_idx[1],
////                                                                                            g_idx[2]);
//      else
//        cv[ig_shared[additional_loaded] * pad_reconstruct + n_var + 1] = sqrt(
//            gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
////        cv[ig_shared[additional_loaded] * n_reconstruct + n_var + 1] = sqrt(
////            gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//      for (auto l = 1; l < 4; ++l) {
//        metric[ig_shared[additional_loaded] * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
//      }
//      jac[ig_shared[additional_loaded]] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//      ++additional_loaded;
//    }
//  }
//  __syncthreads();
//
//  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//  if (auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//    compute_weno_flux_cp<mix_model>(cv, param, tid, metric, jac, fc, i_shared, Fp, Fm, ig_shared, additional_loaded,
//                                    f_1st);
//  } else if (sch == 52 || sch == 72) {
//    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, Fp, Fm, ig_shared, additional_loaded,
//                                    f_1st);
//  }
//  __syncthreads();
//
//  if (param->positive_preserving) {
//    real dt{0};
//    if (param->dt > 0)
//      dt = param->dt;
//    else
//      dt = zone->dt_local(idx[0], idx[1], idx[2]);
//    positive_preserving_limiter<mix_model>(f_1st, n_var, tid, fc, param, i_shared, dt, idx[direction], max_extent, cv,
//                                           jac);
//  }
//  __syncthreads();
//
//  if (tid > 0) {
//    for (int l = 0; l < n_var; ++l) {
//      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * pad_var + l] - fc[(tid - 1) * pad_var + l];
////      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//    }
//  }
//}
//
//template<MixtureModel mix_model>
//__device__ void
//compute_flux_split_component(const real *Q, const real *YQ, real p, real cc, DParameter *param,
//                             const real *metric, real jac, real *Fp, real *Fm) {
//  const int n_var = param->n_var;
//  const real Uk{(Q[1] * metric[0] + Q[2] * metric[1] + Q[3] * metric[2]) / Q[0]};
//  const real cGradK = cc * sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
//  const real lambda0 = abs(Uk) + cGradK;
//
//  Fp[0] = 0.5 * jac * (Uk * Q[0] + lambda0 * Q[0]);
//  Fp[1] = 0.5 * jac * (Uk * Q[1] + p * metric[0] + lambda0 * Q[1]);
//  Fp[2] = 0.5 * jac * (Uk * Q[2] + p * metric[1] + lambda0 * Q[2]);
//  Fp[3] = 0.5 * jac * (Uk * Q[3] + p * metric[2] + lambda0 * Q[3]);
//  Fp[4] = 0.5 * jac * (Uk * (Q[4] + p) + lambda0 * Q[4]);
//
//  Fm[0] = 0.5 * jac * (Uk * Q[0] - lambda0 * Q[0]);
//  Fm[1] = 0.5 * jac * (Uk * Q[1] + p * metric[0] - lambda0 * Q[1]);
//  Fm[2] = 0.5 * jac * (Uk * Q[2] + p * metric[1] - lambda0 * Q[2]);
//  Fm[3] = 0.5 * jac * (Uk * Q[3] + p * metric[2] - lambda0 * Q[3]);
//  Fm[4] = 0.5 * jac * (Uk * (Q[4] + p) - lambda0 * Q[4]);
//
//  for (int l = 0; l < n_var - 5; ++l) {
//    Fp[l + 5] = 0.5 * jac * (Uk * YQ[l] + lambda0 * YQ[l]);
//    Fm[l + 5] = 0.5 * jac * (Uk * YQ[l] - lambda0 * YQ[l]);
//  }
//}
//
//template<MixtureModel mix_model>
//__device__ void
//compute_weno_flux_ch_split_component(const real *cv_b, const real *cv_s, const real *p, const real *cc,
//                                     DParameter *param, int tid, const real *metric, const real *jac, real *fc,
//                                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st) {
//  const int n_var = param->n_var;
//  const int ns = n_var - 5;
//
//  // li xinliang(own flux splitting) method
//  compute_flux_split_component<mix_model>(&cv_b[i_shared * 5], &cv_s[i_shared * ns], p[i_shared], cc[i_shared], param,
//                                          &metric[i_shared * 3], jac[i_shared],
//                                          &Fp[i_shared * n_var], &Fm[i_shared * n_var]);
//  for (size_t i = 0; i < n_add; i++) {
//    compute_flux_split_component<mix_model>(&cv_b[ig_shared[i] * 5], &cv_s[ig_shared[i] * ns], p[ig_shared[i]],
//                                            cc[ig_shared[i]], param, &metric[ig_shared[i] * 3],
//                                            jac[ig_shared[i]], &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
//  }
//
//  // The first n_var in the cv array is conservative vars, followed by p and cm.
//  const real *cvl{&cv_b[i_shared * 5]};
//  const real *cvr{&cv_b[(i_shared + 1) * 5]};
//  const real *yql{&cv_s[i_shared * ns]};
//  const real *yqr{&cv_s[(i_shared + 1) * ns]};
//  // First, compute the Roe average of the half-point variables.
//  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//  const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
//  const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
//  const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
//  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
//  real pl{p[i_shared]}, pr{p[i_shared + 1]};
//  const real hl{(cvl[4] + pl) / cvl[0]};
//  const real hr{(cvr[4] + pr) / cvr[0]};
//  const real hm{rlc * hl + rrc * hr};
//
//  real svm[MAX_SPEC_NUMBER];
//  memset(svm, 0, MAX_SPEC_NUMBER * sizeof(real));
//  for (int l = 0; l < ns; ++l) {
//    svm[l] = (rlc * yql[l] / cvl[0] + rrc * yqr[l] / cvr[0]);
//  }
//
//  const int n_spec{param->n_spec};
//  real mw_inv = 0;
//  for (int l = 0; l < n_spec; ++l) {
//    mw_inv += svm[l] / param->mw[l];
//  }
//
//  const real tl{pl / cvl[0]};
//  const real tr{pr / cvr[0]};
//  const real tm = (rlc * tl + rrc * tr) / (R_u * mw_inv);
//
//  real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
//  compute_enthalpy_and_cp(tm, h_i, cp_i, param);
//  real cp{0}, cv_tot{0};
//  for (int l = 0; l < n_spec; ++l) {
//    cp += svm[l] * cp_i[l];
//    cv_tot += svm[l] * (cp_i[l] - R_u / param->mw[l]);
//  }
//  const real gamma = cp / cv_tot;
//  const real cm = sqrt(gamma * R_u * mw_inv * tm);
//  const real gm1{gamma - 1};
//
//  // Next, we compute the left characteristic matrix at i+1/2.
//  real kx{(jac[i_shared] * metric[i_shared * 3] + jac[i_shared + 1] * metric[(i_shared + 1) * 3]) /
//          (jac[i_shared] + jac[i_shared + 1])};
//  real ky{(jac[i_shared] * metric[i_shared * 3 + 1] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1]) /
//          (jac[i_shared] + jac[i_shared + 1])};
//  real kz{(jac[i_shared] * metric[i_shared * 3 + 2] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2]) /
//          (jac[i_shared] + jac[i_shared + 1])};
//  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
//  kx /= gradK;
//  ky /= gradK;
//  kz /= gradK;
//  const real Uk_bar{kx * um + ky * vm + kz * wm};
//  const real alpha{gm1 * ekm};
//
//  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
//  const real cm2_inv{1.0 / (cm * cm)};
//  gxl::Matrix<real, 5, 5> LR;
//  LR(0, 0) = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//  LR(0, 1) = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//  LR(0, 2) = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//  LR(0, 3) = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//  LR(0, 4) = gm1 * cm2_inv * 0.5;
//  LR(1, 0) = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//  LR(1, 1) = kx * gm1 * um * cm2_inv;
//  LR(1, 2) = (kx * gm1 * vm + kz * cm) * cm2_inv;
//  LR(1, 3) = (kx * gm1 * wm - ky * cm) * cm2_inv;
//  LR(1, 4) = -kx * gm1 * cm2_inv;
//  LR(2, 0) = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//  LR(2, 1) = (ky * gm1 * um - kz * cm) * cm2_inv;
//  LR(2, 2) = ky * gm1 * vm * cm2_inv;
//  LR(2, 3) = (ky * gm1 * wm + kx * cm) * cm2_inv;
//  LR(2, 4) = -ky * gm1 * cm2_inv;
//  LR(3, 0) = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//  LR(3, 1) = (kz * gm1 * um + ky * cm) * cm2_inv;
//  LR(3, 2) = (kz * gm1 * vm - kx * cm) * cm2_inv;
//  LR(3, 3) = kz * gm1 * wm * cm2_inv;
//  LR(3, 4) = -kz * gm1 * cm2_inv;
//  LR(4, 0) = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//  LR(4, 1) = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//  LR(4, 2) = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//  LR(4, 3) = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//  LR(4, 4) = gm1 * cm2_inv * 0.5;
//
//  // Compute the characteristic flux with L.
//  real fChar[5 + MAX_SPEC_NUMBER];
//  constexpr real eps{1e-40};
//  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
//  const real eps_scaled = eps * param->weno_eps_scale * 0.25 *
//                          ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
//                           (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
//                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
//                           (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
//                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
//                           (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
//
//  real alpha_l[MAX_SPEC_NUMBER];
//  // compute the partial derivative of pressure to species density
//  for (int l = 0; l < n_spec; ++l) {
//    alpha_l[l] = gamma * R_u / param->mw[l] * tm - (gamma - 1) * h_i[l];
//    // The computations including this alpha_l are all combined with a division by cm2.
//    alpha_l[l] *= cm2_inv;
//  }
//
//  // Li Xinliang's flux splitting
//  bool pp_limiter{param->positive_preserving};
//  if (pp_limiter) {
//    real spectralRadThis = abs(
//        (metric[i_shared * 3] * cvl[1] + metric[i_shared * 3 + 1] * cvl[2] + metric[i_shared * 3 + 2] * cvl[3]) /
//        cvl[0] + cc[i_shared] * sqrt(metric[(i_shared) * 3] * metric[(i_shared) * 3] +
//                                     metric[(i_shared) * 3 + 1] * metric[(i_shared) * 3 + 1] +
//                                     metric[(i_shared) * 3 + 2] * metric[(i_shared) * 3 + 2]));
//    real spectralRadNext = abs((metric[(i_shared + 1) * 3] * cvr[1] + metric[(i_shared + 1) * 3 + 1] * cvr[2] +
//                                metric[(i_shared + 1) * 3 + 2] * cvr[3]) / cvr[0] + cc[i_shared + 1] * sqrt(
//        metric[(i_shared + 1) * 3] * metric[(i_shared + 1) * 3] +
//        metric[(i_shared + 1) * 3 + 1] * metric[(i_shared + 1) * 3 + 1] +
//        metric[(i_shared + 1) * 3 + 2] * metric[(i_shared + 1) * 3 + 2]));
//    for (int l = 0; l < n_var - 5; ++l) {
//      f_1st[tid * (n_var - 5) + l] = 0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * yql[l] * jac1) +
//                                     0.5 * (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * yqr[l] * jac2);
//    }
//  }
//
//  if (param->inviscid_scheme == 52) {
//    for (int l = 0; l < 5; ++l) {
//      real coeff_alpha_s{0.5};
//      if (l == 1) {
//        coeff_alpha_s = -kx;
//      } else if (l == 2) {
//        coeff_alpha_s = -ky;
//      } else if (l == 3) {
//        coeff_alpha_s = -kz;
//      }
//      real vPlus[5], vMinus[5];
//      memset(vPlus, 0, 5 * sizeof(real));
//      memset(vMinus, 0, 5 * sizeof(real));
//      for (int m = 0; m < 5; ++m) {
//        for (int n = 0; n < 5; ++n) {
//          vPlus[m] += LR(l, n) * Fp[(i_shared - 2 + m) * n_var + n];
//          vMinus[m] += LR(l, n) * Fm[(i_shared - 1 + m) * n_var + n];
//        }
//        for (int n = 0; n < n_spec; ++n) {
//          vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 2 + m) * n_var + 5 + n];
//          vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 1 + m) * n_var + 5 + n];
//        }
//      }
//      fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
//    }
//    for (int l = 0; l < n_spec; ++l) {
//      real vPlus[5], vMinus[5];
//      for (int m = 0; m < 5; ++m) {
//        vPlus[m] = -svm[l] * Fp[(i_shared - 2 + m) * n_var] + Fp[(i_shared - 2 + m) * n_var + 5 + l];
//        vMinus[m] = -svm[l] * Fm[(i_shared - 1 + m) * n_var] + Fm[(i_shared - 1 + m) * n_var + 5 + l];
//      }
//      fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled);
//    }
//  } else if (param->inviscid_scheme == 72) {
//    for (int l = 0; l < 5; ++l) {
//      real coeff_alpha_s{0.5};
//      if (l == 1) {
//        coeff_alpha_s = -kx;
//      } else if (l == 2) {
//        coeff_alpha_s = -ky;
//      } else if (l == 3) {
//        coeff_alpha_s = -kz;
//      }
//      real vPlus[7], vMinus[7];
//      memset(vPlus, 0, 7 * sizeof(real));
//      memset(vMinus, 0, 7 * sizeof(real));
//      for (int m = 0; m < 7; ++m) {
//        for (int n = 0; n < 5; ++n) {
//          vPlus[m] += LR(l, n) * Fp[(i_shared - 3 + m) * n_var + n];
//          vMinus[m] += LR(l, n) * Fm[(i_shared - 2 + m) * n_var + n];
//        }
//        for (int n = 0; n < n_spec; ++n) {
//          vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 3 + m) * n_var + 5 + n];
//          vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 2 + m) * n_var + 5 + n];
//        }
//      }
//      fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
//    }
//    for (int l = 0; l < n_spec; ++l) {
//      real vPlus[7], vMinus[7];
//      for (int m = 0; m < 7; ++m) {
//        vPlus[m] = -svm[l] * Fp[(i_shared - 3 + m) * n_var] + Fp[(i_shared - 3 + m) * n_var + 5 + l];
//        vMinus[m] = -svm[l] * Fm[(i_shared - 2 + m) * n_var] + Fm[(i_shared - 2 + m) * n_var + 5 + l];
//      }
//      fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled);
//    }
//  }
//
//  // Compute the right characteristic matrix
//  LR(0, 0) = 1.0;
//  LR(0, 1) = kx;
//  LR(0, 2) = ky;
//  LR(0, 3) = kz;
//  LR(0, 4) = 1.0;
//  LR(1, 0) = um - kx * cm;
//  LR(1, 1) = kx * um;
//  LR(1, 2) = ky * um - kz * cm;
//  LR(1, 3) = kz * um + ky * cm;
//  LR(1, 4) = um + kx * cm;
//  LR(2, 0) = vm - ky * cm;
//  LR(2, 1) = kx * vm + kz * cm;
//  LR(2, 2) = ky * vm;
//  LR(2, 3) = kz * vm - kx * cm;
//  LR(2, 4) = vm + ky * cm;
//  LR(3, 0) = wm - kz * cm;
//  LR(3, 1) = kx * wm - ky * cm;
//  LR(3, 2) = ky * wm + kx * cm;
//  LR(3, 3) = kz * wm;
//  LR(3, 4) = wm + kz * cm;
//  LR(4, 0) = hm - Uk_bar * cm;
//  LR(4, 1) = kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm;
//  LR(4, 2) = ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm;
//  LR(4, 3) = kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm;
//  LR(4, 4) = hm + Uk_bar * cm;
//
//  // Project the flux back to physical space
//  auto fci = &fc[tid * n_var];
//  fci[0] = LR(0, 0) * fChar[0] + LR(0, 1) * fChar[1] + LR(0, 2) * fChar[2] + LR(0, 3) * fChar[3] + LR(0, 4) * fChar[4];
//  fci[1] = LR(1, 0) * fChar[0] + LR(1, 1) * fChar[1] + LR(1, 2) * fChar[2] + LR(1, 3) * fChar[3] + LR(1, 4) * fChar[4];
//  fci[2] = LR(2, 0) * fChar[0] + LR(2, 1) * fChar[1] + LR(2, 2) * fChar[2] + LR(2, 3) * fChar[3] + LR(2, 4) * fChar[4];
//  fci[3] = LR(3, 0) * fChar[0] + LR(3, 1) * fChar[1] + LR(3, 2) * fChar[2] + LR(3, 3) * fChar[3] + LR(3, 4) * fChar[4];
//
//  fci[4] = LR(4, 0) * fChar[0] + LR(4, 1) * fChar[1] + LR(4, 2) * fChar[2] + LR(4, 3) * fChar[3] + LR(4, 4) * fChar[4];
//  real add{0};
//  for (int l = 0; l < n_spec; ++l) {
//    add += alpha_l[l] * fChar[l + 5];
//  }
//  fci[4] -= add * cm * cm / gm1;
//
//  const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//  for (int l = 0; l < n_spec; ++l) {
//    fci[5 + l] = svm[l] * coeff_add + fChar[l + 5];
//  }
//}
//
//template<MixtureModel mix_model>
//__device__ void
//positive_preserving_limiter_split_component(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param,
//                                            int i_shared,
//                                            real dt, int idx_in_mesh, int max_extent, const real *cv_s,
//                                            const real *jac) {
//  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;
//  int ns = n_var - 5;
//  for (int l = 0; l < ns; ++l) {
//    real theta_p = 1.0, theta_m = 1.0;
//    if (idx_in_mesh > -1) {
//      const real up = 0.5 * alpha * cv_s[i_shared * ns + l] * jac[i_shared] - dt * fc[tid * n_var + 5 + l];
//      if (up < 0) {
//        const real up_lf = 0.5 * alpha * cv_s[i_shared * ns + l] * jac[i_shared] - dt * f_1st[tid * ns + l];
//        if (abs(up - up_lf) > 1e-20) {
//          theta_p = (0 - up_lf) / (up - up_lf);
//          if (theta_p > 1)
//            theta_p = 1.0;
//          else if (theta_p < 0)
//            theta_p = 0;
//        }
//      }
//    }
//
//    if (idx_in_mesh < max_extent - 1) {
//      const real um =
//          0.5 * alpha * cv_s[(i_shared + 1) * ns + l] * jac[i_shared + 1] + dt * fc[tid * n_var + 5 + l];
//      if (um < 0) {
//        const real um_lf = 0.5 * alpha * cv_s[(i_shared + 1) * ns + l] * jac[i_shared + 1] + dt * f_1st[tid * ns + l];
//        if (abs(um - um_lf) > 1e-20) {
//          theta_m = (0 - um_lf) / (um - um_lf);
//          if (theta_m > 1)
//            theta_m = 1.0;
//          else if (theta_m < 0)
//            theta_m = 0;
//        }
//      }
//    }
//
//    fc[tid * n_var + 5 + l] =
//        min(theta_p, theta_m) * (fc[tid * n_var + 5 + l] - f_1st[tid * ns + l]) + f_1st[tid * ns + l];
//  }
//}
//
//template<MixtureModel mix_model>
//__global__ void
//compute_convective_term_weno_1D_split_component(cfd::DZone *zone, int direction, int max_extent, DParameter *param) {
//  int labels[3]{0, 0, 0};
//  labels[direction] = 1;
//  const auto tid = (int) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
//  const auto block_dim = (int) (blockDim.x * blockDim.y * blockDim.z);
//  const auto ngg{zone->ngg};
//  const int n_point = block_dim + 2 * ngg - 1;
//
//  int idx[3];
//  idx[0] = (int) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
//  idx[1] = (int) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
//  idx[2] = (int) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
//  idx[direction] -= 1;
//  if (idx[direction] >= max_extent) return;
//
//  // load variables to shared memory
//  extern __shared__ real s[];
//  const auto n_var{param->n_var};
//  const int ns = n_var - 5;
//  auto n_reconstruct{n_var + 2};
//
//  real *cv_b = s;
//  real *cv_s = &cv_b[n_point * 5];
//  real *p = &cv_s[n_point * ns];
//  real *c = &p[n_point];
//  real *metric = &c[n_point];
//  real *jac = &metric[n_point * 3];
//  real *Fp = &jac[n_point];
//  real *Fm = &Fp[n_point * n_var];
//  real *fc = &Fm[n_point * n_var];
//  real *f_1st = nullptr;
//  if (param->positive_preserving)
//    f_1st = &fc[block_dim * n_var];
//
//  const int i_shared = tid - 1 + ngg;
//  cv_b[i_shared * 5] = zone->cv(idx[0], idx[1], idx[2], 0);
//  cv_b[i_shared * 5 + 1] = zone->cv(idx[0], idx[1], idx[2], 1);
//  cv_b[i_shared * 5 + 2] = zone->cv(idx[0], idx[1], idx[2], 2);
//  cv_b[i_shared * 5 + 3] = zone->cv(idx[0], idx[1], idx[2], 3);
//  cv_b[i_shared * 5 + 4] = zone->cv(idx[0], idx[1], idx[2], 4);
//  for (auto l = 0; l < ns; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//    cv_s[i_shared * ns + l] = zone->cv(idx[0], idx[1], idx[2], l + 5);
//  }
//  p[i_shared] = zone->bv(idx[0], idx[1], idx[2], 4);
//  if constexpr (mix_model != MixtureModel::Air)
//    c[i_shared] = zone->acoustic_speed(idx[0], idx[1], idx[2]);
//  else
//    c[i_shared] = sqrt(gamma_air * R_u / mw_air * zone->bv(idx[0], idx[1], idx[2], 5));
//  metric[i_shared * 3] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 1);
//  metric[i_shared * 3 + 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 2);
//  metric[i_shared * 3 + 2] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 3);
//  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
//
//  // ghost cells
//  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
//  int ig_shared[max_additional_ghost_point_loaded];
//  int additional_loaded{0};
//  if (tid < ngg - 1) {
//    ig_shared[additional_loaded] = tid;
//    const int g_idx[3]{idx[0] - (ngg - 1) * labels[0], idx[1] - (ngg - 1) * labels[1],
//                       idx[2] - (ngg - 1) * labels[2]};
//
//    cv_b[tid * 5] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 0);
//    cv_b[tid * 5 + 1] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 1);
//    cv_b[tid * 5 + 2] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 2);
//    cv_b[tid * 5 + 3] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 3);
//    cv_b[tid * 5 + 4] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
//    for (auto l = 0; l < ns; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv_s[tid * ns + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l + 5);
//    }
//    p[tid] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      c[tid] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
//    else
//      c[tid] = sqrt(gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//    metric[tid * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
//    metric[tid * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
//    metric[tid * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
//    jac[tid] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//    ++additional_loaded;
//  }
//  if (tid > block_dim - ngg - 1 || idx[direction] > max_extent - ngg - 1) {
//    int iSh = tid + 2 * ngg - 1;
//    ig_shared[additional_loaded] = iSh;
//    const int g_idx[3]{idx[0] + ngg * labels[0], idx[1] + ngg * labels[1], idx[2] + ngg * labels[2]};
//    cv_b[iSh * 5] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 0);
//    cv_b[iSh * 5 + 1] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 1);
//    cv_b[iSh * 5 + 2] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 2);
//    cv_b[iSh * 5 + 3] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 3);
//    cv_b[iSh * 5 + 4] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
//    for (auto l = 0; l < ns; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv_s[iSh * ns + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l + 5);
//    }
//    p[iSh] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      c[iSh] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
//    else
//      c[iSh] = sqrt(gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//    metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
//    metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
//    metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
//    jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//    ++additional_loaded;
//  }
//  if (idx[direction] == max_extent - 1 && tid < ngg - 1) {
//    int n_more_left = ngg - 1 - tid - 1;
//    for (int m = 0; m < n_more_left; ++m) {
//      int iSh = tid + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int g_idx[3]{idx[0] - (ngg - 1 - m - 1) * labels[0], idx[1] - (ngg - 1 - m - 1) * labels[1],
//                         idx[2] - (ngg - 1 - m - 1) * labels[2]};
//
//      cv_b[iSh * 5] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 0);
//      cv_b[iSh * 5 + 1] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 1);
//      cv_b[iSh * 5 + 2] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 2);
//      cv_b[iSh * 5 + 3] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 3);
//      cv_b[iSh * 5 + 4] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
//      for (auto l = 0; l < ns; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv_s[iSh * ns + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l + 5);
//      }
//      p[iSh] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        c[iSh] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
//      else
//        c[iSh] = sqrt(gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//      metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
//      metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
//      metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
//      jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//      ++additional_loaded;
//    }
//    int n_more_right = ngg - 1 - tid;
//    for (int m = 0; m < n_more_right; ++m) {
//      int iSh = i_shared + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int g_idx[3]{idx[0] + (m + 1) * labels[0], idx[1] + (m + 1) * labels[1], idx[2] + (m + 1) * labels[2]};
//      cv_b[iSh * 5] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 0);
//      cv_b[iSh * 5 + 1] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 1);
//      cv_b[iSh * 5 + 2] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 2);
//      cv_b[iSh * 5 + 3] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 3);
//      cv_b[iSh * 5 + 4] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
//      for (auto l = 0; l < ns; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv_s[iSh * ns + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l + 5);
//      }
//      p[iSh] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        c[iSh] = zone->acoustic_speed(g_idx[0], g_idx[1], g_idx[2]);
//      else
//        c[iSh] = sqrt(gamma_air * R_u / mw_air * zone->bv(g_idx[0], g_idx[1], g_idx[2], 5));
//      metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
//      metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
//      metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
//      jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
//      ++additional_loaded;
//    }
//  }
//  __syncthreads();
//
//  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//  if (auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
////    compute_weno_flux_cp<mix_model>(cv_b, param, tid, metric, jac, fc, i_shared, Fp, Fm, ig_shared, additional_loaded,
////                                    f_1st);
//  } else if (sch == 52 || sch == 72) {
//    compute_weno_flux_ch_split_component<mix_model>(cv_b, cv_s, p, c, param, tid, metric, jac, fc, i_shared, Fp, Fm,
//                                                    ig_shared, additional_loaded, f_1st);
//  }
//  __syncthreads();
//
//  if (param->positive_preserving) {
//    real dt{0};
//    if (param->dt > 0)
//      dt = param->dt;
//    else
//      dt = zone->dt_local(idx[0], idx[1], idx[2]);
//    positive_preserving_limiter_split_component<mix_model>(f_1st, n_var, tid, fc, param, i_shared, dt, idx[direction],
//                                                           max_extent, cv_s, jac);
//  }
//  __syncthreads();
//
//  if (tid > 0) {
//    for (int l = 0; l < n_var; ++l) {
//      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//    }
//  }
//}
//
//template<MixtureModel mix_model>
//__device__ void
//compute_flux_no_shared_mem(cfd::DZone *zone, DParameter *param, int i, int j, int k, const real *metric, real *Q) {
//  const int n_var = param->n_var;
//  const auto &cv = zone->cv;
//  auto &Fp = zone->Fp;
//  auto &Fm = zone->Fm;
//
//  const real rho{cv(i, j, k, 0)};
//  const real rhoU{cv(i, j, k, 1)}, rhoV{cv(i, j, k, 2)}, rhoW{cv(i, j, k, 3)};
//  const real Uk{(rhoU * metric[0] + rhoV * metric[1] + rhoW * metric[2]) / rho};
//  const real pk{zone->bv(i, j, k, 4)};
//  real cc{0};
//  if constexpr (mix_model == MixtureModel::Air) {
//    cc = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//  } else {
//    cc = zone->acoustic_speed(i, j, k);
//  }
//  const real cGradK = cc * norm3d(metric[0], metric[1], metric[2]);
//  const real lambda0 = abs(Uk) + cGradK;
//  const real jac = zone->jac(i, j, k);
//  const real rhoE{cv(i, j, k, 4)};
//
//  Fp(i, j, k, 0) = 0.5 * jac * (Uk * rho + lambda0 * rho);
//  Fp(i, j, k, 1) = 0.5 * jac * (Uk * rhoU + pk * metric[0] + lambda0 * rhoU);
//  Fp(i, j, k, 2) = 0.5 * jac * (Uk * rhoV + pk * metric[1] + lambda0 * rhoV);
//  Fp(i, j, k, 3) = 0.5 * jac * (Uk * rhoW + pk * metric[2] + lambda0 * rhoW);
//  Fp(i, j, k, 4) = 0.5 * jac * (Uk * (rhoE + pk) + lambda0 * rhoE);
//
//  Fm(i, j, k, 0) = 0.5 * jac * (Uk * rho - lambda0 * rho);
//  Fm(i, j, k, 1) = 0.5 * jac * (Uk * rhoU + pk * metric[0] - lambda0 * rhoU);
//  Fm(i, j, k, 2) = 0.5 * jac * (Uk * rhoV + pk * metric[1] - lambda0 * rhoV);
//  Fm(i, j, k, 3) = 0.5 * jac * (Uk * rhoW + pk * metric[2] - lambda0 * rhoW);
//  Fm(i, j, k, 4) = 0.5 * jac * (Uk * (rhoE + pk) - lambda0 * rhoE);
//
//  for (int l = 5; l < n_var; ++l) {
//    const real rhoYl{cv(i, j, k, l)};
//    Fp(i, j, k, l) = 0.5 * jac * (Uk * rhoYl + lambda0 * rhoYl);
//    Fm(i, j, k, l) = 0.5 * jac * (Uk * rhoYl - lambda0 * rhoYl);
//  }
//}
//
//template<MixtureModel mix_model/*, int ORDER = 7*/>
//__global__ void
//__launch_bounds__(64, 8)
//compute_half_point_flux_weno_x_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param) {
//  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
//  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
//  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (i >= max_extent) return;
//
//  const auto &cv = zone->cv;
//  const auto &bv = zone->bv;
//
//  const auto nv{param->n_var};
//  const int ns{param->n_spec};
//
//  // Load the cv
//  real QL[5 + MAX_SPEC_NUMBER], QR[5 + MAX_SPEC_NUMBER];
//  for (int l = 0; l < nv; ++l) {
//    QL[l] = cv(i, j, k, l);
//    QR[l] = cv(i + 1, j, k, l);
//  }
//
//  // compute the Roe average
//  const real rhoL_inv = 1.0 / QL[0], rhoR_inv = 1.0 / QR[0];
//  const real pl = bv(i, j, k, 4), pr = bv(i + 1, j, k, 4);
//  const real rlc{sqrt(QL[0]) / (sqrt(QL[0]) + sqrt(QR[0]))};
//  const real rrc{sqrt(QR[0]) / (sqrt(QL[0]) + sqrt(QR[0]))};
//  const real um = rlc * QL[1] * rhoL_inv + rrc * QR[1] * rhoR_inv;
//  const real vm = rlc * QL[2] * rhoL_inv + rrc * QR[2] * rhoR_inv;
//  const real wm = rlc * QL[3] * rhoL_inv + rrc * QR[3] * rhoR_inv;
//  const real ekm = 0.5 * (um * um + vm * vm + wm * wm);
//  const real hm = rlc * (QL[4] + pl) * rhoL_inv + rrc * (QR[4] + pr) * rhoR_inv;
//
//  real svm[MAX_SPEC_NUMBER];
//  memset(svm, 0, MAX_SPEC_NUMBER * sizeof(real));
//  for (int l = 0; l < ns; ++l) {
//    svm[l] = (rlc * QL[l + 5] * rhoL_inv + rrc * QR[l + 5] * rhoR_inv);
//  }
//
//  real mw_inv = 0;
//  for (int l = 0; l < ns; ++l) {
//    mw_inv += svm[l] / param->mw[l];
//  }
//  const real tm = (rlc * pl * rhoL_inv + rrc * pr * rhoR_inv) / (R_u * mw_inv);
//
//  real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
//  compute_enthalpy_and_cp(tm, h_i, cp_i, param);
//  real cp{0}, cv_tot{0};
//  for (int l = 0; l < ns; ++l) {
//    cp += svm[l] * cp_i[l];
//    cv_tot += svm[l] * (cp_i[l] - R_u / param->mw[l]);
//  }
//  const real gamma = cp / cv_tot;
//  const real cm = sqrt(gamma * R_u * mw_inv * tm);
//  const real gm1{gamma - 1};
//
//  // Next, we compute the left characteristic matrix at i+1/2.
//  const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i + 1, j, k)};
//  real m_l[3]{zone->metric(i, j, k)(1, 1), zone->metric(i, j, k)(1, 2), zone->metric(i, j, k)(1, 3)};
//  real m_r[3]{zone->metric(i + 1, j, k)(1, 1), zone->metric(i + 1, j, k)(1, 2), zone->metric(i + 1, j, k)(1, 3)};
//  real kxJ{m_l[0] * jac_l + m_r[0] * jac_r};
//  real kyJ{m_l[1] * jac_l + m_r[1] * jac_r};
//  real kzJ{m_l[2] * jac_l + m_r[2] * jac_r};
//  real kx{kxJ / (jac_l + jac_r)};
//  real ky{kyJ / (jac_l + jac_r)};
//  real kz{kzJ / (jac_l + jac_r)};
//  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
//  kx /= gradK;
//  ky /= gradK;
//  kz /= gradK;
//  const real Uk_bar{kx * um + ky * vm + kz * wm};
//  const real alpha{gm1 * ekm};
//
//  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
//  const real cm2_inv{1.0 / (cm * cm)};
//  gxl::Matrix<real, 5, 5> LR;
//  LR(0, 0) = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//  LR(0, 1) = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//  LR(0, 2) = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//  LR(0, 3) = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//  LR(0, 4) = gm1 * cm2_inv * 0.5;
//  LR(1, 0) = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//  LR(1, 1) = kx * gm1 * um * cm2_inv;
//  LR(1, 2) = (kx * gm1 * vm + kz * cm) * cm2_inv;
//  LR(1, 3) = (kx * gm1 * wm - ky * cm) * cm2_inv;
//  LR(1, 4) = -kx * gm1 * cm2_inv;
//  LR(2, 0) = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//  LR(2, 1) = (ky * gm1 * um - kz * cm) * cm2_inv;
//  LR(2, 2) = ky * gm1 * vm * cm2_inv;
//  LR(2, 3) = (ky * gm1 * wm + kx * cm) * cm2_inv;
//  LR(2, 4) = -ky * gm1 * cm2_inv;
//  LR(3, 0) = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//  LR(3, 1) = (kz * gm1 * um + ky * cm) * cm2_inv;
//  LR(3, 2) = (kz * gm1 * vm - kx * cm) * cm2_inv;
//  LR(3, 3) = kz * gm1 * wm * cm2_inv;
//  LR(3, 4) = -kz * gm1 * cm2_inv;
//  LR(4, 0) = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//  LR(4, 1) = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//  LR(4, 2) = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//  LR(4, 3) = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//  LR(4, 4) = gm1 * cm2_inv * 0.5;
//
//  // Compute the characteristic flux with L.
//  real fChar[5 + MAX_SPEC_NUMBER];
//  constexpr real eps{1e-40};
//  const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);
//
//  real alpha_l[MAX_SPEC_NUMBER];
//  // compute the partial derivative of pressure to species density
//  for (int l = 0; l < ns; ++l) {
//    alpha_l[l] = gamma * R_u / param->mw[l] * tm - (gamma - 1) * h_i[l];
//    // The computations including this alpha_l are all combined with a division by cm2.
//    alpha_l[l] *= cm2_inv;
//  }
//
//  auto &Fp = zone->Fp;
//  auto &Fm = zone->Fm;
//  bool pp_limiter{param->positive_preserving};
//  if (pp_limiter) {
//    real cc{0}, ccr;
//    if constexpr (mix_model == MixtureModel::Air) {
//      cc = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//      ccr = sqrt(gamma_air * R_air * zone->bv(i + 1, j, k, 5));
//    } else {
//      cc = zone->acoustic_speed(i, j, k);
//      ccr = zone->acoustic_speed(i + 1, j, k);
//    }
//    real spectralRadThis = abs((m_l[0] * QL[1] + m_l[1] * QL[2] + m_l[2] * QL[3]) * rhoL_inv +
//                               cc * sqrt(m_l[0] * m_l[0] + m_l[1] * m_l[1] + m_l[2] * m_l[2]));
//    real spectralRadNext = abs((m_r[0] * QR[1] + m_r[1] * QR[2] + m_r[2] * QR[3]) * rhoR_inv +
//                               ccr * sqrt(m_r[0] * m_r[0] + m_r[1] * m_r[1] + m_r[2] * m_r[2]));
//    for (int l = 0; l < nv - 5; ++l) {
//      zone->f_1st(i, j, k, l) = 0.5 * (Fp(i, j, k, l + 5) + spectralRadThis * QL[l + 5] * jac_l) +
//                                0.5 * (Fp(i + 1, j, k, l + 5) - spectralRadNext * QR[l + 5] * jac_r);
//    }
//  }
//
//  if (param->inviscid_scheme == 52) {
////    for (int l = 0; l < 5; ++l) {
////      real coeff_alpha_s{0.5};
////      if (l == 1) {
////        coeff_alpha_s = -kx;
////      } else if (l == 2) {
////        coeff_alpha_s = -ky;
////      } else if (l == 3) {
////        coeff_alpha_s = -kz;
////      }
////      real vPlus[5], vMinus[5];
////      memset(vPlus, 0, 5 * sizeof(real));
////      memset(vMinus, 0, 5 * sizeof(real));
////      for (int m = 0; m < 5; ++m) {
////        for (int n = 0; n < 5; ++n) {
////          vPlus[m] += LR(l, n) * Fp[(i_shared - 2 + m) * n_var + n];
////          vMinus[m] += LR(l, n) * Fm[(i_shared - 1 + m) * n_var + n];
////        }
////        for (int n = 0; n < n_spec; ++n) {
////          vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 2 + m) * n_var + 5 + n];
////          vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 1 + m) * n_var + 5 + n];
////        }
////      }
////      fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
////    }
////    for (int l = 0; l < n_spec; ++l) {
////      real vPlus[5], vMinus[5];
////      for (int m = 0; m < 5; ++m) {
////        vPlus[m] = -svm[l] * Fp[(i_shared - 2 + m) * n_var] + Fp[(i_shared - 2 + m) * n_var + 5 + l];
////        vMinus[m] = -svm[l] * Fm[(i_shared - 1 + m) * n_var] + Fm[(i_shared - 1 + m) * n_var + 5 + l];
////      }
////      fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled);
////    }
//  } else if (param->inviscid_scheme == 72) {
//    for (int l = 0; l < 5; ++l) {
//      real coeff_alpha_s{0.5};
//      if (l == 1) {
//        coeff_alpha_s = -kx;
//      } else if (l == 2) {
//        coeff_alpha_s = -ky;
//      } else if (l == 3) {
//        coeff_alpha_s = -kz;
//      }
//      real vPlus[7], vMinus[7];
//      memset(vPlus, 0, 7 * sizeof(real));
//      memset(vMinus, 0, 7 * sizeof(real));
//      for (int m = -3; m < 4; ++m) {
//        for (int n = 0; n < 5; ++n) {
//          vPlus[m + 3] += LR(l, n) * Fp(i + m, j, k, n);
//          vMinus[m + 3] += LR(l, n) * Fm(i + m + 1, j, k, n);
////          if (isnan(abs(Fp(i + m, j, k, n)))) {
////            printf("Fp(%d, %d, %d, %d) is nan\n", i + m, j, k, n);
////          }
////          if (isnan(abs(Fm(i + m + 1, j, k, n)))) {
////            printf("Fm(%d, %d, %d, %d) is nan\n", i + m + 1, j, k, n);
////          }
//        }
//        for (int n = 0; n < ns; ++n) {
//          vPlus[m + 3] += coeff_alpha_s * alpha_l[n] * Fp(i + m, j, k, 5 + n);
//          vMinus[m + 3] += coeff_alpha_s * alpha_l[n] * Fm(i + m + 1, j, k, 5 + n);
////          if (isnan(abs(Fp(i + m, j, k, 5 + n)))) {
////            printf("Fp(%d, %d, %d, %d) is nan\n", i + m, j, k, 5 + n);
////          }
////          if (isnan(abs(Fm(i + m + 1, j, k, 5 + n)))) {
////            printf("Fm(%d, %d, %d, %d) is nan\n", i + m + 1, j, k, 5 + n);
////          }
//        }
//      }
//      fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
//    }
//    for (int l = 0; l < ns; ++l) {
//      real vPlus[7], vMinus[7];
//      for (int m = -3; m < 4; ++m) {
//        vPlus[m + 3] = -svm[l] * Fp(i + m, j, k, 0) + Fp(i + m, j, k, 5 + l);
//        vMinus[m + 3] = -svm[l] * Fm(i + m + 1, j, k, 0) + Fm(i + m + 1, j, k, 5 + l);
//      }
//      fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled);
//    }
//  }
//
//  // Compute the right characteristic matrix
//  LR(0, 0) = 1.0;
//  LR(0, 1) = kx;
//  LR(0, 2) = ky;
//  LR(0, 3) = kz;
//  LR(0, 4) = 1.0;
//  LR(1, 0) = um - kx * cm;
//  LR(1, 1) = kx * um;
//  LR(1, 2) = ky * um - kz * cm;
//  LR(1, 3) = kz * um + ky * cm;
//  LR(1, 4) = um + kx * cm;
//  LR(2, 0) = vm - ky * cm;
//  LR(2, 1) = kx * vm + kz * cm;
//  LR(2, 2) = ky * vm;
//  LR(2, 3) = kz * vm - kx * cm;
//  LR(2, 4) = vm + ky * cm;
//  LR(3, 0) = wm - kz * cm;
//  LR(3, 1) = kx * wm - ky * cm;
//  LR(3, 2) = ky * wm + kx * cm;
//  LR(3, 3) = kz * wm;
//  LR(3, 4) = wm + kz * cm;
//  LR(4, 0) = hm - Uk_bar * cm;
//  LR(4, 1) = kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm;
//  LR(4, 2) = ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm;
//  LR(4, 3) = kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm;
//  LR(4, 4) = hm + Uk_bar * cm;
//
//  // Project the flux back to physical space
//  auto &fc = zone->flux;
//  fc(i, j, k, 0) =
//      LR(0, 0) * fChar[0] + LR(0, 1) * fChar[1] + LR(0, 2) * fChar[2] + LR(0, 3) * fChar[3] + LR(0, 4) * fChar[4];
//  fc(i, j, k, 1) =
//      LR(1, 0) * fChar[0] + LR(1, 1) * fChar[1] + LR(1, 2) * fChar[2] + LR(1, 3) * fChar[3] + LR(1, 4) * fChar[4];
//  fc(i, j, k, 2) =
//      LR(2, 0) * fChar[0] + LR(2, 1) * fChar[1] + LR(2, 2) * fChar[2] + LR(2, 3) * fChar[3] + LR(2, 4) * fChar[4];
//  fc(i, j, k, 3) =
//      LR(3, 0) * fChar[0] + LR(3, 1) * fChar[1] + LR(3, 2) * fChar[2] + LR(3, 3) * fChar[3] + LR(3, 4) * fChar[4];
//
//  fc(i, j, k, 4) =
//      LR(4, 0) * fChar[0] + LR(4, 1) * fChar[1] + LR(4, 2) * fChar[2] + LR(4, 3) * fChar[3] + LR(4, 4) * fChar[4];
//  real add{0};
//  for (int l = 0; l < ns; ++l) {
//    add += alpha_l[l] * fChar[5 + l];
//  }
//  fc(i, j, k, 4) -= add * cm * cm / gm1;
//
//  const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//  for (int l = 0; l < ns; ++l) {
//    fc(i, j, k, 5 + l) = svm[l] * coeff_add + fChar[l + 5];
//  }
////  if ((i == 1 || i == 2) && j == 146 && k == 85) {
////    printf("fc(%d,%d,%d,0:7)=(%e,%e,%e,%e,%e,%e,%e,%e)\n", i, j, k, fc(i, j, k, 0), fc(i, j, k, 1), fc(i, j, k, 2),
////           fc(i, j, k, 3), fc(i, j, k, 4), fc(i, j, k, 5), fc(i, j, k, 6), fc(i, j, k, 7));
////  }
////  for (int l = 0; l < nv; ++l) {
////    if (isnan(abs(fc(i, j, k, l)))) {
////      printf("fc(%d, %d, %d, %d) is nan, fChar[0:4]={%f, %f, %f, %f, %f}\n", i, j, k, l, fChar[0], fChar[1], fChar[2],
////             fChar[3], fChar[4]);
////      break;
////    }
////  }
//  __syncthreads();
//
//  if (param->positive_preserving && ns > 0) {
//    real dt{0};
//    if (param->dt > 0)
//      dt = param->dt;
//    else
//      dt = zone->dt_local(i, j, k);
//
//    const real a = param->dim == 3 ? 1.0 / 3.0 : 0.5;
//    auto &f_1st = zone->f_1st;
//    for (int l = 0; l < ns; ++l) {
//      real theta_p = 1.0, theta_m = 1.0;
//      if (i > -1) {
//        const real up = 0.5 * a * QL[5 + l] * jac_l - dt * fc(i, j, k, 5 + l);
//        if (up < 0) {
//          const real up_lf = 0.5 * alpha * QL[5 + l] * jac_l - dt * f_1st(i, j, k, l);
//          if (abs(up - up_lf) > 1e-20) {
//            theta_p = (0 - up_lf) / (up - up_lf);
//            if (theta_p > 1)
//              theta_p = 1.0;
//            else if (theta_p < 0)
//              theta_p = 0;
//          }
//        }
//      }
//
//      if (i < max_extent - 1) {
//        const real u_minus = 0.5 * a * QR[5 + l] * jac_r + dt * fc(i, j, k, 5 + l);
//        if (u_minus < 0) {
//          const real um_lf = 0.5 * alpha * QR[5 + l] * jac_r + dt * f_1st(i, j, k, l);
//          if (abs(u_minus - um_lf) > 1e-20) {
//            theta_m = (0 - um_lf) / (u_minus - um_lf);
//            if (theta_m > 1)
//              theta_m = 1.0;
//            else if (theta_m < 0)
//              theta_m = 0;
//          }
//        }
//      }
//
//      fc(i, j, k, l) = min(theta_p, theta_m) * (fc(i, j, k, l) - f_1st(i, j, k, l)) + f_1st(i, j, k, l);
//    }
//  }
//}
//
//template<MixtureModel mix_model>
//__global__ void
//__launch_bounds__(64, 8)
//compute_convective_term_weno_y_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param) {
//  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
//  int j = (int) ((blockDim.y - 1) * blockIdx.y + threadIdx.y - 1);
//  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (j >= max_extent) return;
//
//  const auto tid = (int) threadIdx.y;
//  const auto block_dim = (int) blockDim.y;
//  const auto ngg{zone->ngg};
//  const auto n_var{param->n_var};
//  const auto n_reconstruct{n_var + 2};
//  const int n_point = block_dim + 2 * ngg - 1;
//
//  extern __shared__ real s[];
//  real *cv = s;
//  real *metric = &cv[n_point * n_reconstruct];
//  real *jac = &metric[n_point * 3];
//  real *fp = &jac[n_point];
//  real *fm = &fp[n_point * n_var];
//  real *fc = &fm[n_point * n_var];
//  real *f_1st = nullptr;
//  if (param->positive_preserving)
//    f_1st = &fc[block_dim * n_var];
//
//  const int i_shared = tid - 1 + ngg;
//  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//    cv[i_shared * n_reconstruct + l] = zone->cv(i, j, k, l);
//  }
//  cv[i_shared * n_reconstruct + n_var] = zone->bv(i, j, k, 4);
//  if constexpr (mix_model != MixtureModel::Air)
//    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, k);
//  else
//    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//  metric[i_shared * 3] = zone->metric(i, j, k)(2, 1);
//  metric[i_shared * 3 + 1] = zone->metric(i, j, k)(2, 2);
//  metric[i_shared * 3 + 2] = zone->metric(i, j, k)(2, 3);
//  jac[i_shared] = zone->jac(i, j, k);
//
//  // ghost cells
//  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
//  int ig_shared[max_additional_ghost_point_loaded];
//  int additional_loaded{0};
//  if (tid < ngg - 1) {
//    ig_shared[additional_loaded] = tid;
//    const int gj = j - (ngg - 1);
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv[tid * n_reconstruct + l] = zone->cv(i, gj, k, l);
//    }
//    cv[tid * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      cv[tid * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
//    else
//      cv[tid * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//    metric[tid * 3] = zone->metric(i, gj, k)(2, 1);
//    metric[tid * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//    metric[tid * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//    jac[tid] = zone->jac(i, gj, k);
//    ++additional_loaded;
//  }
//  if (tid > block_dim - ngg - 1 || j > max_extent - ngg - 1) {
//    int iSh = tid + 2 * ngg - 1;
//    ig_shared[additional_loaded] = iSh;
//    const int gj = j + ngg;
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
//    }
//    cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
//    else
//      cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//    metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//    metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//    metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//    jac[iSh] = zone->jac(i, gj, k);
//    ++additional_loaded;
//  }
//  if (j == max_extent - 1 && tid < ngg - 1) {
//    int n_more_left = ngg - 1 - tid - 1;
//    for (int m = 0; m < n_more_left; ++m) {
//      int iSh = tid + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int gj = j - (ngg - 1 - m - 1);
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
//      }
//      cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
//      else
//        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//      metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//      metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//      metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//      jac[iSh] = zone->jac(i, gj, k);
//      ++additional_loaded;
//    }
//    int n_more_right = ngg - 1 - tid;
//    for (int m = 0; m < n_more_right; ++m) {
//      int iSh = i_shared + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int gj = j + (m + 1);
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
//      }
//      cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
//      else
//        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//      metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//      metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//      metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//      jac[iSh] = zone->jac(i, gj, k);
//      ++additional_loaded;
//    }
//  }
//  __syncthreads();
//
//  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//  if (auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//    compute_weno_flux_cp<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
//                                    f_1st);
//  } else if (sch == 52 || sch == 72) {
//    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
//                                    f_1st);
//  }
//  __syncthreads();
//
//  if (param->positive_preserving) {
//    real dt{0};
//    if (param->dt > 0)
//      dt = param->dt;
//    else
//      dt = zone->dt_local(i, j, k);
//    positive_preserving_limiter<mix_model>(f_1st, n_var, tid, fc, param, i_shared, dt, j, max_extent, cv, jac);
//  }
//  __syncthreads();
//
//  if (tid > 0) {
//    for (int l = 0; l < n_var; ++l) {
//      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//    }
//  }
//}
//
//template<MixtureModel mix_model>
//__global__ void
//__launch_bounds__(64, 8)
//compute_convective_term_weno_z_no_shared_mem(cfd::DZone *zone, int max_extent, DParameter *param) {
//  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
//  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
//  int k = (int) ((blockDim.z - 1) * blockIdx.z + threadIdx.z - 1);
//  if (k >= max_extent) return;
//
//  const auto tid = (int) threadIdx.z;
//  const auto block_dim = (int) blockDim.z;
//  const auto ngg{zone->ngg};
//  const auto n_var{param->n_var};
//  const auto n_reconstruct{n_var + 2};
//  const int n_point = block_dim + 2 * ngg - 1;
//
//  extern __shared__ real s[];
//  real *cv = s;
//  real *metric = &cv[n_point * n_reconstruct];
//  real *jac = &metric[n_point * 3];
//  real *fp = &jac[n_point];
//  real *fm = &fp[n_point * n_var];
//  real *fc = &fm[n_point * n_var];
//  real *f_1st = nullptr;
//  if (param->positive_preserving)
//    f_1st = &fc[block_dim * n_var];
//
//  const int i_shared = tid - 1 + ngg;
//  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//    cv[i_shared * n_reconstruct + l] = zone->cv(i, j, k, l);
//  }
//  cv[i_shared * n_reconstruct + n_var] = zone->bv(i, j, k, 4);
//  if constexpr (mix_model != MixtureModel::Air)
//    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, k);
//  else
//    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//  metric[i_shared * 3] = zone->metric(i, j, k)(3, 1);
//  metric[i_shared * 3 + 1] = zone->metric(i, j, k)(3, 2);
//  metric[i_shared * 3 + 2] = zone->metric(i, j, k)(3, 3);
//  jac[i_shared] = zone->jac(i, j, k);
//
//  // ghost cells
//  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
//  int ig_shared[max_additional_ghost_point_loaded];
//  int additional_loaded{0};
//  if (tid < ngg - 1) {
//    ig_shared[additional_loaded] = tid;
//    const int gk = k - (ngg - 1);
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv[tid * n_reconstruct + l] = zone->cv(i, j, gk, l);
//    }
//    cv[tid * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      cv[tid * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
//    else
//      cv[tid * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//    metric[tid * 3] = zone->metric(i, j, gk)(3, 1);
//    metric[tid * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//    metric[tid * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//    jac[tid] = zone->jac(i, j, gk);
//    ++additional_loaded;
//  }
//  if (tid > block_dim - ngg - 1 || k > max_extent - ngg - 1) {
//    int iSh = tid + 2 * ngg - 1;
//    ig_shared[additional_loaded] = iSh;
//    const int gk = k + ngg;
//    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//      cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
//    }
//    cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
//    if constexpr (mix_model != MixtureModel::Air)
//      cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
//    else
//      cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//    metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//    metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//    metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//    jac[iSh] = zone->jac(i, j, gk);
//    ++additional_loaded;
//  }
//  if (k == max_extent - 1 && tid < ngg - 1) {
//    int n_more_left = ngg - 1 - tid - 1;
//    for (int m = 0; m < n_more_left; ++m) {
//      int iSh = tid + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int gk = k - (ngg - 1 - m - 1);
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
//        cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
//      }
//      cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
//      else
//        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//      metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//      metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//      metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//      jac[iSh] = zone->jac(i, j, gk);
//      ++additional_loaded;
//    }
//    int n_more_right = ngg - 1 - tid;
//    for (int m = 0; m < n_more_right; ++m) {
//      int iSh = i_shared + m + 1;
//      ig_shared[additional_loaded] = iSh;
//      const int gk = k + (m + 1);
//      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E
//        cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
//      }
//      cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
//      if constexpr (mix_model != MixtureModel::Air)
//        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
//      else
//        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//      metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//      metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//      metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//      jac[iSh] = zone->jac(i, j, gk);
//      ++additional_loaded;
//    }
//  }
//  __syncthreads();
//
//  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//  if (auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//    compute_weno_flux_cp<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
//                                    f_1st);
//  } else if (sch == 52 || sch == 72) {
//    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
//                                    f_1st);
//  }
//  __syncthreads();
//
//  if (param->positive_preserving) {
//    real dt{0};
//    if (param->dt > 0)
//      dt = param->dt;
//    else
//      dt = zone->dt_local(i, j, k);
//    positive_preserving_limiter<mix_model>(f_1st, n_var, tid, fc, param, i_shared, dt, k, max_extent, cv, jac);
//  }
//  __syncthreads();
//
//  if (tid > 0) {
//    for (int l = 0; l < n_var; ++l) {
//      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//    }
//  }
//}
//
//__device__ real WENO5_opt(const real *vp, const real *vm, real eps) {
//  constexpr real one6th{1.0 / 6};
//  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
//  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
//  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
//  constexpr real thirteen12th{13.0 / 12};
//  real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
//               0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
//  real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
//               0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
//  real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
//               0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
//  constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
//  real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
//  real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
//  real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
//  real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
//  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};
//
//  v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
//  v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
//  v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
//  beta0 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
//          0.25 * (3 * vm[2] - 4 * vm[3] + vm[4]) * (3 * vm[2] - 4 * vm[3] + vm[4]);
//  beta1 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
//          0.25 * (vm[1] - vm[3]) * (vm[1] - vm[3]);
//  beta2 = thirteen12th * (vm[0] + vm[2] - 2 * vm[1]) * (vm[0] + vm[2] - 2 * vm[1]) +
//          0.25 * (vm[0] - 4 * vm[1] + 3 * vm[2]) * (vm[0] - 4 * vm[1] + 3 * vm[2]);
//  tau5sqr = (beta0 - beta2) * (beta0 - beta2);
//  a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
//  a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
//  a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
//  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};
//
//  return (fPlus + fMinus);
//}
//
//__device__ real WENO7_opt(const real *vp, const real *vm, real eps) {
//  constexpr real one6th{1.0 / 6};
//  // 1st order derivative
//  real s10{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
//  real s11{one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4])};
//  real s12{one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5])};
//  real s13{one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6])};
//  // 2nd order derivative
//  real s20{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
//  real s21{vp[2] - 2 * vp[3] + vp[4]};
//  real s22{s21};
//  real s23{2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6]};
//  // 3rd order derivative
//  real s30{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
//  real s31{-vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4]};
//  real s32{-vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5]};
//  real s33{-vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6]};
//
//  constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};
//  real beta0{s10 * s10 + d12 * s20 * s20 + d13 * s30 * s30 + d14 * s10 * s30};
//  real beta1{s11 * s11 + d12 * s21 * s21 + d13 * s31 * s31 + d14 * s11 * s31};
//  real beta2{s12 * s12 + d12 * s22 * s22 + d13 * s32 * s32 + d14 * s12 * s32};
//  real beta3{s13 * s13 + d12 * s23 * s23 + d13 * s33 * s33 + d14 * s13 * s33};
//
//  real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
//  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
//  real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
//  real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
//  real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
//  real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};
//
//  constexpr real one12th{1.0 / 12};
//  real v0{one12th * (-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3])};
//  real v1{one12th * (vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4])};
//  real v2{one12th * (-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5])};
//  real v3{one12th * (3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6])};
//  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};
//
//  // Minus part
//  s10 = one6th * (-2 * vm[6] + 9 * vm[5] - 18 * vm[4] + 11 * vm[3]);
//  s11 = one6th * (vm[5] - 6 * vm[4] + 3 * vm[3] + 2 * vm[2]);
//  s12 = one6th * (-2 * vm[4] - 3 * vm[3] + 6 * vm[2] - vm[1]);
//  s13 = one6th * (-11 * vm[3] + 18 * vm[2] - 9 * vm[1] + 2 * vm[0]);
//
//  s20 = -vm[6] + 4 * vm[5] - 5 * vm[4] + 2 * vm[3];
//  s21 = vm[4] - 2 * vm[3] + vm[2];
//  s22 = s21;
//  s23 = 2 * vm[3] - 5 * vm[2] + 4 * vm[1] - vm[0];
//
//  s30 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
//  s31 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
//  s32 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
//  s33 = -vm[3] + 3 * vm[2] - 3 * vm[1] + vm[0];
//
//  beta0 = s10 * s10 + d12 * s20 * s20 + d13 * s30 * s30 + d14 * s10 * s30;
//  beta1 = s11 * s11 + d12 * s21 * s21 + d13 * s31 * s31 + d14 * s11 * s31;
//  beta2 = s12 * s12 + d12 * s22 * s22 + d13 * s32 * s32 + d14 * s12 * s32;
//  beta3 = s13 * s13 + d12 * s23 * s23 + d13 * s33 * s33 + d14 * s13 * s33;
//
//  tau7sqr = (beta0 - beta3) * (beta0 - beta3);
//  a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
//  a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
//  a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
//  a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));
//
//  v0 = one12th * (-3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3]);
//  v1 = one12th * (vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2]);
//  v2 = one12th * (-vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1]);
//  v3 = one12th * (3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0]);
//  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};
//
//  return (fPlus + fMinus);
//}
//
//template void
//compute_convective_term_weno_opt<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
//                                                    const Parameter &parameter);
//
//template void
//compute_convective_term_weno_opt<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
//                                                        int n_var, const Parameter &parameter);
//
//template void
//compute_convective_term_weno_opt<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
//                                                                int n_var, const Parameter &parameter);
//
//template void
//compute_convective_term_weno_opt<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
//                                                   const Parameter &parameter);
//
//template void
//compute_convective_term_weno_opt<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
//                                                   const Parameter &parameter);
//}