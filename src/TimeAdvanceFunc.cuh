#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "Field.h"
#include "Constants.h"
#include "TurbMethod.hpp"
#include "FieldOperation.cuh"

namespace cfd {
__global__ void store_last_step(DZone *zone);

template<MixtureModel mixture, class turb_method>
__global__ void local_time_step(DZone *zone, DParameter *param);

__global__ void compute_square_of_dbv(DZone *zone);

real global_time_step(const Mesh &mesh, const Parameter &parameter, const std::vector<Field> &field);

__global__ void min_of_arr(real *arr, int size);

__global__ void update_physical_time(DParameter *param, real t);

template<MixtureModel mixture, class turb_method>
__global__ void limit_flow(DZone *zone, DParameter *param);
}

template<MixtureModel mixture, class turb_method>
__global__ void cfd::local_time_step(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &m{zone->metric(i, j, k)};
  const auto &bv = zone->bv;
  const int dim{zone->mz == 1 ? 2 : 3};

  const real grad_xi = std::sqrt(m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2) + m(1, 3) * m(1, 3));
  const real grad_eta = std::sqrt(m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2) + m(2, 3) * m(2, 3));
  const real grad_zeta = std::sqrt(m(3, 1) * m(3, 1) + m(3, 2) * m(3, 2) + m(3, 3) * m(3, 3));

  const real u{bv(i, j, k, 1)}, v{bv(i, j, k, 2)}, w{bv(i, j, k, 3)};
  const real U = u * m(1, 1) + v * m(1, 2) + w * m(1, 3);
  const real V = u * m(2, 1) + v * m(2, 2) + w * m(2, 3);
  const real W = u * m(3, 1) + v * m(3, 2) + w * m(3, 3);

  real acoustic_speed{0};
  if constexpr (mixture == MixtureModel::Air) {
    acoustic_speed = std::sqrt(gamma_air * R_u * bv(i, j, k, 5) / mw_air);
  } else {
    acoustic_speed = zone->acoustic_speed(i, j, k);
  }
  auto &inviscid_spectral_radius = zone->inv_spectr_rad(i, j, k);
  inviscid_spectral_radius[0] = std::abs(U) + acoustic_speed * grad_xi;
  inviscid_spectral_radius[1] = std::abs(V) + acoustic_speed * grad_eta;
  inviscid_spectral_radius[2] = 0;
  if (dim == 3) {
    inviscid_spectral_radius[2] = std::abs(W) + acoustic_speed * grad_zeta;
  }
  const real spectral_radius_inv =
      inviscid_spectral_radius[0] + inviscid_spectral_radius[1] + inviscid_spectral_radius[2];

  // Next, compute the viscous spectral radius
  real gamma{gamma_air};
  if constexpr (mixture != MixtureModel::Air) {
    gamma = zone->gamma(i, j, k);
  }
  const real coeff_1 = max(gamma, 4.0 / 3.0) / bv(i, j, k, 0);
  real coeff_2 = zone->mul(i, j, k) / param->Pr;
  if constexpr (TurbMethod<turb_method>::hasMut) {
    coeff_2 += zone->mut(i, j, k) / param->Prt;
  }
  real spectral_radius_viscous = grad_xi * grad_xi + grad_eta * grad_eta;
  if (dim == 3) {
    spectral_radius_viscous += grad_zeta * grad_zeta;
  }
  spectral_radius_viscous *= coeff_1 * coeff_2;

  zone->dt_local(i, j, k) = param->cfl / (spectral_radius_inv + spectral_radius_viscous);
//  zone->udv(i, j, k, 0) = zone->dt_local(i, j, k);
}

template<MixtureModel mixture, class turb_method>
__global__ void cfd::limit_flow(DZone *zone, DParameter *param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // Record the computed values. First for flow variables and mass fractions
  constexpr int n_flow_var = 5;
  real var[n_flow_var];
  var[0] = bv(i, j, k, 0);
  var[1] = bv(i, j, k, 1);
  var[2] = bv(i, j, k, 2);
  var[3] = bv(i, j, k, 3);
  var[4] = bv(i, j, k, 4);
  const int n_spec{param->n_spec};

  // Find the unphysical values and limit them
  const auto ll = param->limit_flow.ll;
  const auto ul = param->limit_flow.ul;
  bool unphysical{false};
  for (int l = 0; l < n_flow_var; ++l) {
    if (isnan(var[l])) {
      unphysical = true;
      break;
    }
    if (var[l] < ll[l] || var[l] > ul[l]) {
      unphysical = true;
      break;
    }
  }

  __syncthreads();

  if (unphysical) {
    // printf("Unphysical values appear in process %d, block %d, i = %d, j = %d, k = %d.\n", param->myid, blk_id, i, j, k);

    real updated_var[n_flow_var/* + MAX_SPEC_NUMBER + 4*/] = {};
    int kn{0};
    // Compute the sum of all "good" points surrounding the "bad" point
    for (int ka = -1; ka < 2; ++ka) {
      const int k1{k + ka};
      if (k1 < 0 || k1 >= mz) continue;
      for (int ja = -1; ja < 2; ++ja) {
        const int j1{j + ja};
        if (j1 < 0 || j1 >= my) continue;
        for (int ia = -1; ia < 2; ++ia) {
          const int i1{i + ia};
          if (i1 < 0 || i1 >= mx)continue;

          if (isnan(bv(i1, j1, k1, 0)) || isnan(bv(i1, j1, k1, 1)) || isnan(bv(i1, j1, k1, 2)) ||
              isnan(bv(i1, j1, k1, 3)) || isnan(bv(i1, j1, k1, 4)) || bv(i1, j1, k1, 0) < ll[0] ||
              bv(i1, j1, k1, 1) < ll[1] || bv(i1, j1, k1, 2) < ll[2] || bv(i1, j1, k1, 3) < ll[3] ||
              bv(i1, j1, k1, 4) < ll[4] || bv(i1, j1, k1, 0) > ul[0] || bv(i1, j1, k1, 1) > ul[1] ||
              bv(i1, j1, k1, 2) > ul[2] || bv(i1, j1, k1, 3) > ul[3] || bv(i1, j1, k1, 4) > ul[4]) {
            continue;
          }

          updated_var[0] += bv(i1, j1, k1, 0);
          updated_var[1] += bv(i1, j1, k1, 1);
          updated_var[2] += bv(i1, j1, k1, 2);
          updated_var[3] += bv(i1, j1, k1, 3);
          updated_var[4] += bv(i1, j1, k1, 4);

          ++kn;
        }
      }
    }

    // Compute the average of the surrounding points
    if (kn > 0) {
      const real kn_inv{1.0 / kn};
      for (double & l : updated_var) {
        l *= kn_inv;
      }
    } else {
      // The surrounding points are all "bad."
      for (int l = 0; l < 5; ++l) {
        updated_var[l] = max(var[l], ll[l]);
        updated_var[l] = min(updated_var[l], ul[l]);
      }
    }

    // Assign averaged values for the bad point
    bv(i, j, k, 0) = updated_var[0];
    bv(i, j, k, 1) = updated_var[1];
    bv(i, j, k, 2) = updated_var[2];
    bv(i, j, k, 3) = updated_var[3];
    bv(i, j, k, 4) = updated_var[4];
    if constexpr (mixture == MixtureModel::Air) {
      bv(i, j, k, 5) = updated_var[4] * mw_air / (updated_var[0] * R_u);
    } else {
      real mw = 0;
      for (int l = 0; l < n_spec; ++l) {
        mw += sv(i, j, k, l) / param->mw[l];
      }
      mw = 1 / mw;
      bv(i, j, k, 5) = updated_var[4] * mw / (updated_var[0] * R_u);
    }

    compute_cv_from_bv_1_point<mixture, turb_method>(zone, param, i, j, k);
  }
  __syncthreads();

  // Limit the turbulent values
  if constexpr (TurbMethod<turb_method>::label == TurbMethodLabel::SST) {
    // Record the computed values
    constexpr int n_turb = 2;
    real t_var[n_turb];
    t_var[0] = sv(i, j, k, n_spec);
    t_var[1] = sv(i, j, k, n_spec + 1);

    // Find the unphysical values and limit them
    unphysical = false;
    if (isnan(t_var[0]) || isnan(t_var[1]) || t_var[0] < 0 || t_var[1] < 0) {
      unphysical = true;
    }

    if (unphysical) {
      real updated_var[n_turb] = {};
      int kn{0};
      // Compute the sum of all "good" points surrounding the "bad" point
      for (int ka = -1; ka < 2; ++ka) {
        const int k1{k + ka};
        if (k1 < 0 || k1 >= mz) continue;
        for (int ja = -1; ja < 2; ++ja) {
          const int j1{j + ja};
          if (j1 < 0 || j1 >= my) continue;
          for (int ia = -1; ia < 2; ++ia) {
            const int i1{i + ia};
            if (i1 < 0 || i1 >= mx)continue;

            if (isnan(sv(i1, j1, k1, n_spec)) || isnan(sv(i1, j1, k1, 1 + n_spec)) || sv(i1, j1, k1, n_spec) < 0 ||
                sv(i1, j1, k1, n_spec + 1) < 0) {
              continue;
            }

            updated_var[0] += sv(i1, j1, k1, n_spec);
            updated_var[1] += sv(i1, j1, k1, 1 + n_spec);

            ++kn;
          }
        }
      }

      // Compute the average of the surrounding points
      if (kn > 0) {
        const real kn_inv{1.0 / kn};
        updated_var[0] *= kn_inv;
        updated_var[1] *= kn_inv;
      } else {
        // The surrounding points are all "bad"
        updated_var[0] = t_var[0] < 0 ? param->limit_flow.sv_inf[n_spec] : t_var[0];
        updated_var[1] = t_var[1] < 0 ? param->limit_flow.sv_inf[n_spec + 1] : t_var[1];
      }

      // Assign averaged values for the bad point
      if (sv(i, j, k, n_spec) < 0)
        sv(i, j, k, n_spec) = updated_var[0];
      if (sv(i, j, k, n_spec + 1) < 0)
        sv(i, j, k, n_spec + 1) = updated_var[1];

      zone->cv(i, j, k, n_spec) = bv(i, j, k, 0) * sv(i, j, k, n_spec);
      zone->cv(i, j, k, n_spec + 1) = bv(i, j, k, 0) * sv(i, j, k, n_spec + 1);
    }
  }

  // Limit the mixture fraction values
  if constexpr (mixture == MixtureModel::FL || mixture == MixtureModel::MixtureFraction) {
    // Record the computed values
    real z_var[2];
    const int i_fl{param->i_fl};
    z_var[0] = sv(i, j, k, i_fl);
    z_var[1] = sv(i, j, k, i_fl + 1);

    // Find the unphysical values and limit them
    unphysical = false;
    if (isnan(z_var[0]) || isnan(z_var[1]) || z_var[0] < 0 || z_var[1] < 0 || z_var[0] > 1 || z_var[1] > 0.25) {
      unphysical = true;
    }

    if (unphysical) {
      real updated_var[2] = {};
      int kn{0};
      // Compute the sum of all "good" points surrounding the "bad" point
      for (int ka = -1; ka < 2; ++ka) {
        const int k1{k + ka};
        if (k1 < 0 || k1 >= mz) continue;
        for (int ja = -1; ja < 2; ++ja) {
          const int j1{j + ja};
          if (j1 < 0 || j1 >= my) continue;
          for (int ia = -1; ia < 2; ++ia) {
            const int i1{i + ia};
            if (i1 < 0 || i1 >= mx)continue;

            if (isnan(sv(i1, j1, k1, i_fl)) || sv(i1, j1, k1, i_fl) < 0 || sv(i1, j1, k1, i_fl) > 1
                || isnan(sv(i1, j1, k1, 1 + i_fl)) || sv(i1, j1, k1, i_fl + 1) < 0 || sv(i1, j1, k1, i_fl + 1) > 0.25) {
              continue;
            }

            updated_var[0] += sv(i1, j1, k1, i_fl);
            updated_var[1] += sv(i1, j1, k1, 1 + i_fl);

            ++kn;
          }
        }
      }

      // Compute the average of the surrounding points
      if (kn > 0) {
        const real kn_inv{1.0 / kn};
        updated_var[0] *= kn_inv;
        updated_var[1] *= kn_inv;
      } else {
        // The surrounding points are all "bad"
        updated_var[0] = min(1.0, max(0.0, z_var[0]));
        updated_var[1] = min(0.25, max(0.0, z_var[1]));
      }

      // Assign averaged values for the bad point
      if (sv(i, j, k, i_fl) < 0 || sv(i, j, k, i_fl) > 1)
        sv(i, j, k, i_fl) = updated_var[0];
      if (sv(i, j, k, i_fl + 1) < 0 || sv(i, j, k, i_fl + 1) > 0.25)
        sv(i, j, k, i_fl + 1) = updated_var[1];

      zone->cv(i, j, k, i_fl) = bv(i, j, k, 0) * sv(i, j, k, i_fl);
      zone->cv(i, j, k, i_fl + 1) = bv(i, j, k, 0) * sv(i, j, k, i_fl + 1);
    }
  }
}
