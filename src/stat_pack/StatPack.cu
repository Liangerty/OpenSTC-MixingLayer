#include "StatPack.cuh"
#include "../Field.h"
#include "../DParameter.cuh"
#include "../Transport.cuh"

namespace cfd {
__device__ void
velFluc_scalarFluc_correlation::collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k,
                                        integer collect_idx) {
  auto &stat = zone->userCollectForStat;
  const auto &sv = zone->sv;
  stat(i, j, k, collect_idx) += zone->bv(i, j, k, 0) * zone->bv(i, j, k, 1) * sv(i, j, k, 0);
  stat(i, j, k, collect_idx + 1) += zone->bv(i, j, k, 0) * zone->bv(i, j, k, 2) * sv(i, j, k, 0);
  stat(i, j, k, collect_idx + 2) += zone->bv(i, j, k, 0) * zone->bv(i, j, k, 3) * sv(i, j, k, 0);
}

__device__ void
velFluc_scalarFluc_correlation::compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud,
                                        integer i, integer j, integer k, integer counter, integer stat_idx,
                                        integer collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collected_moments = zone->userCollectForStat;
  auto &mean = zone->mean_value;
  stat(i, j, k, stat_idx) =
      collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] / mean(i, j, k, 0) -
      mean(i, j, k, 6) * mean(i, j, k, 1);
  stat(i, j, k, stat_idx + 1) =
      collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] / mean(i, j, k, 0) -
      mean(i, j, k, 6) * mean(i, j, k, 2);
  stat(i, j, k, stat_idx + 2) =
      collected_moments(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2] / mean(i, j, k, 0) -
      mean(i, j, k, 6) * mean(i, j, k, 3);
}

__device__ void
resolved_tke::compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j,
                      integer k, integer counter, integer stat_idx, integer collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &rey_tensor = zone->reynolds_stress_tensor;
  auto k_res = max(0.5 * (rey_tensor(i, j, k, 0) + rey_tensor(i, j, k, 1) + rey_tensor(i, j, k, 2)), 0.0);
  stat(i, j, k, stat_idx) = k_res;
  stat(i, j, k, stat_idx + 1) = k_res / max((k_res + zone->sv(i, j, k, param->n_spec)), 1e-25);
}

__device__ void
Z_variance_and_dissipation_rate::collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k,
                                         integer collect_idx) {
  auto &stat = zone->userCollectForStat; // There may be mistakes!
  const auto &sv = zone->sv;
  constexpr integer z_idx{0};
  stat(i, j, k, collect_idx) += zone->bv(i, j, k, 0) * sv(i, j, k, z_idx) * sv(i, j, k, z_idx);
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
  const real z_x = 0.5 * (xi_x * (sv(i + 1, j, k, z_idx) - sv(i - 1, j, k, z_idx)) +
                          eta_x * (sv(i, j + 1, k, z_idx) - sv(i, j - 1, k, z_idx)) +
                          zeta_x * (sv(i, j, k + 1, z_idx) - sv(i, j, k - 1, z_idx)));
  const real z_y = 0.5 * (xi_y * (sv(i + 1, j, k, z_idx) - sv(i - 1, j, k, z_idx)) +
                          eta_y * (sv(i, j + 1, k, z_idx) - sv(i, j - 1, k, z_idx)) +
                          zeta_y * (sv(i, j, k + 1, z_idx) - sv(i, j, k - 1, z_idx)));
  const real z_z = 0.5 * (xi_z * (sv(i + 1, j, k, z_idx) - sv(i - 1, j, k, z_idx)) +
                          eta_z * (sv(i, j + 1, k, z_idx) - sv(i, j - 1, k, z_idx)) +
                          zeta_z * (sv(i, j, k + 1, z_idx) - sv(i, j, k - 1, z_idx)));
  const real rho_chi = 2.0 * zone->rho_D(i, j, k, z_idx) * (z_x * z_x + z_y * z_y + z_z * z_z);
  stat(i, j, k, collect_idx + 1) += rho_chi;
}

__device__ void
Z_variance_and_dissipation_rate::compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i,
                                         integer j, integer k, integer counter, integer stat_idx,
                                         integer collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collected_moments = zone->userCollectForStat;
  auto &mean = zone->mean_value;
  stat(i, j, k, stat_idx) = max(
      collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] / mean(i, j, k, 0) -
      mean(i, j, k, 6) * mean(i, j, k, 6), 0.0);
  stat(i, j, k, stat_idx + 1) =
      collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] / mean(i, j, k, 0);
}

__device__ void
turbulent_dissipation_rate::collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k,
                                    integer collect_idx) {
  auto &collect = zone->userCollectForStat;
  const auto &bv = zone->bv;

  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
  const real u_x = 0.5 * (xi_x * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_x * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_x * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_y = 0.5 * (xi_y * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_y * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_y * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_z = 0.5 * (xi_z * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_z * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_z * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));

  collect(i, j, k, collect_idx) += u_x;
  collect(i, j, k, collect_idx + 1) += u_y;
  collect(i, j, k, collect_idx + 2) += u_z;
  collect(i, j, k, collect_idx + 3) += v_x;
  collect(i, j, k, collect_idx + 4) += v_y;
  collect(i, j, k, collect_idx + 5) += v_z;
  collect(i, j, k, collect_idx + 6) += w_x;
  collect(i, j, k, collect_idx + 7) += w_y;
  collect(i, j, k, collect_idx + 8) += w_z;

  const real mu = zone->mul(i, j, k);
  const real sigma11 = mu * (4.0 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real sigma12 = mu * (u_y + v_x);
  const real sigma13 = mu * (u_z + w_x);
  const real sigma22 = mu * (4.0 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real sigma23 = mu * (v_z + w_y);
  const real sigma33 = mu * (4.0 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  collect(i, j, k, collect_idx + 9) += sigma11;
  collect(i, j, k, collect_idx + 10) += sigma12;
  collect(i, j, k, collect_idx + 11) += sigma13;
  collect(i, j, k, collect_idx + 12) += sigma22;
  collect(i, j, k, collect_idx + 13) += sigma23;
  collect(i, j, k, collect_idx + 14) += sigma33;

  collect(i, j, k, collect_idx + 15) += sigma11 * u_x + sigma12 * u_y + sigma13 * u_z
                                        + sigma12 * v_x + sigma22 * v_y + sigma23 * v_z
                                        + sigma13 * w_x + sigma23 * w_y + sigma33 * w_z;
}

__device__ void
turbulent_dissipation_rate::compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i,
                                    integer j, integer k, integer counter, integer stat_idx, integer collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &mean = zone->mean_value;

  auto rhoEps = collect(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15]
                - collect(i, j, k, collected_idx + 9) * collect(i, j, k, collected_idx) /
                  (counter_ud[collected_idx] * counter_ud[collected_idx + 9])
                - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 1) /
                  (counter_ud[collected_idx + 1] * counter_ud[collected_idx + 10])
                - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 2) /
                  (counter_ud[collected_idx + 2] * counter_ud[collected_idx + 11])
                - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 3) /
                  (counter_ud[collected_idx + 3] * counter_ud[collected_idx + 10])
                - collect(i, j, k, collected_idx + 12) * collect(i, j, k, collected_idx + 4) /
                  (counter_ud[collected_idx + 4] * counter_ud[collected_idx + 12])
                - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 5) /
                  (counter_ud[collected_idx + 5] * counter_ud[collected_idx + 13])
                - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 6) /
                  (counter_ud[collected_idx + 6] * counter_ud[collected_idx + 11])
                - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 7) /
                  (counter_ud[collected_idx + 7] * counter_ud[collected_idx + 13])
                - collect(i, j, k, collected_idx + 14) * collect(i, j, k, collected_idx + 8) /
                  (counter_ud[collected_idx + 8] * counter_ud[collected_idx + 14]);
  stat(i, j, k, stat_idx) = rhoEps / mean(i, j, k, 0);
  real nu{0};
  if (param->n_spec > 0) {
    real Y[MAX_SPEC_NUMBER], mw{0};
    for (int l = 0; l < param->n_spec; ++l) {
      Y[l] = mean(i, j, k, l + 6);
      mw += mean(i, j, k, l + 6) / param->mw[l];
    }
    mw = 1.0 / mw;
    nu = compute_viscosity(mean(i, j, k, 5), mw, Y, param) / mean(i, j, k, 0);
  } else {
    nu = Sutherland(mean(i, j, k, 5)) / mean(i, j, k, 0);
  }
  stat(i, j, k, stat_idx + 1) = pow(nu * nu * nu / stat(i, j, k, stat_idx), 0.25);
}

__device__ void turbulent_dissipation_rate::compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param,
                                                                     const integer *counter_ud, integer i, integer j,
                                                                     integer mz, integer counter, integer stat_idx,
                                                                     integer collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &firstOrderMoment = zone->firstOrderMoment;

  const real counter_inv{1.0 / counter};
  real add{0}, add_kolmogorov{0};
  for (int k = 0; k < mz; ++k) {
    const real density = firstOrderMoment(i, j, k, 0) * counter_inv;
    auto rhoEps = collect(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15]
                  - collect(i, j, k, collected_idx + 9) * collect(i, j, k, collected_idx) /
                    (counter_ud[collected_idx] * counter_ud[collected_idx + 9])
                  - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 1) /
                    (counter_ud[collected_idx + 1] * counter_ud[collected_idx + 10])
                  - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 2) /
                    (counter_ud[collected_idx + 2] * counter_ud[collected_idx + 11])
                  - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 3) /
                    (counter_ud[collected_idx + 3] * counter_ud[collected_idx + 10])
                  - collect(i, j, k, collected_idx + 12) * collect(i, j, k, collected_idx + 4) /
                    (counter_ud[collected_idx + 4] * counter_ud[collected_idx + 12])
                  - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 5) /
                    (counter_ud[collected_idx + 5] * counter_ud[collected_idx + 13])
                  - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 6) /
                    (counter_ud[collected_idx + 6] * counter_ud[collected_idx + 11])
                  - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 7) /
                    (counter_ud[collected_idx + 7] * counter_ud[collected_idx + 13])
                  - collect(i, j, k, collected_idx + 14) * collect(i, j, k, collected_idx + 8) /
                    (counter_ud[collected_idx + 8] * counter_ud[collected_idx + 14]);
    const auto Eps{rhoEps / density};
    add += Eps;

    real nu;
    auto T{firstOrderMoment(i, j, k, 5) * counter_inv};
    if (param->n_spec > 0) {
      real Y[MAX_SPEC_NUMBER], mw{0};
      for (int l = 0; l < param->n_spec; ++l) {
        Y[l] = firstOrderMoment(i, j, k, l + 6) * counter_inv;
        mw += Y[l] / param->mw[l];
      }
      mw = 1.0 / mw;
      nu = compute_viscosity(T, mw, Y, param) / density;
    } else {
      nu = Sutherland(T) / density;
    }
    add_kolmogorov += pow(nu * nu * nu / Eps, 0.25);
  }
  stat(i, j, 0, stat_idx) = add / mz;
  stat(i, j, 0, stat_idx + 1) = add_kolmogorov / mz;
}
}
