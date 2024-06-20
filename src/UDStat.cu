#include "UDStat.h"
#include "DParameter.cuh"
#include "Transport.cuh"

namespace cfd {
__device__ void
ThermRMS::collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx) {
  auto &collect = zone->userCollectForStat;
  const auto &bv = zone->bv;

  collect(i, j, k, collect_idx) += bv(i, j, k, 0) * bv(i, j, k, 0);
  collect(i, j, k, collect_idx + 1) += bv(i, j, k, 4) * bv(i, j, k, 4);
  collect(i, j, k, collect_idx + 2) += bv(i, j, k, 5) * bv(i, j, k, 5);
  collect(i, j, k, collect_idx + 3) += bv(i, j, k, 5);
}

__device__ void
ThermRMS::compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                                   int mz, int counter, int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &firstOrderMoment = zone->firstOrderMoment;

  const real counter_inv{1.0 / counter};
  real add_rho{0}, add_p{0}, add_T{0};
  for (int k = 0; k < mz; ++k) {
    const real density = firstOrderMoment(i, j, k, 0) * counter_inv;
    add_rho += sqrt(max(collect(i, j, k, collected_idx) / counter_ud[collected_idx] - density * density, 0.0));
    add_p += sqrt(max(collect(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                      firstOrderMoment(i, j, k, 4) * firstOrderMoment(i, j, k, 4) * counter_inv * counter_inv, 0.0));
    const real T_mean = collect(i, j, k, collected_idx + 3) / counter_ud[collected_idx + 3];
    const real TT_mean = collect(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2];
    const real T_favre = firstOrderMoment(i, j, k, 5) / firstOrderMoment(i, j, k, 0);
    add_T += sqrt(max(TT_mean - 2 * T_mean * T_favre + T_favre * T_favre, 0.0));
  }
  stat(i, j, 0, stat_idx) = add_rho / mz;
  stat(i, j, 0, stat_idx + 1) = add_p / mz;
  stat(i, j, 0, stat_idx + 2) = add_T / mz;
}

__device__ void
ThermRMS::compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k, int counter,
                  int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &mean = zone->mean_value;
  stat(i, j, k, stat_idx) = sqrt(
      max(collect(i, j, k, collected_idx) / counter_ud[collected_idx] - mean(i, j, k, 0) * mean(i, j, k, 0), 0.0));
  stat(i, j, k, stat_idx + 1) = sqrt(
      max(collect(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] - mean(i, j, k, 4) * mean(i, j, k, 4),
          0.0));
  const real T_mean = collect(i, j, k, collected_idx + 3) / counter_ud[collected_idx + 3];
  stat(i, j, k, stat_idx + 2) = sqrt(
      max(collect(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2] - 2 * T_mean * mean(i, j, k, 5) +
          mean(i, j, k, 5) * mean(i, j, k, 5), 0.0));
}

__device__ void
turbulent_dissipation_rate::collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx) {
  auto &collect = zone->userCollectForStat;
  const auto &bv = zone->bv;

  collect(i, j, k, collect_idx) += bv(i, j, k, 1);
  collect(i, j, k, collect_idx + 1) += bv(i, j, k, 2);
  collect(i, j, k, collect_idx + 2) += bv(i, j, k, 3);

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
//  collect(i, j, k, collect_idx) += u_x;
//  collect(i, j, k, collect_idx + 1) += u_y;
//  collect(i, j, k, collect_idx + 2) += u_z;
//  collect(i, j, k, collect_idx + 3) += v_x;
//  collect(i, j, k, collect_idx + 4) += v_y;
//  collect(i, j, k, collect_idx + 5) += v_z;
//  collect(i, j, k, collect_idx + 6) += w_x;
//  collect(i, j, k, collect_idx + 7) += w_y;
//  collect(i, j, k, collect_idx + 8) += w_z;

  const real mu = zone->mul(i, j, k);
  const real sigma11 = mu * (4.0 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real sigma12 = mu * (u_y + v_x);
  const real sigma13 = mu * (u_z + w_x);
  const real sigma22 = mu * (4.0 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real sigma23 = mu * (v_z + w_y);
  const real sigma33 = mu * (4.0 * w_z - 2 * u_x - 2 * v_y) / 3.0;
//  collect(i, j, k, collect_idx + 9) += sigma11;
//  collect(i, j, k, collect_idx + 10) += sigma12;
//  collect(i, j, k, collect_idx + 11) += sigma13;
//  collect(i, j, k, collect_idx + 12) += sigma22;
//  collect(i, j, k, collect_idx + 13) += sigma23;
//  collect(i, j, k, collect_idx + 14) += sigma33;
//
//  collect(i, j, k, collect_idx + 15) += sigma11 * u_x + sigma12 * u_y + sigma13 * u_z
//                                       + sigma12 * v_x + sigma22 * v_y + sigma23 * v_z
//                                       + sigma13 * w_x + sigma23 * w_y + sigma33 * w_z;

  collect(i, j, k, collect_idx + 3) += sigma11;
  collect(i, j, k, collect_idx + 4) += sigma12;
  collect(i, j, k, collect_idx + 5) += sigma13;
  collect(i, j, k, collect_idx + 6) += sigma22;
  collect(i, j, k, collect_idx + 7) += sigma23;
  collect(i, j, k, collect_idx + 8) += sigma33;

  collect(i, j, k, collect_idx + 9) += sigma11 * u_x + sigma12 * u_y + sigma13 * u_z
                                       + sigma12 * v_x + sigma22 * v_y + sigma23 * v_z
                                       + sigma13 * w_x + sigma23 * w_y + sigma33 * w_z;
}

__device__ void
turbulent_dissipation_rate::compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                                    int k, int counter, int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &mean = zone->mean_value;

  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  auto mx{zone->mx}, my{zone->my}, mz{zone->mz};
  real rhoEps{0};
  if (mz == 1) {
    // 2D case
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1) {
      const real u_x = 0.5 * (xi_x * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_x * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_y = 0.5 * (xi_y * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_y * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_z = 0.5 * (xi_z * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_z * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx))) /
                       counter_ud[collected_idx];
      const real v_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real w_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      rhoEps = collect(i, j, k, collected_idx + 9) / counter_ud[collected_idx + 9] -
               collect(i, j, k, collected_idx + 3) * u_x / counter_ud[collected_idx + 3]
               - collect(i, j, k, collected_idx + 4) * u_y / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 5) * u_z / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 4) * v_x / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 6) * v_y / counter_ud[collected_idx + 6]
               - collect(i, j, k, collected_idx + 7) * v_z / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 5) * w_x / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 7) * w_y / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 8) * w_z / counter_ud[collected_idx + 8];
    }
  } else {
    // 3D case
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
      const real u_x = 0.5 * (xi_x * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_x * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_x * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_y = 0.5 * (xi_y * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_y * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_y * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_z = 0.5 * (xi_z * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_z * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_z * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real v_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_x *
                 (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_y *
                 (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_z *
                 (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real w_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_x *
                 (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_y *
                 (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_z *
                 (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      rhoEps = collect(i, j, k, collected_idx + 9) / counter_ud[collected_idx + 9] -
               collect(i, j, k, collected_idx + 3) * u_x / counter_ud[collected_idx + 3]
               - collect(i, j, k, collected_idx + 4) * u_y / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 5) * u_z / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 4) * v_x / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 6) * v_y / counter_ud[collected_idx + 6]
               - collect(i, j, k, collected_idx + 7) * v_z / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 5) * w_x / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 7) * w_y / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 8) * w_z / counter_ud[collected_idx + 8];
    }
  }


//  auto rhoEps = collect(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15]
//                - collect(i, j, k, collected_idx + 9) * collect(i, j, k, collected_idx) /
//                  (counter_ud[collected_idx] * counter_ud[collected_idx + 9])
//                - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 1) /
//                  (counter_ud[collected_idx + 1] * counter_ud[collected_idx + 10])
//                - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 2) /
//                  (counter_ud[collected_idx + 2] * counter_ud[collected_idx + 11])
//                - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 3) /
//                  (counter_ud[collected_idx + 3] * counter_ud[collected_idx + 10])
//                - collect(i, j, k, collected_idx + 12) * collect(i, j, k, collected_idx + 4) /
//                  (counter_ud[collected_idx + 4] * counter_ud[collected_idx + 12])
//                - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 5) /
//                  (counter_ud[collected_idx + 5] * counter_ud[collected_idx + 13])
//                - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 6) /
//                  (counter_ud[collected_idx + 6] * counter_ud[collected_idx + 11])
//                - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 7) /
//                  (counter_ud[collected_idx + 7] * counter_ud[collected_idx + 13])
//                - collect(i, j, k, collected_idx + 14) * collect(i, j, k, collected_idx + 8) /
//                  (counter_ud[collected_idx + 8] * counter_ud[collected_idx + 14]);
  // In case of accumulated numerical error, the turbulent dissipation rate is limited to be positive.
  stat(i, j, k, stat_idx) = max(rhoEps / mean(i, j, k, 0), 1e-30);
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
  // The Kolmogorov scale \eta = (nu^3 / epsilon)^(1/4)
  stat(i, j, k, stat_idx + 1) = pow(nu * nu * nu / stat(i, j, k, stat_idx), 0.25);
  // The Kolmogorov time scale t_eta = \sqrt{nu / epsilon}
  stat(i, j, k, stat_idx + 2) = sqrt(nu / stat(i, j, k, stat_idx));
  auto &rey_tensor = zone->reynolds_stress_tensor;
  real tke = 0.5 * (rey_tensor(i, j, k, 0) + rey_tensor(i, j, k, 1) + rey_tensor(i, j, k, 2));
  // The turbulent time scale t_turb = tke / epsilon.
  stat(i, j, k, stat_idx + 3) = tke / stat(i, j, k, stat_idx);
}

__device__ void
turbulent_dissipation_rate::compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud,
                                                     int i, int j, int mz, int counter, int stat_idx,
                                                     int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &firstOrderMoment = zone->firstOrderMoment;

  const real counter_inv{1.0 / counter};
  real add{0};
  auto mx{zone->mx}, my{zone->my};
  if (mz == 1) {
    // 2D case
    const real density = firstOrderMoment(i, j, 0, 0) * counter_inv;
    const auto &m = zone->metric(i, j, 0);
    const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
    const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
    const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
    const real u_x = 0.5 * (xi_x * (collect(i + 1, j, 0, collected_idx) - collect(i - 1, j, 0, collected_idx)) +
                            eta_x * (collect(i, j + 1, 0, collected_idx) - collect(i, j - 1, 0, collected_idx))) /
                     counter_ud[collected_idx];
    const real u_y = 0.5 * (xi_y * (collect(i + 1, j, 0, collected_idx) - collect(i - 1, j, 0, collected_idx)) +
                            eta_y * (collect(i, j + 1, 0, collected_idx) - collect(i, j - 1, 0, collected_idx))) /
                     counter_ud[collected_idx];
    const real u_z = 0.5 * (xi_z * (collect(i + 1, j, 0, collected_idx) - collect(i - 1, j, 0, collected_idx)) +
                            eta_z * (collect(i, j + 1, 0, collected_idx) - collect(i, j - 1, 0, collected_idx))) /
                     counter_ud[collected_idx];
    const real v_x = 0.5 * (xi_x * (collect(i + 1, j, 0, collected_idx + 1) - collect(i - 1, j, 0, collected_idx + 1)) +
                            eta_x *
                            (collect(i, j + 1, 0, collected_idx + 1) - collect(i, j - 1, 0, collected_idx + 1))) /
                     counter_ud[collected_idx + 1];
    const real v_y = 0.5 * (xi_y * (collect(i + 1, j, 0, collected_idx + 1) - collect(i - 1, j, 0, collected_idx + 1)) +
                            eta_y *
                            (collect(i, j + 1, 0, collected_idx + 1) - collect(i, j - 1, 0, collected_idx + 1))) /
                     counter_ud[collected_idx + 1];
    const real v_z = 0.5 * (xi_z * (collect(i + 1, j, 0, collected_idx + 1) - collect(i - 1, j, 0, collected_idx + 1)) +
                            eta_z *
                            (collect(i, j + 1, 0, collected_idx + 1) - collect(i, j - 1, 0, collected_idx + 1))) /
                     counter_ud[collected_idx + 1];
    const real w_x = 0.5 * (xi_x * (collect(i + 1, j, 0, collected_idx + 2) - collect(i - 1, j, 0, collected_idx + 2)) +
                            eta_x *
                            (collect(i, j + 1, 0, collected_idx + 2) - collect(i, j - 1, 0, collected_idx + 2))) /
                     counter_ud[collected_idx + 2];
    const real w_y = 0.5 * (xi_y * (collect(i + 1, j, 0, collected_idx + 2) - collect(i - 1, j, 0, collected_idx + 2)) +
                            eta_y *
                            (collect(i, j + 1, 0, collected_idx + 2) - collect(i, j - 1, 0, collected_idx + 2))) /
                     counter_ud[collected_idx + 2];
    const real w_z = 0.5 * (xi_z * (collect(i + 1, j, 0, collected_idx + 2) - collect(i - 1, j, 0, collected_idx + 2)) +
                            eta_z *
                            (collect(i, j + 1, 0, collected_idx + 2) - collect(i, j - 1, 0, collected_idx + 2))) /
                     counter_ud[collected_idx + 2];
    real rhoEps = collect(i, j, 0, collected_idx + 9) / counter_ud[collected_idx + 9] -
                  collect(i, j, 0, collected_idx + 3) * u_x / counter_ud[collected_idx + 3]
                  - collect(i, j, 0, collected_idx + 4) * u_y / counter_ud[collected_idx + 4] -
                  collect(i, j, 0, collected_idx + 5) * u_z / counter_ud[collected_idx + 5]
                  - collect(i, j, 0, collected_idx + 4) * v_x / counter_ud[collected_idx + 4] -
                  collect(i, j, 0, collected_idx + 6) * v_y / counter_ud[collected_idx + 6]
                  - collect(i, j, 0, collected_idx + 7) * v_z / counter_ud[collected_idx + 7] -
                  collect(i, j, 0, collected_idx + 5) * w_x / counter_ud[collected_idx + 5]
                  - collect(i, j, 0, collected_idx + 7) * w_y / counter_ud[collected_idx + 7] -
                  collect(i, j, 0, collected_idx + 8) * w_z / counter_ud[collected_idx + 8];
    stat(i, j, 0, stat_idx) = max(rhoEps / density, 1e-30);
    auto &mean = zone->mean_value;
    real nu{0};
    if (param->n_spec > 0) {
      real Y[MAX_SPEC_NUMBER], mw{0};
      for (int l = 0; l < param->n_spec; ++l) {
        Y[l] = mean(i, j, 0, l + 6);
        mw += mean(i, j, 0, l + 6) / param->mw[l];
      }
      mw = 1.0 / mw;
      nu = compute_viscosity(mean(i, j, 0, 5), mw, Y, param) / mean(i, j, 0, 0);
    } else {
      nu = Sutherland(mean(i, j, 0, 5)) / mean(i, j, 0, 0);
    }
    // The Kolmogorov scale \eta = (nu^3 / epsilon)^(1/4)
    stat(i, j, 0, stat_idx + 1) = pow(nu * nu * nu / stat(i, j, 0, stat_idx), 0.25);
    // The Kolmogorov time scale t_eta = \sqrt{nu / epsilon}
    stat(i, j, 0, stat_idx + 2) = sqrt(nu / stat(i, j, 0, stat_idx));
    auto &rey_tensor = zone->reynolds_stress_tensor;
    real tke = 0.5 * (rey_tensor(i, j, 0, 0) + rey_tensor(i, j, 0, 1) + rey_tensor(i, j, 0, 2));
    // The turbulent time scale t_turb = tke / epsilon.
    stat(i, j, 0, stat_idx + 3) = tke / stat(i, j, 0, stat_idx);
    return;
  }
  int useful_counter{0};
  for (int k = 0; k < mz; ++k) {
    const real density = firstOrderMoment(i, j, k, 0) * counter_inv;
    real rhoEps{0};
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
      const auto &m = zone->metric(i, j, k);
      const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
      const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
      const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
      const real u_x = 0.5 * (xi_x * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_x * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_x * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_y = 0.5 * (xi_y * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_y * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_y * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real u_z = 0.5 * (xi_z * (collect(i + 1, j, k, collected_idx) - collect(i - 1, j, k, collected_idx)) +
                              eta_z * (collect(i, j + 1, k, collected_idx) - collect(i, j - 1, k, collected_idx)) +
                              zeta_z * (collect(i, j, k + 1, collected_idx) - collect(i, j, k - 1, collected_idx))) /
                       counter_ud[collected_idx];
      const real v_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_x * (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_y * (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real v_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 1) - collect(i - 1, j, k, collected_idx + 1)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 1) - collect(i, j - 1, k, collected_idx + 1)) +
                 zeta_z * (collect(i, j, k + 1, collected_idx + 1) - collect(i, j, k - 1, collected_idx + 1))) /
          counter_ud[collected_idx + 1];
      const real w_x =
          0.5 * (xi_x * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_x * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_x * (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_y =
          0.5 * (xi_y * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_y * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_y * (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      const real w_z =
          0.5 * (xi_z * (collect(i + 1, j, k, collected_idx + 2) - collect(i - 1, j, k, collected_idx + 2)) +
                 eta_z * (collect(i, j + 1, k, collected_idx + 2) - collect(i, j - 1, k, collected_idx + 2)) +
                 zeta_z * (collect(i, j, k + 1, collected_idx + 2) - collect(i, j, k - 1, collected_idx + 2))) /
          counter_ud[collected_idx + 2];
      rhoEps = collect(i, j, k, collected_idx + 9) / counter_ud[collected_idx + 9] -
               collect(i, j, k, collected_idx + 3) * u_x / counter_ud[collected_idx + 3]
               - collect(i, j, k, collected_idx + 4) * u_y / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 5) * u_z / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 4) * v_x / counter_ud[collected_idx + 4] -
               collect(i, j, k, collected_idx + 6) * v_y / counter_ud[collected_idx + 6]
               - collect(i, j, k, collected_idx + 7) * v_z / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 5) * w_x / counter_ud[collected_idx + 5]
               - collect(i, j, k, collected_idx + 7) * w_y / counter_ud[collected_idx + 7] -
               collect(i, j, k, collected_idx + 8) * w_z / counter_ud[collected_idx + 8];
    }
//    auto rhoEps = collect(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15]
//                  - collect(i, j, k, collected_idx + 9) * collect(i, j, k, collected_idx) /
//                    (counter_ud[collected_idx] * counter_ud[collected_idx + 9])
//                  - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 1) /
//                    (counter_ud[collected_idx + 1] * counter_ud[collected_idx + 10])
//                  - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 2) /
//                    (counter_ud[collected_idx + 2] * counter_ud[collected_idx + 11])
//                  - collect(i, j, k, collected_idx + 10) * collect(i, j, k, collected_idx + 3) /
//                    (counter_ud[collected_idx + 3] * counter_ud[collected_idx + 10])
//                  - collect(i, j, k, collected_idx + 12) * collect(i, j, k, collected_idx + 4) /
//                    (counter_ud[collected_idx + 4] * counter_ud[collected_idx + 12])
//                  - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 5) /
//                    (counter_ud[collected_idx + 5] * counter_ud[collected_idx + 13])
//                  - collect(i, j, k, collected_idx + 11) * collect(i, j, k, collected_idx + 6) /
//                    (counter_ud[collected_idx + 6] * counter_ud[collected_idx + 11])
//                  - collect(i, j, k, collected_idx + 13) * collect(i, j, k, collected_idx + 7) /
//                    (counter_ud[collected_idx + 7] * counter_ud[collected_idx + 13])
//                  - collect(i, j, k, collected_idx + 14) * collect(i, j, k, collected_idx + 8) /
//                    (counter_ud[collected_idx + 8] * counter_ud[collected_idx + 14]);
    if (auto Eps = rhoEps / density;Eps > 1e-30) {
      add += Eps;
      ++useful_counter;
    }
  }
  stat(i, j, 0, stat_idx) = max(add / useful_counter, 1e-30);
  real nu;
  auto &mean = zone->mean_value;
  if (param->n_spec > 0) {
    real Y[MAX_SPEC_NUMBER], mw{0};
    for (int l = 0; l < param->n_spec; ++l) {
      Y[l] = mean(i, j, 0, l + 6);
      mw += mean(i, j, 0, l + 6) / param->mw[l];
    }
    mw = 1.0 / mw;
    nu = compute_viscosity(mean(i, j, 0, 5), mw, Y, param) / mean(i, j, 0, 0);
  } else {
    nu = Sutherland(mean(i, j, 0, 5)) / mean(i, j, 0, 0);
  }
  stat(i, j, 0, stat_idx + 1) = pow(nu * nu * nu / stat(i, j, 0, stat_idx), 0.25);
  stat(i, j, 0, stat_idx + 2) = min(sqrt(nu / stat(i, j, 0, stat_idx)), 1.0);
  auto &rey_tensor = zone->reynolds_stress_tensor;
  stat(i, j, 0, stat_idx + 3) =
      min(0.5 * (rey_tensor(i, j, 0, 0) + rey_tensor(i, j, 0, 1) + rey_tensor(i, j, 0, 2)) / stat(i, j, 0, stat_idx),
          1.0);
}

__device__ void
H2AirMixingLayer::collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx) {
  auto &collect = zone->userCollectForStat;
  const auto &sv = zone->sv;
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // H2
  collect(i, j, k, collect_idx) += zone->bv(i, j, k, 0) * sv(i, j, k, 0) * sv(i, j, k, 0);
  real zx = 0.5 * (xi_x * (sv(i + 1, j, k, 0) - sv(i - 1, j, k, 0)) +
                   eta_x * (sv(i, j + 1, k, 0) - sv(i, j - 1, k, 0)) +
                   zeta_x * (sv(i, j, k + 1, 0) - sv(i, j, k - 1, 0)));
  real zy = 0.5 * (xi_y * (sv(i + 1, j, k, 0) - sv(i - 1, j, k, 0)) +
                   eta_y * (sv(i, j + 1, k, 0) - sv(i, j - 1, k, 0)) +
                   zeta_y * (sv(i, j, k + 1, 0) - sv(i, j, k - 1, 0)));
  real zz = 0.5 * (xi_z * (sv(i + 1, j, k, 0) - sv(i - 1, j, k, 0)) +
                   eta_z * (sv(i, j + 1, k, 0) - sv(i, j - 1, k, 0)) +
                   zeta_z * (sv(i, j, k + 1, 0) - sv(i, j, k - 1, 0)));
  auto rhoD = zone->rho_D(i, j, k, 0);
  // Rho*D*GradH2*GradH2
  collect(i, j, k, collect_idx + 1) += rhoD * (zx * zx + zy * zy + zz * zz);
  // Rho*D*H2x
  collect(i, j, k, collect_idx + 2) += rhoD * zx;
  // Rho*D*H2y
  collect(i, j, k, collect_idx + 3) += rhoD * zy;
  // Rho*D*H2z
  collect(i, j, k, collect_idx + 4) += rhoD * zz;
  collect(i, j, k, collect_idx + 5) += rhoD;

  // O2
  collect(i, j, k, collect_idx + 6) += zone->bv(i, j, k, 0) * sv(i, j, k, 1) * sv(i, j, k, 1);
  zx = 0.5 * (xi_x * (sv(i + 1, j, k, 1) - sv(i - 1, j, k, 1)) +
              eta_x * (sv(i, j + 1, k, 1) - sv(i, j - 1, k, 1)) +
              zeta_x * (sv(i, j, k + 1, 1) - sv(i, j, k - 1, 1)));
  zy = 0.5 * (xi_y * (sv(i + 1, j, k, 1) - sv(i - 1, j, k, 1)) +
              eta_y * (sv(i, j + 1, k, 1) - sv(i, j - 1, k, 1)) +
              zeta_y * (sv(i, j, k + 1, 1) - sv(i, j, k - 1, 1)));
  zz = 0.5 * (xi_z * (sv(i + 1, j, k, 1) - sv(i - 1, j, k, 1)) +
              eta_z * (sv(i, j + 1, k, 1) - sv(i, j - 1, k, 1)) +
              zeta_z * (sv(i, j, k + 1, 1) - sv(i, j, k - 1, 1)));
  rhoD = zone->rho_D(i, j, k, 1);
  // Rho*D*GradO2*GradO2
  collect(i, j, k, collect_idx + 7) += rhoD * (zx * zx + zy * zy + zz * zz);
  // Rho*D*O2x
  collect(i, j, k, collect_idx + 8) += rhoD * zx;
  // Rho*D*O2y
  collect(i, j, k, collect_idx + 9) += rhoD * zy;
  // Rho*D*O2z
  collect(i, j, k, collect_idx + 10) += rhoD * zz;
  collect(i, j, k, collect_idx + 11) += rhoD;

  // N2
  collect(i, j, k, collect_idx + 12) += zone->bv(i, j, k, 0) * sv(i, j, k, 2) * sv(i, j, k, 2);
  zx = 0.5 * (xi_x * (sv(i + 1, j, k, 2) - sv(i - 1, j, k, 2)) +
              eta_x * (sv(i, j + 1, k, 2) - sv(i, j - 1, k, 2)) +
              zeta_x * (sv(i, j, k + 1, 2) - sv(i, j, k - 1, 2)));
  zy = 0.5 * (xi_y * (sv(i + 1, j, k, 2) - sv(i - 1, j, k, 2)) +
              eta_y * (sv(i, j + 1, k, 2) - sv(i, j - 1, k, 2)) +
              zeta_y * (sv(i, j, k + 1, 2) - sv(i, j, k - 1, 2)));
  zz = 0.5 * (xi_z * (sv(i + 1, j, k, 2) - sv(i - 1, j, k, 2)) +
              eta_z * (sv(i, j + 1, k, 2) - sv(i, j - 1, k, 2)) +
              zeta_z * (sv(i, j, k + 1, 2) - sv(i, j, k - 1, 2)));
  rhoD = zone->rho_D(i, j, k, 2);
  // Rho*D*GradN2*GradN2
  collect(i, j, k, collect_idx + 13) += rhoD * (zx * zx + zy * zy + zz * zz);
  // Rho*D*N2x
  collect(i, j, k, collect_idx + 14) += rhoD * zx;
  // Rho*D*N2y
  collect(i, j, k, collect_idx + 15) += rhoD * zy;
  // Rho*D*N2z
  collect(i, j, k, collect_idx + 16) += rhoD * zz;
  collect(i, j, k, collect_idx + 17) += rhoD;

  auto &bv = zone->bv;
  auto rho = zone->bv(i, j, k, 0), u = zone->bv(i, j, k, 1), v = zone->bv(i, j, k, 2), w = zone->bv(i, j, k, 3);
  // flux of H2
  real h2 = sv(i, j, k, 0);
  collect(i, j, k, collect_idx + 18) += rho * h2 * u;
  collect(i, j, k, collect_idx + 19) += rho * h2 * v;
  collect(i, j, k, collect_idx + 20) += rho * h2 * w;
  // flux of O2
  real o2 = sv(i, j, k, 1);
  collect(i, j, k, collect_idx + 21) += rho * o2 * u;
  collect(i, j, k, collect_idx + 22) += rho * o2 * v;
  collect(i, j, k, collect_idx + 23) += rho * o2 * w;
  // flux of N2
  real n2 = sv(i, j, k, 2);
  collect(i, j, k, collect_idx + 24) += rho * n2 * u;
  collect(i, j, k, collect_idx + 25) += rho * n2 * v;
  collect(i, j, k, collect_idx + 26) += rho * n2 * w;
  // flux of u H2 H2
  collect(i, j, k, collect_idx + 27) += rho * h2 * h2 * u;
  collect(i, j, k, collect_idx + 28) += rho * h2 * h2 * v;
  collect(i, j, k, collect_idx + 29) += rho * h2 * h2 * w;
  // flux of u O2 O2
  collect(i, j, k, collect_idx + 30) += rho * o2 * o2 * u;
  collect(i, j, k, collect_idx + 31) += rho * o2 * o2 * v;
  collect(i, j, k, collect_idx + 32) += rho * o2 * o2 * w;
  // flux of u N2 N2
  collect(i, j, k, collect_idx + 33) += rho * n2 * n2 * u;
  collect(i, j, k, collect_idx + 34) += rho * n2 * n2 * v;
  collect(i, j, k, collect_idx + 35) += rho * n2 * n2 * w;

  // laminar Schmidt number
  collect(i, j, k, collect_idx + 36) += zone->mul(i, j, k) / zone->rho_D(i, j, k, 0);
  collect(i, j, k, collect_idx + 37) += zone->mul(i, j, k) / zone->rho_D(i, j, k, 1);
  collect(i, j, k, collect_idx + 38) += zone->mul(i, j, k) / zone->rho_D(i, j, k, 2);
}

__device__ void
H2AirMixingLayer::compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
                          int counter, int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collect = zone->userCollectForStat;
  auto &mean = zone->mean_value;
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // nut
  auto &rey_tensor = zone->reynolds_stress_tensor;
  const real tke = 0.5 * (rey_tensor(i, j, k, 0) + rey_tensor(i, j, k, 1) + rey_tensor(i, j, k, 2));
  real Tau11 = rey_tensor(i, j, k, 0) - 2.0 / 3 * tke;
  real Tau22 = rey_tensor(i, j, k, 1) - 2.0 / 3 * tke;
  real Tau33 = rey_tensor(i, j, k, 2) - 2.0 / 3 * tke;
  real Tau12 = rey_tensor(i, j, k, 3);
  real Tau13 = rey_tensor(i, j, k, 4);
  real Tau23 = rey_tensor(i, j, k, 5);
  real ux = 0.5 * (xi_x * (mean(i + 1, j, k, 1) - mean(i - 1, j, k, 1)) +
                   eta_x * (mean(i, j + 1, k, 1) - mean(i, j - 1, k, 1)) +
                   zeta_x * (mean(i, j, k + 1, 1) - mean(i, j, k - 1, 1)));
  real uy = 0.5 * (xi_y * (mean(i + 1, j, k, 1) - mean(i - 1, j, k, 1)) +
                   eta_y * (mean(i, j + 1, k, 1) - mean(i, j - 1, k, 1)) +
                   zeta_y * (mean(i, j, k + 1, 1) - mean(i, j, k - 1, 1)));
  real uz = 0.5 * (xi_z * (mean(i + 1, j, k, 1) - mean(i - 1, j, k, 1)) +
                   eta_z * (mean(i, j + 1, k, 1) - mean(i, j - 1, k, 1)) +
                   zeta_z * (mean(i, j, k + 1, 1) - mean(i, j, k - 1, 1)));
  real vx = 0.5 * (xi_x * (mean(i + 1, j, k, 2) - mean(i - 1, j, k, 2)) +
                   eta_x * (mean(i, j + 1, k, 2) - mean(i, j - 1, k, 2)) +
                   zeta_x * (mean(i, j, k + 1, 2) - mean(i, j, k - 1, 2)));
  real vy = 0.5 * (xi_y * (mean(i + 1, j, k, 2) - mean(i - 1, j, k, 2)) +
                   eta_y * (mean(i, j + 1, k, 2) - mean(i, j - 1, k, 2)) +
                   zeta_y * (mean(i, j, k + 1, 2) - mean(i, j, k - 1, 2)));
  real vz = 0.5 * (xi_z * (mean(i + 1, j, k, 2) - mean(i - 1, j, k, 2)) +
                   eta_z * (mean(i, j + 1, k, 2) - mean(i, j - 1, k, 2)) +
                   zeta_z * (mean(i, j, k + 1, 2) - mean(i, j, k - 1, 2)));
  real wx = 0.5 * (xi_x * (mean(i + 1, j, k, 3) - mean(i - 1, j, k, 3)) +
                   eta_x * (mean(i, j + 1, k, 3) - mean(i, j - 1, k, 3)) +
                   zeta_x * (mean(i, j, k + 1, 3) - mean(i, j, k - 1, 3)));
  real wy = 0.5 * (xi_y * (mean(i + 1, j, k, 3) - mean(i - 1, j, k, 3)) +
                   eta_y * (mean(i, j + 1, k, 3) - mean(i, j - 1, k, 3)) +
                   zeta_y * (mean(i, j, k + 1, 3) - mean(i, j, k - 1, 3)));
  real wz = 0.5 * (xi_z * (mean(i + 1, j, k, 3) - mean(i - 1, j, k, 3)) +
                   eta_z * (mean(i, j + 1, k, 3) - mean(i, j - 1, k, 3)) +
                   zeta_z * (mean(i, j, k + 1, 3) - mean(i, j, k - 1, 3)));
  real divV = ux + vy + wz;
  stat(i, j, k, stat_idx) = divV;
  real S11 = 2 * ux - 2.0 / 3 * divV;
  real S22 = 2 * vy - 2.0 / 3 * divV;
  real S33 = 2 * wz - 2.0 / 3 * divV;
  real S12 = uy + vx;
  real S13 = uz + wx;
  real S23 = vz + wy;
  stat(i, j, k, 1 + stat_idx) = S11 > 1e-25 ? -Tau11 / S11 : 0;
  stat(i, j, k, 2 + stat_idx) = S22 > 1e-25 ? -Tau22 / S22 : 0;
  stat(i, j, k, 3 + stat_idx) = S33 > 1e-25 ? -Tau33 / S33 : 0;
  stat(i, j, k, 4 + stat_idx) = S12 > 1e-25 ? -Tau12 / S12 : 0;
  stat(i, j, k, 5 + stat_idx) = S13 > 1e-25 ? -Tau13 / S13 : 0;
  stat(i, j, k, 6 + stat_idx) = S23 > 1e-25 ? -Tau23 / S23 : 0;
  real SiiSii = S11 * S11 + S22 * S22 + S33 * S33 + 2 * S12 * S12 + 2 * S13 * S13 + 2 * S23 * S23;
  real nut =
      SiiSii > 1e-25 ? -(Tau11 * S11 + Tau22 * S22 + Tau33 * S33 + Tau12 * S12 + Tau13 * S13 + Tau23 * S23) / SiiSii
                     : 0;
  stat(i, j, k, 7 + stat_idx) = nut;


  const real rho_inv = 1.0 / mean(i, j, k, 0);
  // H2
  // {H2''H2''}
  const real h2 = mean(i, j, k, 6);
  stat(i, j, k, stat_idx + 8) = max(
      collect(i, j, k, collected_idx) / counter_ud[collected_idx] * rho_inv - h2 * h2, 1e-30);
  // chi=2/<rho>*[<rho*D*gradH2*gradH2>-2<rho*D*H2x>*{H2}_x-2<rho*D*H2y>*{H2}_y-2<rho*D*H2z>*{H2}_z+<rho*D>*grad{H2}*grad{H2}]
  real zx = 0.5 * (xi_x * (mean(i + 1, j, k, 6) - mean(i - 1, j, k, 6)) +
                   eta_x * (mean(i, j + 1, k, 6) - mean(i, j - 1, k, 6)) +
                   zeta_x * (mean(i, j, k + 1, 6) - mean(i, j, k - 1, 6)));
  real zy = 0.5 * (xi_y * (mean(i + 1, j, k, 6) - mean(i - 1, j, k, 6)) +
                   eta_y * (mean(i, j + 1, k, 6) - mean(i, j - 1, k, 6)) +
                   zeta_y * (mean(i, j, k + 1, 6) - mean(i, j, k - 1, 6)));
  real zz = 0.5 * (xi_z * (mean(i + 1, j, k, 6) - mean(i - 1, j, k, 6)) +
                   eta_z * (mean(i, j + 1, k, 6) - mean(i, j - 1, k, 6)) +
                   zeta_z * (mean(i, j, k + 1, 6) - mean(i, j, k - 1, 6)));
  real zizi = zx * zx + zy * zy + zz * zz;
  const real two_rho_inv = 2.0 / mean(i, j, k, 0);
  real chi = two_rho_inv * (collect(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                            2 * collect(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2] * zx -
                            2 * collect(i, j, k, collected_idx + 3) / counter_ud[collected_idx + 3] * zy -
                            2 * collect(i, j, k, collected_idx + 4) / counter_ud[collected_idx + 4] * zz +
                            collect(i, j, k, collected_idx + 5) / counter_ud[collected_idx + 5] * zizi);
  stat(i, j, k, stat_idx + 9) = max(chi, 1e-30);
  stat(i, j, k, stat_idx + 10) = min(stat(i, j, k, stat_idx + 8) / max(chi, 1e-30), 1.0);
  // {u''H2''}, {v''H2''}, {w''H2''}
  real upZp = collect(i, j, k, collected_idx + 6) / counter_ud[collected_idx + 6] * rho_inv - mean(i, j, k, 1) * h2;
  real vpZp = collect(i, j, k, collected_idx + 7) / counter_ud[collected_idx + 7] * rho_inv - mean(i, j, k, 2) * h2;
  real wpZp = collect(i, j, k, collected_idx + 8) / counter_ud[collected_idx + 8] * rho_inv - mean(i, j, k, 3) * h2;
  stat(i, j, k, stat_idx + 11) = upZp;
  stat(i, j, k, stat_idx + 12) = vpZp;
  stat(i, j, k, stat_idx + 13) = wpZp;
  // Dt_H2,x; Dt_H2,y; Dt_H2,z. And corresponding Sct.
  if (abs(zx) > 1e-25 && abs(upZp) > 1e-25) {
    stat(i, j, k, stat_idx + 14) = -upZp / zx;
    stat(i, j, k, stat_idx + 19) = -nut * zx / upZp;
  }
  if (abs(zy) > 1e-25 && abs(vpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 15) = -vpZp / zy;
    stat(i, j, k, stat_idx + 20) = -nut * zy / vpZp;
  }
  if (abs(zz) > 1e-25 && abs(wpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 16) = -wpZp / zz;
    stat(i, j, k, stat_idx + 21) = -nut * zz / wpZp;
  }
  // Dt_H2
  if (real isoPart = upZp * zx + vpZp * zy + wpZp * zz; zizi > 1e-25 && abs(isoPart) > 1e-25) {
    stat(i, j, k, stat_idx + 17) = -isoPart / zizi;
    stat(i, j, k, stat_idx + 22) = -nut * zizi / isoPart;
  }
  // Sc_H2_laminar
  stat(i, j, k, stat_idx + 18) = collect(i, j, k, collected_idx + 12) / counter_ud[collected_idx + 12];
  // Z''2 flux
  stat(i, j, k, stat_idx + 23) = (collect(i, j, k, collected_idx + 9) / counter_ud[collected_idx + 9] -
                                  2 * collect(i, j, k, collected_idx + 6) / counter_ud[collected_idx + 6] * h2 -
                                  collect(i, j, k, collected_idx) / counter_ud[collected_idx] * mean(i, j, k, 1)) *
                                 rho_inv + 2 * h2 * h2 * mean(i, j, k, 1);
  stat(i, j, k, stat_idx + 24) = (collect(i, j, k, collected_idx + 10) / counter_ud[collected_idx + 10] -
                                  2 * collect(i, j, k, collected_idx + 7) / counter_ud[collected_idx + 7] * h2 -
                                  collect(i, j, k, collected_idx) / counter_ud[collected_idx] * mean(i, j, k, 2)) *
                                 rho_inv + 2 * h2 * h2 * mean(i, j, k, 2);
  stat(i, j, k, stat_idx + 25) = (collect(i, j, k, collected_idx + 11) / counter_ud[collected_idx + 11] -
                                  2 * collect(i, j, k, collected_idx + 8) / counter_ud[collected_idx + 8] * h2 -
                                  collect(i, j, k, collected_idx) / counter_ud[collected_idx] * mean(i, j, k, 3)) *
                                 rho_inv + 2 * h2 * h2 * mean(i, j, k, 3);

  // O2
  const real o2 = mean(i, j, k, 7);
  // {O2''O2''}
  stat(i, j, k, stat_idx + 30) = max(
      collect(i, j, k, collected_idx + 13) / counter_ud[collected_idx + 13] * rho_inv - o2 * o2, 1e-30);
  // chi=2/<rho>*[<rho*D*gradO2*gradO2>-2<rho*D*O2x>*{O2}_x-2<rho*D*O2y>*{O2}_y-2<rho*D*O2z>*{O2}_z+<rho*D>*grad{O2}*grad{O2}]
  zx = 0.5 * (xi_x * (mean(i + 1, j, k, 7) - mean(i - 1, j, k, 7)) +
              eta_x * (mean(i, j + 1, k, 7) - mean(i, j - 1, k, 7)) +
              zeta_x * (mean(i, j, k + 1, 7) - mean(i, j, k - 1, 7)));
  zy = 0.5 * (xi_y * (mean(i + 1, j, k, 7) - mean(i - 1, j, k, 7)) +
              eta_y * (mean(i, j + 1, k, 7) - mean(i, j - 1, k, 7)) +
              zeta_y * (mean(i, j, k + 1, 7) - mean(i, j, k - 1, 7)));
  zz = 0.5 * (xi_z * (mean(i + 1, j, k, 7) - mean(i - 1, j, k, 7)) +
              eta_z * (mean(i, j + 1, k, 7) - mean(i, j - 1, k, 7)) +
              zeta_z * (mean(i, j, k + 1, 7) - mean(i, j, k - 1, 7)));
  zizi = zx * zx + zy * zy + zz * zz;
  chi = two_rho_inv * (collect(i, j, k, collected_idx + 14) / counter_ud[collected_idx + 14] -
                       2 * collect(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15] * zx -
                       2 * collect(i, j, k, collected_idx + 16) / counter_ud[collected_idx + 16] * zy -
                       2 * collect(i, j, k, collected_idx + 17) / counter_ud[collected_idx + 17] * zz +
                       collect(i, j, k, collected_idx + 18) / counter_ud[collected_idx + 18] * zizi);
  stat(i, j, k, stat_idx + 31) = max(chi, 1e-30);
  stat(i, j, k, stat_idx + 32) = min(stat(i, j, k, stat_idx + 30) / max(chi, 1e-30), 1.0);
  // {u''O2''}, {v''O2''}, {w''O2''}
  upZp = collect(i, j, k, collected_idx + 19) / counter_ud[collected_idx + 19] * rho_inv - mean(i, j, k, 1) * o2;
  vpZp = collect(i, j, k, collected_idx + 20) / counter_ud[collected_idx + 20] * rho_inv - mean(i, j, k, 2) * o2;
  wpZp = collect(i, j, k, collected_idx + 21) / counter_ud[collected_idx + 21] * rho_inv - mean(i, j, k, 3) * o2;
  stat(i, j, k, stat_idx + 33) = upZp;
  stat(i, j, k, stat_idx + 34) = vpZp;
  stat(i, j, k, stat_idx + 35) = wpZp;
  // Dt_O2,x; Dt_O2,y; Dt_O2,z. And corresponding Sct.
  if (abs(zx) > 1e-25 && abs(upZp) > 1e-25) {
    stat(i, j, k, stat_idx + 36) = -upZp / zx;
    stat(i, j, k, stat_idx + 41) = -nut * zx / upZp;
  }
  if (abs(zy) > 1e-25 && abs(vpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 37) = -vpZp / zy;
    stat(i, j, k, stat_idx + 42) = -nut * zy / vpZp;
  }
  if (abs(zz) > 1e-25 && abs(wpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 38) = -wpZp / zz;
    stat(i, j, k, stat_idx + 43) = -nut * zz / wpZp;
  }
  // Dt_O2
  if (real isoPart = upZp * zx + vpZp * zy + wpZp * zz; zizi > 1e-25 && abs(isoPart) > 1e-25) {
    stat(i, j, k, stat_idx + 39) = -isoPart / zizi;
    stat(i, j, k, stat_idx + 44) = -nut * zizi / isoPart;
  }
  // Sc_O2_laminar
  stat(i, j, k, stat_idx + 40) = collect(i, j, k, collected_idx + 25) / counter_ud[collected_idx + 25];
  // Z''2 flux
  stat(i, j, k, stat_idx + 45) = (collect(i, j, k, collected_idx + 22) / counter_ud[collected_idx + 22] -
                                  2 * collect(i, j, k, collected_idx + 19) / counter_ud[collected_idx + 19] * o2 -
                                  collect(i, j, k, collected_idx + 13) / counter_ud[collected_idx + 13] *
                                  mean(i, j, k, 1)) * rho_inv + 2 * o2 * o2 * mean(i, j, k, 1);
  stat(i, j, k, stat_idx + 46) = (collect(i, j, k, collected_idx + 23) / counter_ud[collected_idx + 23] -
                                  2 * collect(i, j, k, collected_idx + 20) / counter_ud[collected_idx + 20] * o2 -
                                  collect(i, j, k, collected_idx + 13) / counter_ud[collected_idx + 13] *
                                  mean(i, j, k, 2)) * rho_inv + 2 * o2 * o2 * mean(i, j, k, 2);
  stat(i, j, k, stat_idx + 47) = (collect(i, j, k, collected_idx + 24) / counter_ud[collected_idx + 24] -
                                  2 * collect(i, j, k, collected_idx + 21) / counter_ud[collected_idx + 21] * o2 -
                                  collect(i, j, k, collected_idx + 13) / counter_ud[collected_idx + 13] *
                                  mean(i, j, k, 3)) * rho_inv + 2 * o2 * o2 * mean(i, j, k, 3);

  // N2
  const real n2 = mean(i, j, k, 8);
  // {N2''N2''}
  stat(i, j, k, stat_idx + 52) = max(
      collect(i, j, k, collected_idx + 26) / counter_ud[collected_idx + 26] * rho_inv - n2 * n2, 1e-30);
  // chi=2/<rho>*[<rho*D*gradN2*gradN2>-2<rho*D*N2x>*{N2}_x-2<rho*D*N2y>*{N2}_y-2<rho*D*N2z>*{N2}_z+<rho*D>*grad{N2}*grad{N2}]
  zx = 0.5 * (xi_x * (mean(i + 1, j, k, 8) - mean(i - 1, j, k, 8)) +
              eta_x * (mean(i, j + 1, k, 8) - mean(i, j - 1, k, 8)) +
              zeta_x * (mean(i, j, k + 1, 8) - mean(i, j, k - 1, 8)));
  zy = 0.5 * (xi_y * (mean(i + 1, j, k, 8) - mean(i - 1, j, k, 8)) +
              eta_y * (mean(i, j + 1, k, 8) - mean(i, j - 1, k, 8)) +
              zeta_y * (mean(i, j, k + 1, 8) - mean(i, j, k - 1, 8)));
  zz = 0.5 * (xi_z * (mean(i + 1, j, k, 8) - mean(i - 1, j, k, 8)) +
              eta_z * (mean(i, j + 1, k, 8) - mean(i, j - 1, k, 8)) +
              zeta_z * (mean(i, j, k + 1, 8) - mean(i, j, k - 1, 8)));
  zizi = zx * zx + zy * zy + zz * zz;
  chi = two_rho_inv * (collect(i, j, k, collected_idx + 27) / counter_ud[collected_idx + 27] -
                       2 * collect(i, j, k, collected_idx + 28) / counter_ud[collected_idx + 28] * zx -
                       2 * collect(i, j, k, collected_idx + 29) / counter_ud[collected_idx + 29] * zy -
                       2 * collect(i, j, k, collected_idx + 30) / counter_ud[collected_idx + 30] * zz +
                       collect(i, j, k, collected_idx + 31) / counter_ud[collected_idx + 31] * zizi);
  stat(i, j, k, stat_idx + 53) = max(chi, 1e-30);
  stat(i, j, k, stat_idx + 54) = min(stat(i, j, k, stat_idx + 52) / max(chi, 1e-30), 1.0);
  // {u''N2''}, {v''N2''}, {w''N2''}
  upZp = collect(i, j, k, collected_idx + 32) / counter_ud[collected_idx + 32] * rho_inv - mean(i, j, k, 1) * n2;
  vpZp = collect(i, j, k, collected_idx + 33) / counter_ud[collected_idx + 33] * rho_inv - mean(i, j, k, 2) * n2;
  wpZp = collect(i, j, k, collected_idx + 34) / counter_ud[collected_idx + 34] * rho_inv - mean(i, j, k, 3) * n2;
  stat(i, j, k, stat_idx + 55) = upZp;
  stat(i, j, k, stat_idx + 56) = vpZp;
  stat(i, j, k, stat_idx + 57) = wpZp;
  // Dt_N2,x; Dt_N2,y; Dt_N2,z. And corresponding Sct.
  if (abs(zx) > 1e-25 && abs(upZp) > 1e-25) {
    stat(i, j, k, stat_idx + 58) = -upZp / zx;
    stat(i, j, k, stat_idx + 63) = -nut * zx / upZp;
  }
  if (abs(zy) > 1e-25 && abs(vpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 59) = -vpZp / zy;
    stat(i, j, k, stat_idx + 64) = -nut * zy / vpZp;
  }
  if (abs(zz) > 1e-25 && abs(wpZp) > 1e-25) {
    stat(i, j, k, stat_idx + 60) = -wpZp / zz;
    stat(i, j, k, stat_idx + 65) = -nut * zz / wpZp;
  }
  // Dt_N2
  if (real isoPart = upZp * zx + vpZp * zy + wpZp * zz; zizi > 1e-25 && abs(isoPart) > 1e-25) {
    stat(i, j, k, stat_idx + 61) = -isoPart / zizi;
    stat(i, j, k, stat_idx + 66) = -nut * zizi / isoPart;
  }
  // Sc_N2_laminar
  stat(i, j, k, stat_idx + 62) = collect(i, j, k, collected_idx + 38) / counter_ud[collected_idx + 38];
  // Z''2 flux
  stat(i, j, k, stat_idx + 67) = (collect(i, j, k, collected_idx + 35) / counter_ud[collected_idx + 35] -
                                  2 * collect(i, j, k, collected_idx + 32) / counter_ud[collected_idx + 32] * n2 -
                                  collect(i, j, k, collected_idx + 26) / counter_ud[collected_idx + 26] *
                                  mean(i, j, k, 1)) * rho_inv + 2 * n2 * n2 * mean(i, j, k, 1);
  stat(i, j, k, stat_idx + 68) = (collect(i, j, k, collected_idx + 36) / counter_ud[collected_idx + 36] -
                                  2 * collect(i, j, k, collected_idx + 33) / counter_ud[collected_idx + 33] * n2 -
                                  collect(i, j, k, collected_idx + 26) / counter_ud[collected_idx + 26] *
                                  mean(i, j, k, 2)) * rho_inv + 2 * n2 * n2 * mean(i, j, k, 2);
  stat(i, j, k, stat_idx + 69) = (collect(i, j, k, collected_idx + 37) / counter_ud[collected_idx + 37] -
                                  2 * collect(i, j, k, collected_idx + 34) / counter_ud[collected_idx + 34] * n2 -
                                  collect(i, j, k, collected_idx + 26) / counter_ud[collected_idx + 26] *
                                  mean(i, j, k, 3)) * rho_inv + 2 * n2 * n2 * mean(i, j, k, 3);
}

__device__ void
H2AirMixingLayer::compute_2nd_level(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                                    int k,
                                    int counter, int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  if (i > 0 && i < zone->mx - 1 && j > 0 && j < zone->my - 1) {
    if (zone->mz == 1) {
      // Compute the gradient of {Z''Z''}, which is computed with the first level stat.
      // H2
      real zx = 0.5 * (xi_x * (stat(i + 1, j, 0, stat_idx + 8) - stat(i - 1, j, 0, stat_idx + 8)) +
                       eta_x * (stat(i, j + 1, 0, stat_idx + 8) - stat(i, j - 1, 0, stat_idx + 8)));
      real zy = 0.5 * (xi_y * (stat(i + 1, j, 0, stat_idx + 8) - stat(i - 1, j, 0, stat_idx + 8)) +
                       eta_y * (stat(i, j + 1, 0, stat_idx + 8) - stat(i, j - 1, 0, stat_idx + 8)));
      real upH2pH2p = stat(i, j, k, stat_idx + 23), vpH2pH2p = stat(i, j, k, stat_idx + 24);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 26) = -stat(i, j, 0, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 27) = -stat(i, j, 0, stat_idx + 7) * zy / vpH2pH2p;
      }
      real ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, 0, stat_idx + 29) = -stat(i, j, 0, stat_idx + 7) * (zx * zx + zy * zy) / ziUZFlux;
      }

      // O2
      zx = 0.5 * (xi_x * (stat(i + 1, j, 0, stat_idx + 30) - stat(i - 1, j, 0, stat_idx + 30)) +
                  eta_x * (stat(i, j + 1, 0, stat_idx + 30) - stat(i, j - 1, 0, stat_idx + 30)));
      zy = 0.5 * (xi_y * (stat(i + 1, j, 0, stat_idx + 30) - stat(i - 1, j, 0, stat_idx + 30)) +
                  eta_y * (stat(i, j + 1, 0, stat_idx + 30) - stat(i, j - 1, 0, stat_idx + 30)));
      upH2pH2p = stat(i, j, k, stat_idx + 45);
      vpH2pH2p = stat(i, j, k, stat_idx + 46);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 48) = -stat(i, j, 0, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 49) = -stat(i, j, 0, stat_idx + 7) * zy / vpH2pH2p;
      }
      ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, 0, stat_idx + 51) = -stat(i, j, 0, stat_idx + 7) * (zx * zx + zy * zy) / ziUZFlux;
      }

      // N2
      zx = 0.5 * (xi_x * (stat(i + 1, j, 0, stat_idx + 52) - stat(i - 1, j, 0, stat_idx + 52)) +
                  eta_x * (stat(i, j + 1, 0, stat_idx + 52) - stat(i, j - 1, 0, stat_idx + 52)));
      zy = 0.5 * (xi_y * (stat(i + 1, j, 0, stat_idx + 52) - stat(i - 1, j, 0, stat_idx + 52)) +
                  eta_y * (stat(i, j + 1, 0, stat_idx + 52) - stat(i, j - 1, 0, stat_idx + 52)));
      upH2pH2p = stat(i, j, k, stat_idx + 67);
      vpH2pH2p = stat(i, j, k, stat_idx + 68);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 70) = -stat(i, j, 0, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, 0, stat_idx + 71) = -stat(i, j, 0, stat_idx + 7) * zy / vpH2pH2p;
      }
      ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, 0, stat_idx + 73) = -stat(i, j, 0, stat_idx + 7) * (zx * zx + zy * zy) / ziUZFlux;
      }
    } else if (k > 0 && k < zone->mz - 1) {
      // H2
      real zx = 0.5 * (xi_x * (stat(i + 1, j, k, stat_idx + 8) - stat(i - 1, j, k, stat_idx + 8)) +
                       eta_x * (stat(i, j + 1, k, stat_idx + 8) - stat(i, j - 1, k, stat_idx + 8)) +
                       zeta_x * (stat(i, j, k + 1, stat_idx + 8) - stat(i, j, k - 1, stat_idx + 8)));
      real zy = 0.5 * (xi_y * (stat(i + 1, j, k, stat_idx + 8) - stat(i - 1, j, k, stat_idx + 8)) +
                       eta_y * (stat(i, j + 1, k, stat_idx + 8) - stat(i, j - 1, k, stat_idx + 8)) +
                       zeta_y * (stat(i, j, k + 1, stat_idx + 8) - stat(i, j, k - 1, stat_idx + 8)));
      real zz = 0.5 * (xi_z * (stat(i + 1, j, k, stat_idx + 8) - stat(i - 1, j, k, stat_idx + 8)) +
                       eta_z * (stat(i, j + 1, k, stat_idx + 8) - stat(i, j - 1, k, stat_idx + 8)) +
                       zeta_z * (stat(i, j, k + 1, stat_idx + 8) - stat(i, j, k - 1, stat_idx + 8)));
      real upH2pH2p = stat(i, j, k, stat_idx + 23), vpH2pH2p = stat(i, j, k, stat_idx + 24),
          wpH2pH2p = stat(i, j, k, stat_idx + 25);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 26) = -stat(i, j, k, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 27) = -stat(i, j, k, stat_idx + 7) * zy / vpH2pH2p;
      }
      if (abs(wpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 28) = -stat(i, j, k, stat_idx + 7) * zz / wpH2pH2p;
      }
      real ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy + wpH2pH2p * zz;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, k, stat_idx + 29) = -stat(i, j, k, stat_idx + 7) * (zx * zx + zy * zy + zz * zz) / ziUZFlux;
      }

      // O2
      zx = 0.5 * (xi_x * (stat(i + 1, j, k, stat_idx + 30) - stat(i - 1, j, k, stat_idx + 30)) +
                  eta_x * (stat(i, j + 1, k, stat_idx + 30) - stat(i, j - 1, k, stat_idx + 30)) +
                  zeta_x * (stat(i, j, k + 1, stat_idx + 30) - stat(i, j, k - 1, stat_idx + 30)));
      zy = 0.5 * (xi_y * (stat(i + 1, j, k, stat_idx + 30) - stat(i - 1, j, k, stat_idx + 30)) +
                  eta_y * (stat(i, j + 1, k, stat_idx + 30) - stat(i, j - 1, k, stat_idx + 30)) +
                  zeta_y * (stat(i, j, k + 1, stat_idx + 30) - stat(i, j, k - 1, stat_idx + 30)));
      zz = 0.5 * (xi_z * (stat(i + 1, j, k, stat_idx + 30) - stat(i - 1, j, k, stat_idx + 30)) +
                  eta_z * (stat(i, j + 1, k, stat_idx + 30) - stat(i, j - 1, k, stat_idx + 30)) +
                  zeta_z * (stat(i, j, k + 1, stat_idx + 30) - stat(i, j, k - 1, stat_idx + 30)));
      upH2pH2p = stat(i, j, k, stat_idx + 45);
      vpH2pH2p = stat(i, j, k, stat_idx + 46);
      wpH2pH2p = stat(i, j, k, stat_idx + 47);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 48) = -stat(i, j, k, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 49) = -stat(i, j, k, stat_idx + 7) * zy / vpH2pH2p;
      }
      if (abs(wpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 50) = -stat(i, j, k, stat_idx + 7) * zz / wpH2pH2p;
      }
      ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy + wpH2pH2p * zz;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, k, stat_idx + 51) = -stat(i, j, k, stat_idx + 7) * (zx * zx + zy * zy + zz * zz) / ziUZFlux;
      }

      // N2
      zx = 0.5 * (xi_x * (stat(i + 1, j, k, stat_idx + 52) - stat(i - 1, j, k, stat_idx + 52)) +
                  eta_x * (stat(i, j + 1, k, stat_idx + 52) - stat(i, j - 1, k, stat_idx + 52)) +
                  zeta_x * (stat(i, j, k + 1, stat_idx + 52) - stat(i, j, k - 1, stat_idx + 52)));
      zy = 0.5 * (xi_y * (stat(i + 1, j, k, stat_idx + 52) - stat(i - 1, j, k, stat_idx + 52)) +
                  eta_y * (stat(i, j + 1, k, stat_idx + 52) - stat(i, j - 1, k, stat_idx + 52)) +
                  zeta_y * (stat(i, j, k + 1, stat_idx + 52) - stat(i, j, k - 1, stat_idx + 52)));
      zz = 0.5 * (xi_z * (stat(i + 1, j, k, stat_idx + 52) - stat(i - 1, j, k, stat_idx + 52)) +
                  eta_z * (stat(i, j + 1, k, stat_idx + 52) - stat(i, j - 1, k, stat_idx + 52)) +
                  zeta_z * (stat(i, j, k + 1, stat_idx + 52) - stat(i, j, k - 1, stat_idx + 52)));
      upH2pH2p = stat(i, j, k, stat_idx + 67);
      vpH2pH2p = stat(i, j, k, stat_idx + 68);
      wpH2pH2p = stat(i, j, k, stat_idx + 69);
      if (abs(upH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 70) = -stat(i, j, k, stat_idx + 7) * zx / upH2pH2p;
      }
      if (abs(vpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 71) = -stat(i, j, k, stat_idx + 7) * zy / vpH2pH2p;
      }
      if (abs(wpH2pH2p) > 1e-25) {
        stat(i, j, k, stat_idx + 72) = -stat(i, j, k, stat_idx + 7) * zz / wpH2pH2p;
      }
      ziUZFlux = upH2pH2p * zx + vpH2pH2p * zy + wpH2pH2p * zz;
      if (abs(ziUZFlux) > 1e-25) {
        stat(i, j, k, stat_idx + 73) = -stat(i, j, k, stat_idx + 7) * (zx * zx + zy * zy + zz * zz) / ziUZFlux;
      }
    }
  }
}


__device__ void
H2AirMixingLayer::compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i,
                                           int j, int mz, int counter, int stat_idx, int collected_idx) {
  if (mz == 1) {
    compute(zone, param, counter_ud, i, j, 0, counter, stat_idx, collected_idx);
    return;
  }
  auto &stat = zone->user_defined_statistical_data;
  auto &collected_moments = zone->userCollectForStat;
  auto &firstOrderMoment = zone->firstOrderMoment;
  auto &vel2ndMoment = zone->velocity2ndMoment;

  const real counter_inv = 1.0 / counter;
  int useful[6]{0, 0, 0, 0, 0, 0};
  real add[6]{0, 0, 0, 0, 0, 0};
  int useful_nut[7]{0, 0, 0, 0, 0, 0, 0};
  real nut_add[7]{0, 0, 0, 0, 0, 0, 0};
  real divV_add{0};
  for (int k = 0; k < mz; ++k) {
    const real sum_density_inv = 1.0 / firstOrderMoment(i, j, k, 0);
    const real rho_inv = counter / firstOrderMoment(i, j, k, 0);
    const auto &m = zone->metric(i, j, k);
    const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
    const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
    const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
    // H2
    real yk = firstOrderMoment(i, j, k, 6) * sum_density_inv;
    // {H2''H2''}
    if (real temp = collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] * rho_inv - yk * yk;
        temp > 1e-30) {
      add[0] += temp;
      ++useful[0];
    }
    real dz_xi = firstOrderMoment(i + 1, j, k, 6) / firstOrderMoment(i + 1, j, k, 0) -
                 firstOrderMoment(i - 1, j, k, 6) / firstOrderMoment(i - 1, j, k, 0);
    real dz_eta = firstOrderMoment(i, j + 1, k, 6) / firstOrderMoment(i, j + 1, k, 0) -
                  firstOrderMoment(i, j - 1, k, 6) / firstOrderMoment(i, j - 1, k, 0);
    real dz_zeta = firstOrderMoment(i, j, k + 1, 6) / firstOrderMoment(i, j, k + 1, 0) -
                   firstOrderMoment(i, j, k - 1, 6) / firstOrderMoment(i, j, k - 1, 0);
    real zx = 0.5 * (xi_x * dz_xi + eta_x * dz_eta + zeta_x * dz_zeta);
    real zy = 0.5 * (xi_y * dz_xi + eta_y * dz_eta + zeta_y * dz_zeta);
    real zz = 0.5 * (xi_z * dz_xi + eta_z * dz_eta + zeta_z * dz_zeta);
    // chi=2/<rho>*[<rho*D*gradH2*gradH2>-2<rho*D*H2x>*{H2}_x-2<rho*D*H2y>*{H2}_y-2<rho*D*H2z>*{H2}_z+<rho*D>*grad{H2}*grad{H2}]
    real chi = 2 * rho_inv * (collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                              2 * collected_moments(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2] * zx -
                              2 * collected_moments(i, j, k, collected_idx + 3) / counter_ud[collected_idx + 3] * zy -
                              2 * collected_moments(i, j, k, collected_idx + 4) / counter_ud[collected_idx + 4] * zz +
                              collected_moments(i, j, k, collected_idx + 5) / counter_ud[collected_idx + 5] *
                              (zx * zx + zy * zy + zz * zz));
    if (chi > 1e-30) {
      add[1] += chi;
      ++useful[1];
    }

    // O2
    yk = firstOrderMoment(i, j, k, 7) * sum_density_inv;
    // {O2''O2''}
    if (real temp =
          collected_moments(i, j, k, collected_idx + 6) / counter_ud[collected_idx + 6] * rho_inv - yk * yk;
        temp > 1e-30) {
      add[2] += temp;
      ++useful[2];
    }
    dz_xi = firstOrderMoment(i + 1, j, k, 7) / firstOrderMoment(i + 1, j, k, 0) -
            firstOrderMoment(i - 1, j, k, 7) / firstOrderMoment(i - 1, j, k, 0);
    dz_eta = firstOrderMoment(i, j + 1, k, 7) / firstOrderMoment(i, j + 1, k, 0) -
             firstOrderMoment(i, j - 1, k, 7) / firstOrderMoment(i, j - 1, k, 0);
    dz_zeta = firstOrderMoment(i, j, k + 1, 7) / firstOrderMoment(i, j, k + 1, 0) -
              firstOrderMoment(i, j, k - 1, 7) / firstOrderMoment(i, j, k - 1, 0);
    zx = 0.5 * (xi_x * dz_xi + eta_x * dz_eta + zeta_x * dz_zeta);
    zy = 0.5 * (xi_y * dz_xi + eta_y * dz_eta + zeta_y * dz_zeta);
    zz = 0.5 * (xi_z * dz_xi + eta_z * dz_eta + zeta_z * dz_zeta);
    // chi=2/<rho>*[<rho*D*gradO2*gradO2>-2<rho*D*O2x>*{O2}_x-2<rho*D*O2y>*{O2}_y-2<rho*D*O2z>*{O2}_z+<rho*D>*grad{O2}*grad{O2}]
    chi = 2 * rho_inv * (collected_moments(i, j, k, collected_idx + 7) / counter_ud[collected_idx + 7] -
                         2 * collected_moments(i, j, k, collected_idx + 8) / counter_ud[collected_idx + 8] * zx -
                         2 * collected_moments(i, j, k, collected_idx + 9) / counter_ud[collected_idx + 9] * zy -
                         2 * collected_moments(i, j, k, collected_idx + 10) / counter_ud[collected_idx + 10] * zz +
                         collected_moments(i, j, k, collected_idx + 11) / counter_ud[collected_idx + 11] *
                         (zx * zx + zy * zy + zz * zz));
    if (chi > 1e-30) {
      add[3] += chi;
      ++useful[3];
    }

    // N2
    yk = firstOrderMoment(i, j, k, 8) * sum_density_inv;
    // {N2''N2''}
    if (real temp =
          collected_moments(i, j, k, collected_idx + 12) / counter_ud[collected_idx + 12] * rho_inv - yk * yk;
        temp > 1e-30) {
      add[4] += temp;
      ++useful[4];
    }
    dz_xi = firstOrderMoment(i + 1, j, k, 8) / firstOrderMoment(i + 1, j, k, 0) -
            firstOrderMoment(i - 1, j, k, 8) / firstOrderMoment(i - 1, j, k, 0);
    dz_eta = firstOrderMoment(i, j + 1, k, 8) / firstOrderMoment(i, j + 1, k, 0) -
             firstOrderMoment(i, j - 1, k, 8) / firstOrderMoment(i, j - 1, k, 0);
    dz_zeta = firstOrderMoment(i, j, k + 1, 8) / firstOrderMoment(i, j, k + 1, 0) -
              firstOrderMoment(i, j, k - 1, 8) / firstOrderMoment(i, j, k - 1, 8);
    zx = 0.5 * (xi_x * dz_xi + eta_x * dz_eta + zeta_x * dz_zeta);
    zy = 0.5 * (xi_y * dz_xi + eta_y * dz_eta + zeta_y * dz_zeta);
    zz = 0.5 * (xi_z * dz_xi + eta_z * dz_eta + zeta_z * dz_zeta);
    // chi=2/<rho>*[<rho*D*gradN2*gradN2>-2<rho*D*N2x>*{N2}_x-2<rho*D*N2y>*{N2}_y-2<rho*D*N2z>*{N2}_z+<rho*D>*grad{N2}*grad{N2}]
    chi = 2 * rho_inv * (collected_moments(i, j, k, collected_idx + 13) / counter_ud[collected_idx + 13] -
                         2 * collected_moments(i, j, k, collected_idx + 14) / counter_ud[collected_idx + 14] * zx -
                         2 * collected_moments(i, j, k, collected_idx + 15) / counter_ud[collected_idx + 15] * zy -
                         2 * collected_moments(i, j, k, collected_idx + 16) / counter_ud[collected_idx + 16] * zz +
                         collected_moments(i, j, k, collected_idx + 17) / counter_ud[collected_idx + 17] *
                         (zx * zx + zy * zy + zz * zz));
    if (chi > 1e-30) {
      add[5] += chi;
      ++useful[5];
    }

    // nut
    const auto sumRho2Inv{sum_density_inv * sum_density_inv};
    real uu = vel2ndMoment(i, j, k, 0) * sum_density_inv -
              firstOrderMoment(i, j, k, 1) * firstOrderMoment(i, j, k, 1) * sumRho2Inv;
    real vv = vel2ndMoment(i, j, k, 1) * sum_density_inv -
              firstOrderMoment(i, j, k, 2) * firstOrderMoment(i, j, k, 2) * sumRho2Inv;
    real ww = vel2ndMoment(i, j, k, 2) * sum_density_inv -
              firstOrderMoment(i, j, k, 3) * firstOrderMoment(i, j, k, 3) * sumRho2Inv;
    real Tau12 = vel2ndMoment(i, j, k, 3) * sum_density_inv -
                 firstOrderMoment(i, j, k, 1) * firstOrderMoment(i, j, k, 2) * sumRho2Inv;
    real Tau13 = vel2ndMoment(i, j, k, 4) * sum_density_inv -
                 firstOrderMoment(i, j, k, 1) * firstOrderMoment(i, j, k, 3) * sumRho2Inv;
    real Tau23 = vel2ndMoment(i, j, k, 5) * sum_density_inv -
                 firstOrderMoment(i, j, k, 2) * firstOrderMoment(i, j, k, 3) * sumRho2Inv;
    const real tke = 0.5 * (uu + vv + ww);
    real Tau11 = uu - 2.0 / 3 * tke;
    real Tau22 = vv - 2.0 / 3 * tke;
    real Tau33 = ww - 2.0 / 3 * tke;
    real ux = 0.5 * (xi_x * (firstOrderMoment(i + 1, j, k, 1) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 1) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_x * (firstOrderMoment(i, j + 1, k, 1) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 1) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_x * (firstOrderMoment(i, j, k + 1, 1) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 1) / firstOrderMoment(i, j, k - 1, 0)));
    real uy = 0.5 * (xi_y * (firstOrderMoment(i + 1, j, k, 1) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 1) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_y * (firstOrderMoment(i, j + 1, k, 1) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 1) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_y * (firstOrderMoment(i, j, k + 1, 1) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 1) / firstOrderMoment(i, j, k - 1, 0)));
    real uz = 0.5 * (xi_z * (firstOrderMoment(i + 1, j, k, 1) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 1) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_z * (firstOrderMoment(i, j + 1, k, 1) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 1) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_z * (firstOrderMoment(i, j, k + 1, 1) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 1) / firstOrderMoment(i, j, k - 1, 0)));
    real vx = 0.5 * (xi_x * (firstOrderMoment(i + 1, j, k, 2) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 2) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_x * (firstOrderMoment(i, j + 1, k, 2) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 2) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_x * (firstOrderMoment(i, j, k + 1, 2) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 2) / firstOrderMoment(i, j, k - 1, 0)));
    real vy = 0.5 * (xi_y * (firstOrderMoment(i + 1, j, k, 2) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 2) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_y * (firstOrderMoment(i, j + 1, k, 2) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 2) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_y * (firstOrderMoment(i, j, k + 1, 2) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 2) / firstOrderMoment(i, j, k - 1, 0)));
    real vz = 0.5 * (xi_z * (firstOrderMoment(i + 1, j, k, 2) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 2) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_z * (firstOrderMoment(i, j + 1, k, 2) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 2) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_z * (firstOrderMoment(i, j, k + 1, 2) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 2) / firstOrderMoment(i, j, k - 1, 0)));
    real wx = 0.5 * (xi_x * (firstOrderMoment(i + 1, j, k, 3) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 3) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_x * (firstOrderMoment(i, j + 1, k, 3) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 3) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_x * (firstOrderMoment(i, j, k + 1, 3) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 3) / firstOrderMoment(i, j, k - 1, 0)));
    real wy = 0.5 * (xi_y * (firstOrderMoment(i + 1, j, k, 3) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 3) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_y * (firstOrderMoment(i, j + 1, k, 3) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 3) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_y * (firstOrderMoment(i, j, k + 1, 3) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 3) / firstOrderMoment(i, j, k - 1, 0)));
    real wz = 0.5 * (xi_z * (firstOrderMoment(i + 1, j, k, 3) / firstOrderMoment(i + 1, j, k, 0) -
                             firstOrderMoment(i - 1, j, k, 3) / firstOrderMoment(i - 1, j, k, 0)) +
                     eta_z * (firstOrderMoment(i, j + 1, k, 3) / firstOrderMoment(i, j + 1, k, 0) -
                              firstOrderMoment(i, j - 1, k, 3) / firstOrderMoment(i, j - 1, k, 0)) +
                     zeta_z * (firstOrderMoment(i, j, k + 1, 3) / firstOrderMoment(i, j, k + 1, 0) -
                               firstOrderMoment(i, j, k - 1, 3) / firstOrderMoment(i, j, k - 1, 0)));
    real divV = ux + vy + wz;
    divV_add += divV;
    real S11 = 2 * ux - 2.0 / 3 * divV;
    real S22 = 2 * vy - 2.0 / 3 * divV;
    real S33 = 2 * wz - 2.0 / 3 * divV;
    real S12 = uy + vx;
    real S13 = uz + wx;
    real S23 = vz + wy;
    real SiiSii = S11 * S11 + S22 * S22 + S33 * S33 + 2 * S12 * S12 + 2 * S13 * S13 + 2 * S23 * S23;
    if (S11 > 1e-25) {
      nut_add[0] += -Tau11 / S11;
      ++useful_nut[0];
    }
    if (S22 > 1e-25) {
      nut_add[1] += -Tau22 / S22;
      ++useful_nut[1];
    }
    if (S33 > 1e-25) {
      nut_add[2] += -Tau33 / S33;
      ++useful_nut[2];
    }
    if (S12 > 1e-25) {
      nut_add[3] += -Tau12 / S12;
      ++useful_nut[3];
    }
    if (S13 > 1e-25) {
      nut_add[4] += -Tau13 / S13;
      ++useful_nut[4];
    }
    if (S23 > 1e-25) {
      nut_add[5] += -Tau23 / S23;
      ++useful_nut[5];
    }
    if (SiiSii > 1e-25) {
      nut_add[6] += -(Tau11 * S11 + Tau22 * S22 + Tau33 * S33 + Tau12 * S12 + Tau13 * S13 + Tau23 * S23) / SiiSii;
      ++useful_nut[6];
    }
  }
  stat(i, j, 0, stat_idx) = useful[0] > 0 ? max(add[0] / useful[0], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 1) = useful[1] > 0 ? max(add[1] / useful[1], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 2) = min(stat(i, j, 0, stat_idx) / stat(i, j, 0, stat_idx + 1), 1.0);
  stat(i, j, 0, stat_idx + 3) = useful[2] > 0 ? max(add[2] / useful[2], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 4) = useful[3] > 0 ? max(add[3] / useful[3], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 5) = min(stat(i, j, 0, stat_idx + 3) / stat(i, j, 0, stat_idx + 4), 1.0);
  stat(i, j, 0, stat_idx + 6) = useful[4] > 0 ? max(add[4] / useful[4], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 7) = useful[5] > 0 ? max(add[5] / useful[5], 1e-30) : 1e-30;
  stat(i, j, 0, stat_idx + 8) = min(stat(i, j, 0, stat_idx + 6) / stat(i, j, 0, stat_idx + 7), 1.0);
  stat(i, j, 0, stat_idx + 9) = divV_add / mz;
  stat(i, j, 0, stat_idx + 10) = useful_nut[0] > 0 ? nut_add[0] / useful_nut[0] : 0;
  stat(i, j, 0, stat_idx + 11) = useful_nut[1] > 0 ? nut_add[1] / useful_nut[1] : 0;
  stat(i, j, 0, stat_idx + 12) = useful_nut[2] > 0 ? nut_add[2] / useful_nut[2] : 0;
  stat(i, j, 0, stat_idx + 13) = useful_nut[3] > 0 ? nut_add[3] / useful_nut[3] : 0;
  stat(i, j, 0, stat_idx + 14) = useful_nut[4] > 0 ? nut_add[4] / useful_nut[4] : 0;
  stat(i, j, 0, stat_idx + 15) = useful_nut[5] > 0 ? nut_add[5] / useful_nut[5] : 0;
  stat(i, j, 0, stat_idx + 16) = useful_nut[6] > 0 ? nut_add[6] / useful_nut[6] : 0;
}

__device__ void
H2_variance_and_dissipation_rate::collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k,
                                          int collect_idx) {
  auto &stat = zone->userCollectForStat; // There may be mistakes!
  const auto &sv = zone->sv;

  // Rho*Z*Z
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
  const auto rhoD = zone->rho_D(i, j, k, z_idx);
  // Rho*D*GradZ*GradZ
  const real rhoChi = rhoD * (z_x * z_x + z_y * z_y + z_z * z_z);
  stat(i, j, k, collect_idx + 1) += rhoChi;
  // Rho*D*Zx
  stat(i, j, k, collect_idx + 2) += rhoD * z_x;
  // Rho*D*Zy
  stat(i, j, k, collect_idx + 3) += rhoD * z_y;
  // Rho*D*Zz
  stat(i, j, k, collect_idx + 4) += rhoD * z_z;
  stat(i, j, k, collect_idx + 5) += rhoD;
}

__device__ void
H2_variance_and_dissipation_rate::compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                                          int k, int counter, int stat_idx, int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collected_moments = zone->userCollectForStat;
  auto &mean = zone->mean_value;

  // {z''z''}
  stat(i, j, k, stat_idx) = max(
      collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] / mean(i, j, k, 0) -
      mean(i, j, k, 6 + z_idx) * mean(i, j, k, 6 + z_idx), 1e-30);

  // chi=2/<rho>*[<rho*D*gradZ*gradZ>-2<rho*D*Zx>*{Z}_x-2<rho*D*Zy>*{Z}_y-2<rho*D*Zz>*{Z}_z+<rho*D>*grad{Z}*grad{Z}]
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
  auto mx{zone->mx}, my{zone->my}, mz{zone->mz};
  real chi{0.0};
  if (mz == 1) {
    // 2D case
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1) {
      const real z_x = 0.5 * (xi_x * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_x * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)));
      const real z_y = 0.5 * (xi_y * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_y * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)));
      const real z_z = 0.5 * (xi_z * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_z * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)));
      chi = 2.0 / mean(i, j, k, 0) * (collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                                      2.0 * collected_moments(i, j, k, collected_idx + 2) /
                                      counter_ud[collected_idx + 2] * z_x -
                                      2.0 * collected_moments(i, j, k, collected_idx + 3) /
                                      counter_ud[collected_idx + 3] * z_y -
                                      2.0 * collected_moments(i, j, k, collected_idx + 4) /
                                      counter_ud[collected_idx + 4] * z_z +
                                      collected_moments(i, j, k, collected_idx + 5) / counter_ud[collected_idx + 5] *
                                      (z_x * z_x + z_y * z_y + z_z * z_z));
    }
  } else {
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
      const real z_x = 0.5 * (xi_x * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_x * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)) +
                              zeta_x * (mean(i, j, k + 1, 6 + z_idx) - mean(i, j, k - 1, 6 + z_idx)));
      const real z_y = 0.5 * (xi_y * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_y * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)) +
                              zeta_y * (mean(i, j, k + 1, 6 + z_idx) - mean(i, j, k - 1, 6 + z_idx)));
      const real z_z = 0.5 * (xi_z * (mean(i + 1, j, k, 6 + z_idx) - mean(i - 1, j, k, 6 + z_idx)) +
                              eta_z * (mean(i, j + 1, k, 6 + z_idx) - mean(i, j - 1, k, 6 + z_idx)) +
                              zeta_z * (mean(i, j, k + 1, 6 + z_idx) - mean(i, j, k - 1, 6 + z_idx)));
      chi = 2.0 / mean(i, j, k, 0) * (collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                                      2.0 * collected_moments(i, j, k, collected_idx + 2) /
                                      counter_ud[collected_idx + 2] * z_x -
                                      2.0 * collected_moments(i, j, k, collected_idx + 3) /
                                      counter_ud[collected_idx + 3] * z_y -
                                      2.0 * collected_moments(i, j, k, collected_idx + 4) /
                                      counter_ud[collected_idx + 4] * z_z +
                                      collected_moments(i, j, k, collected_idx + 5) / counter_ud[collected_idx + 5] *
                                      (z_x * z_x + z_y * z_y + z_z * z_z));
    }
  }
  stat(i, j, k, stat_idx + 1) = max(chi, 1e-30);
  stat(i, j, k, stat_idx + 2) = min(stat(i, j, k, stat_idx) / max(chi, 1e-30), 1.0);
}

__device__ void H2_variance_and_dissipation_rate::compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param,
                                                                           const int *counter_ud, int i, int j, int mz,
                                                                           int counter, int stat_idx,
                                                                           int collected_idx) {
  auto &stat = zone->user_defined_statistical_data;
  auto &collected_moments = zone->userCollectForStat;
  auto &firstOrderMoment = zone->firstOrderMoment;

  const real counter_inv{1.0 / counter};
  real add_zVar{0}, add_chi{0};
  auto mx{zone->mx}, my{zone->my};
  if (mz == 1) {
    const real density = firstOrderMoment(i, j, 0, 0) * counter_inv;
    const real z_favre = firstOrderMoment(i, j, 0, 6 + z_idx) * counter_inv / density;
    // {z''z''}
    add_zVar += max(collected_moments(i, j, 0, collected_idx) / counter_ud[collected_idx] / density -
                    z_favre * z_favre, 1e-30);

    // compute the surrounding 8 points' z_favre
    auto chi{0.0};
    if (i > 0 && i < mx - 1 && j > 0 && j < my - 1) {
      const real d_zFavre_x = firstOrderMoment(i + 1, j, 0, 6 + z_idx) / firstOrderMoment(i + 1, j, 0, 0) -
                              firstOrderMoment(i - 1, j, 0, 6 + z_idx) / firstOrderMoment(i - 1, j, 0, 0);
      const real d_zFavre_y = firstOrderMoment(i, j + 1, 0, 6 + z_idx) / firstOrderMoment(i, j + 1, 0, 0) -
                              firstOrderMoment(i, j - 1, 0, 6 + z_idx) / firstOrderMoment(i, j - 1, 0, 0);
      // compute the gradient of z
      const auto &m = zone->metric(i, j, 0);
      const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
      const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
      const real z_x = 0.5 * (xi_x * d_zFavre_x + eta_x * d_zFavre_y);
      const real z_y = 0.5 * (xi_y * d_zFavre_x + eta_y * d_zFavre_y);
      const real z_z = 0.5 * (xi_z * d_zFavre_x + eta_z * d_zFavre_y);
      // chi=2/<rho>*[<rho*D*gradZ*gradZ>-2<rho*D*Zx>*{Z}_x-2<rho*D*Zy>*{Z}_y-2<rho*D*Zz>*{Z}_z+<rho*D>*grad{Z}*grad{Z}]
      chi = 2 / density * (collected_moments(i, j, 0, collected_idx + 1) / counter_ud[collected_idx + 1] -
                           2 * collected_moments(i, j, 0, collected_idx + 2) / counter_ud[collected_idx + 2] *
                           z_x -
                           2 * collected_moments(i, j, 0, collected_idx + 3) / counter_ud[collected_idx + 3] *
                           z_y -
                           2 * collected_moments(i, j, 0, collected_idx + 4) / counter_ud[collected_idx + 4] *
                           z_z + collected_moments(i, j, 0, collected_idx + 5) / counter_ud[collected_idx + 5] *
                                 (z_x * z_x + z_y * z_y + z_z * z_z));
    }
    stat(i, j, 0, stat_idx) = add_zVar;
    stat(i, j, 0, stat_idx + 1) = max(chi, 1e-30);
    stat(i, j, 0, stat_idx + 2) = min(add_zVar / max(chi, 1e-30), 1.0);
  } else {
    int useful_counter[2]{0, 0};
    for (int k = 0; k < mz; ++k) {
      const real density = firstOrderMoment(i, j, k, 0) * counter_inv;
      const real z_favre = firstOrderMoment(i, j, k, 6 + z_idx) * counter_inv / density;
      // {z''z''}
      if (auto temp = collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] / density -
                      z_favre * z_favre;temp > 1e-30) {
        add_zVar += temp;
        ++useful_counter[0];
      }
//      add_zVar += max(collected_moments(i, j, k, collected_idx) / counter_ud[collected_idx] / density -
//                      z_favre * z_favre, 1e-30);

      // compute the surrounding 8 points' z_favre
      if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
        const real d_zFavre_x = firstOrderMoment(i + 1, j, k, 6 + z_idx) / firstOrderMoment(i + 1, j, k, 0) -
                                firstOrderMoment(i - 1, j, k, 6 + z_idx) / firstOrderMoment(i - 1, j, k, 0);
        const real d_zFavre_y = firstOrderMoment(i, j + 1, k, 6 + z_idx) / firstOrderMoment(i, j + 1, k, 0) -
                                firstOrderMoment(i, j - 1, k, 6 + z_idx) / firstOrderMoment(i, j - 1, k, 0);
        const real d_zFavre_z = firstOrderMoment(i, j, k + 1, 6 + z_idx) / firstOrderMoment(i, j, k + 1, 0) -
                                firstOrderMoment(i, j, k - 1, 6 + z_idx) / firstOrderMoment(i, j, k - 1, 0);
        // compute the gradient of z
        const auto &m = zone->metric(i, j, k);
        const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
        const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
        const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};
        const real z_x = 0.5 * (xi_x * d_zFavre_x + eta_x * d_zFavre_y + zeta_x * d_zFavre_z);
        const real z_y = 0.5 * (xi_y * d_zFavre_x + eta_y * d_zFavre_y + zeta_y * d_zFavre_z);
        const real z_z = 0.5 * (xi_z * d_zFavre_x + eta_z * d_zFavre_y + zeta_z * d_zFavre_z);
        // chi=2/<rho>*[<rho*D*gradZ*gradZ>-2<rho*D*Zx>*{Z}_x-2<rho*D*Zy>*{Z}_y-2<rho*D*Zz>*{Z}_z+<rho*D>*grad{Z}*grad{Z}]
        auto chi = 2.0 / density * (collected_moments(i, j, k, collected_idx + 1) / counter_ud[collected_idx + 1] -
                                    2 * collected_moments(i, j, k, collected_idx + 2) / counter_ud[collected_idx + 2] *
                                    z_x -
                                    2 * collected_moments(i, j, k, collected_idx + 3) / counter_ud[collected_idx + 3] *
                                    z_y -
                                    2 * collected_moments(i, j, k, collected_idx + 4) / counter_ud[collected_idx + 4] *
                                    z_z +
                                    collected_moments(i, j, k, collected_idx + 5) / counter_ud[collected_idx + 5] *
                                    (z_x * z_x + z_y * z_y + z_z * z_z));
        if (chi > 1e-30) {
          add_chi += chi;
          ++useful_counter[1];
        }
//        add_chi += max(chi, 1e-30);
      }
    }
    stat(i, j, 0, stat_idx) = max(add_zVar / useful_counter[0], 1e-30);
    stat(i, j, 0, stat_idx + 1) = max(add_chi / useful_counter[1], 1e-30);
//    stat(i, j, 0, stat_idx) = max(add_zVar / mz, 1e-30);
//    stat(i, j, 0, stat_idx + 1) = max(add_chi / (mz - 2), 1e-30);
    stat(i, j, 0, stat_idx + 2) = min(stat(i, j, 0, stat_idx) / stat(i, j, 0, stat_idx + 1), 1.0);
  }
}
}