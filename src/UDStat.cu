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