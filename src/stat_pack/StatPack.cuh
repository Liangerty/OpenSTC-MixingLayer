#pragma once

#include "../Define.h"
#include <string>
#include <array>

namespace cfd {
struct DZone;
struct DParameter;

struct ThermRMS {
  static constexpr integer n_collect = 3;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rho2", "p2", "T2"};
  static constexpr integer n_stat = 3;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"rho<sub>rms</sub>", "p<sub>rms</sub>",
                                                                     "T<sub>rms</sub>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j,
                           integer mz,
                           integer counter, integer stat_idx, integer collected_idx);

};

struct StrainRateSquared {
  static constexpr integer n_collect = 1;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"SijSij"};
  static constexpr integer n_stat = 1;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<S_ij S_ij>_F"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j,
                           integer mz,
                           integer counter, integer stat_idx, integer collected_idx);
};

struct mixtureFractionMixingLayer {
  static constexpr integer n_collect = 1;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"Z"};
  static constexpr integer n_stat = 1;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<Z>_F"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j,
                           integer mz,
                           integer counter, integer stat_idx, integer collected_idx);
};

struct turbulent_dissipation_rate {
  static constexpr integer n_collect = 16;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"u_x", "u_y", "u_z", "v_x", "v_y", "v_z",
                                                                           "w_x", "w_y", "w_z",
                                                                           "sigma11", "sigma12", "sigma13", "sigma22",
                                                                           "sigma23", "sigma33", "sigmaIjUi_xj"};
  static constexpr integer n_stat = 2;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<epsilon><sub>F</sub>", "eta"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j,
                           integer mz,
                           integer counter, integer stat_idx, integer collected_idx);
};

struct velFluc_scalarFluc_correlation {
  static constexpr integer n_collect = 3;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rho uZ", "rho vZ", "rho wZ"};
  static constexpr integer n_stat = 3;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<u'Z'>_F", "<v'Z'>_F", "<w'Z'>_F"};
//  static integer collect_idx[3];

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);
};

struct resolved_tke {
  static constexpr integer n_collect = 0;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{};
  static constexpr integer n_stat = 2;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"k_resolved", "grid_resolution"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx) {}

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);
};

struct Z_variance_and_dissipation_rate {
  static constexpr integer n_collect = 2;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rhoZ2", "rho chi"};
  static constexpr integer n_stat = 2;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<Z'Z'>_F", "<scalar_dissipation_rate>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k, integer collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud, integer i, integer j, integer k,
          integer counter, integer stat_idx, integer collected_idx);
};

}
