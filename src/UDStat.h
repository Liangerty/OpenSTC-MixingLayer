#pragma once

#define USER_DEFINE_STATISTICS ThermRMS,turbulent_dissipation_rate,H2_variance_and_dissipation_rate

#include <array>
#include <string_view>
#include "Define.h"
#include "Field.h"

namespace cfd {
struct ThermRMS {
  static constexpr int n_collect = 4;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rho2", "p2", "T2", "T"};
  static constexpr int n_stat = 3;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"rho<sub>rms</sub>", "p<sub>rms</sub>",
                                                                     "T<sub>rms</sub>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int stat_idx, int collected_idx);

};

struct turbulent_dissipation_rate {
//  static constexpr int n_collect = 16;
//  static constexpr std::array<std::string_view, n_collect> namelistCollect{"u_x", "u_y", "u_z", "v_x", "v_y", "v_z",
//                                                                           "w_x", "w_y", "w_z",
//                                                                           "sigma11", "sigma12", "sigma13", "sigma22",
//                                                                           "sigma23", "sigma33", "sigmaIjUi_xj"};
  static constexpr int n_collect = 10;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"u", "v", "w", "sigma11", "sigma12",
                                                                           "sigma13", "sigma22", "sigma23", "sigma33",
                                                                           "sigmaIjUi_xj"};

  static constexpr int n_stat = 4;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<<greek>e</greek>><sub>F</sub>",
                                                                     "<greek>h</greek>", "t<sub><greek>h</greek></sub>",
                                                                     "t<sub>turb</sub>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int stat_idx, int collected_idx);
};

struct H2_variance_and_dissipation_rate {
  static constexpr int n_collect = 6;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rhoZZ", "rhoDGradZGradZ", "rhoDZx",
                                                                           "rhoDZy", "rhoDZz", "rhoD"};
  static constexpr int n_stat = 3;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<Z''Z''><sub>F</sub>", // z''z''
                                                                     "<<greek>c</greek>><sub>F</sub>", // chi
                                                                     "t<sub>scalar</sub>"}; // z''z''/chi
  // The index of H2, which is used as the mixture fraction.
  static constexpr int z_idx{0};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int stat_idx, int collected_idx);
};
}

// User-defined statistical data part
// We should not modify this part.
namespace cfd {
template<typename... stats>
struct UDStat {
  constexpr static int n_collect = (0 + ... + stats::n_collect);
  constexpr static int n_stat = (0 + ... + stats::n_stat);

  constexpr static std::array<std::string_view, n_stat> namelistStat() {
    std::array<std::string_view, n_stat> name_list;
    int i = 0;
    ([&]() {
      for (int j = 0; j < stats::n_stat; ++j) {
        name_list[i++] = stats::namelistStat[j];
      }
    }(), ...);
    return name_list;
  }

  constexpr static std::array<std::string_view, n_collect> namelistCollect() {
    std::array<std::string_view, n_collect> name_list;
    auto i = 0;
    ([&]() {
      for (int j = 0; j < stats::n_collect; ++j) {
        name_list[i++] = stats::namelistCollect[j];
      }
    }(), ...);
    return name_list;
  }
};

template<typename... stats>
constexpr std::array<std::string, (... + stats::n_collect)> acquire_user_defined_statistical_data_name() {
  std::array<std::string, (... + stats::n_collect)> name_list;
  int i = 0;
  ([&]() {
    for (int j = 0; j < stats::n_collect; ++j) {
      name_list[i++] = stats::namelistCollect[j];
    }
  }(), ...);
  return name_list;
}

template<typename... stats>
__device__ void
collect_user_defined_statistics(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k) {
  auto l = 0;
  ([&]() {
    stats::collect(zone, param, i, j, k, l);
    l += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__device__ void
compute_user_defined_statistical_data(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud,
                                      int i, int j, int k, int counter) {
  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute(zone, param, counter_ud, i, j, k, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__global__ void
compute_user_defined_statistical_data(DZone *zone, DParameter *param, int counter, const int *counter_ud) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute(zone, param, counter_ud, i, j, k, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__device__ void
compute_user_defined_statistical_data_with_spanwise_average(cfd::DZone *zone, cfd::DParameter *param,
                                                            const int *counter_ud, int i, int j, int mz, int counter) {
  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute_spanwise_average(zone, param, counter_ud, i, j, mz, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__global__ void
compute_user_defined_statistical_data_with_spanwise_average(cfd::DZone *zone, cfd::DParameter *param,
                                                            const int *counter_ud, int counter) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  if (i >= extent[0] || j >= extent[1]) return;

  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute_spanwise_average(zone, param, counter_ud, i, j, zone->mz, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

using UserDefineStat = UDStat<USER_DEFINE_STATISTICS>;
}
