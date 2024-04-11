#pragma once

#include "Define.h"
#include "UDStat.cu"
#include <array>
#include <string>
#include <vector>
#include "stat_pack/StatPack.cuh"

namespace cfd {

template<typename... stats>
struct UDStat {
  constexpr static integer n_collect = (0 + ... + stats::n_collect);
  constexpr static integer n_stat = (0 + ... + stats::n_stat);

  constexpr static std::array<std::string_view, n_stat> namelistStat() {
    std::array<std::string_view, n_stat> name_list;
    integer i = 0;
    ([&]() {
      for (integer j = 0; j < stats::n_stat; ++j) {
        name_list[i++] = stats::namelistStat[j];
      }
    }(), ...);
    return name_list;
  }

  constexpr static std::array<std::string_view, n_collect> namelistCollect() {
    std::array<std::string_view, n_collect> name_list;
    auto i = 0;
    ([&]() {
      for (integer j = 0; j < stats::n_collect; ++j) {
        name_list[i++] = stats::namelistCollect[j];
      }
    }(), ...);
    return name_list;
  }
};

template<typename... stats>
constexpr std::array<std::string, (... + stats::n_collect)> acquire_user_defined_statistical_data_name() {
  std::array<std::string, (... + stats::n_collect)> name_list;
  integer i = 0;
  ([&]() {
    for (integer j = 0; j < stats::n_collect; ++j) {
      name_list[i++] = stats::namelistCollect[j];
    }
  }(), ...);
  return name_list;
}

template<typename... stats>
__device__ void
collect_user_defined_statistics(cfd::DZone *zone, cfd::DParameter *param, integer i, integer j, integer k) {
  auto l = 0;
  ([&]() {
    stats::collect(zone, param, i, j, k, l);
    l += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__device__ void
compute_user_defined_statistical_data(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud,
                                      integer i, integer j, integer k, integer counter) {
  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute(zone, param, counter_ud, i, j, k, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__device__ void
compute_user_defined_statistical_data_with_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const integer *counter_ud,
                                      integer i, integer j, integer mz, integer counter) {
  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute_spanwise_average(zone, param, counter_ud, i, j, mz, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}

using UserDefineStat = UDStat<USER_DEFINE_STATISTICS>;
}