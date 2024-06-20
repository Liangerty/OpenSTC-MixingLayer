#pragma once

#define USER_DEFINE_STATISTICS ThermRMS,turbulent_dissipation_rate,H2AirMixingLayer
//,H2_variance_and_dissipation_rate
#define SECOND_ORDER_UDSTAT H2AirMixingLayer

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

struct H2AirMixingLayer {
  static constexpr int n_collect = 18 + 9 + 9 + 3;
  static constexpr std::array<std::string_view, n_collect>
      namelistCollect{"rhoH2H2"/*0*/, "rDGradH2GradH2"/*1*/, "rDH2x"/*2*/, "rDH2y"/*3*/, "rDH2z"/*4*/, "rDH2"/*5*/,
                      "rhoUH2"/*6*/, "rhoVH2"/*7*/, "rhoWH2"/*8*/, "rhoUH2H2"/*9*/, "rhoVH2H2"/*10*/,
                      "rhoWH2H2"/*11*/, "Sc_H2"/*12*/,
                      "rhoO2O2"/*13*/, "rDGradO2GradO2"/*14*/, "rDO2x"/*15*/, "rDO2y"/*16*/, "rDO2z"/*17*/, "rDO2"/*18*/,
                      "rhoUO2"/*19*/, "rhoVO2"/*20*/, "rhoWO2"/*21*/, "rhoUO2O2"/*22*/, "rhoVO2O2"/*23*/,
                      "rhoWO2O2"/*24*/, "Sc_O2"/*25*/,
                      "rhoN2N2"/*26*/, "rDGradN2GradN2"/*27*/, "rDN2x"/*28*/, "rDN2y"/*29*/, "rDN2z"/*30*/, "rDN2"/*31*/,
                      "rhoUN2"/*32*/, "rhoVN2"/*33*/, "rhoWN2"/*34*/, "rhoUN2N2"/*35*/, "rhoVN2N2"/*36*/,
                      "rhoWN2N2"/*37*/, "Sc_N2"/*38*/,
  };
  static constexpr int n_stat = 8 + 22 * 3;
  static constexpr std::array<std::string_view, n_stat>
      namelistStat{"divV"/*0*/, "nut11"/*1*/, "nut22"/*2*/, "nut33"/*3*/, "nut12"/*4*/, "nut13"/*5*/, "nut23"/*6*/,
                   "nut"/*7*/,
      // H2
                   "{H2''H2''}"/*8*/, "<<greek>c</greek>><sub>H2</sub>"/*9*/, "t<sub>H2</sub>"/*10*/,
                   "{u''H2''}"/*11*/, "{v''H2''}"/*12*/, "{w''H2''}"/*13*/, "Dt<sub>H2,x</sub>"/*14*/,
                   "Dt<sub>H2,y</sub>"/*15*/, "Dt<sub>H2,z</sub>"/*16*/, "Dt<sub>H2</sub>"/*17*/,
                   "Sc<sub>H2</sub>"/*18*/, "Sct<sub>H2,x</sub>"/*19*/, "Sct<sub>H2,y</sub>"/*20*/,
                   "Sct<sub>H2,z</sub>"/*21*/, "Sct<sub>H2</sub>"/*22*/, "{u''H2''<sup>2</sup>}"/*23*/,
                   "{v''H2''<sup>2</sup>}"/*24*/, "{w''H2''<sup>2</sup>}"/*25*/, "Sct2<sub>H2,x</sub>"/*26*/,
                   "Sct2<sub>H2,y</sub>"/*27*/, "Sct2<sub>H2,z</sub>"/*28*/, "Sct2<sub>H2</sub>"/*29*/,
      //O2
                   "{O2''O2''}"/*30*/, "<<greek>c</greek>><sub>O2</sub>"/*31*/, "t<sub>O2</sub>"/*32*/,
                   "{u''O2''}"/*33*/, "{v''O2''}"/*34*/, "{w''O2''}"/*35*/, "Dt<sub>O2,x</sub>"/*36*/,
                   "Dt<sub>O2,y</sub>"/*37*/, "Dt<sub>O2,z</sub>"/*38*/, "Dt<sub>O2</sub>"/*39*/,
                   "Sc<sub>O2</sub>"/*40*/, "Sct<sub>O2,x</sub>"/*41*/, "Sct<sub>O2,y</sub>"/*42*/,
                   "Sct<sub>O2,z</sub>"/*43*/, "Sct<sub>O2</sub>"/*44*/, "{u''O2''<sup>2</sup>}"/*45*/,
                   "{v''O2''<sup>2</sup>}"/*46*/, "{w''O2''<sup>2</sup>}"/*47*/, "Sct2<sub>O2,x</sub>"/*48*/,
                   "Sct2<sub>O2,y</sub>"/*49*/, "Sct2<sub>O2,z</sub>"/*50*/, "Sct2<sub>O2</sub>"/*51*/,
      //N2
                   "{N2''N2''}"/*52*/, "<<greek>c</greek>><sub>N2</sub>"/*53*/, "t<sub>N2</sub>"/*54*/,
                   "{u''N2''}"/*55*/, "{v''N2''}"/*56*/, "{w''N2''}"/*57*/, "Dt<sub>N2,x</sub>"/*58*/,
                   "Dt<sub>N2,y</sub>"/*59*/, "Dt<sub>N2,z</sub>"/*60*/, "Dt<sub>N2</sub>"/*61*/,
                   "Sc<sub>N2</sub>"/*62*/, "Sct<sub>N2,x</sub>"/*63*/, "Sct<sub>N2,y</sub>"/*64*/,
                   "Sct<sub>N2,z</sub>"/*65*/, "Sct<sub>N2</sub>"/*66*/, "{u''N2''<sup>2</sup>}"/*67*/,
                   "{v''N2''<sup>2</sup>}"/*68*/, "{w''N2''<sup>2</sup>}"/*69*/, "Sct2<sub>N2,x</sub>"/*70*/,
                   "Sct2<sub>N2,y</sub>"/*71*/, "Sct2<sub>N2,z</sub>"/*72*/, "Sct2<sub>N2</sub>"/*73*/
  };

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int stat_idx, int collected_idx);

  __device__ static void
  compute_2nd_level(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
                    int counter, int stat_idx, int collected_idx);
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

template<typename... stats>
__global__ void
compute_UD_stat_data_2(DZone *zone, DParameter *param, int counter, const int *counter_ud) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto l = 0, collect_idx = 0;
  ([&]() {
    stats::compute_2nd_level(zone, param, counter_ud, i, j, k, counter, l, collect_idx);
    l += stats::n_stat;
    collect_idx += stats::n_collect;
  }(), ...);
}


using UserDefineStat = UDStat<USER_DEFINE_STATISTICS>;
}
