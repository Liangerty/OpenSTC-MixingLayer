#pragma once

#define USER_DEFINE_STATISTICS ThermRMS,turbulent_dissipation_rate

//,H2AirMixingLayer

#include <array>
#include <string_view>
#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"

namespace cfd {
struct ThermRMS {
  static constexpr int n_collect = 4;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"rho2", "p2", "T2", "T"};
  static constexpr int n_vol_stat_when_span_ave = 3;
  static constexpr int n_stat = 3;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"rho<sub>rms</sub>", "p<sub>rms</sub>",
                                                                     "T<sub>rms</sub>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int vol_stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int mz,
                           int counter, int span_stat_idx, int collected_idx, int vol_stat_idx);

  __device__ static void
  compute_2nd_level(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
                    int counter, int stat_idx, int collected_idx) {};
};

struct turbulent_dissipation_rate {
  static constexpr int n_collect = 10;
  static constexpr std::array<std::string_view, n_collect> namelistCollect{"u", "v", "w", "sigma11", "sigma12",
                                                                           "sigma13", "sigma22", "sigma23", "sigma33",
                                                                           "sigmaIjUi_xj"};

  static constexpr int n_vol_stat_when_span_ave = 1;
  static constexpr int n_stat = 4;
  static constexpr std::array<std::string_view, n_stat> namelistStat{"<<greek>e</greek>><sub>F</sub>",
                                                                     "<greek>h</greek>", "t<sub><greek>h</greek></sub>",
                                                                     "t<sub>turb</sub>"};

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int vol_stat_idx, int coll_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int span_stat_idx, int collected_idx, int vol_stat_idx);

  __device__ static void
  compute_2nd_level(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
                    int counter, int stat_idx, int collected_idx) {};
};

struct H2AirMixingLayer {
  static constexpr int n_collect = 12 * 3 + 1;
  static constexpr std::array<std::string_view, n_collect> namelistCollect
      {"rhoH2H2"/*0*/, "rDGradH2GradH2"/*1*/, "rDH2x"/*2*/, "rDH2y"/*3*/, "rDH2z"/*4*/, "rDH2"/*5*/, "rhoUH2"/*6*/,
       "rhoVH2"/*7*/, "rhoWH2"/*8*/, "rhoUH2H2"/*9*/, "rhoVH2H2"/*10*/, "rhoWH2H2"/*11*/,
       "rhoO2O2"/*12*/, "rDGradO2GradO2"/*13*/, "rDO2x"/*14*/, "rDO2y"/*15*/, "rDO2z"/*16*/, "rDO2"/*17*/,
       "rhoUO2"/*18*/, "rhoVO2"/*19*/, "rhoWO2"/*20*/, "rhoUO2O2"/*21*/, "rhoVO2O2"/*22*/, "rhoWO2O2"/*23*/,
       "rhoN2N2"/*24*/, "rDGradN2GradN2"/*25*/, "rDN2x"/*26*/, "rDN2y"/*27*/, "rDN2z"/*28*/, "rDN2"/*29*/,
       "rhoUN2"/*30*/, "rhoVN2"/*31*/, "rhoWN2"/*32*/, "rhoUN2N2"/*33*/, "rhoVN2N2"/*34*/, "rhoWN2N2"/*35*/,
       "mul"/*36*/
      };
  static constexpr int n_vol_stat_when_span_ave = 8 + 17 * 3 + 1;
  static constexpr std::array<std::string_view, n_vol_stat_when_span_ave> namelistStatWhenSpanAve
      {"divV"/*0*/, "nut11"/*1*/, "nut22"/*2*/, "nut33"/*3*/, "nut12"/*4*/, "nut13"/*5*/, "nut23"/*6*/, "nut"/*7*/,
          // H2
       "{H2''H2''}"/*8*/, "<<greek>c</greek>><sub>H2</sub>"/*9*/, "{u''H2''}"/*10*/, "{v''H2''}"/*11*/,
       "{w''H2''}"/*12*/, "Dt<sub>H2,x</sub>"/*13*/, "Dt<sub>H2,y</sub>"/*14*/, "Dt<sub>H2,z</sub>"/*15*/,
       "Dt<sub>H2</sub>"/*16*/, "{u''H2''<sup>2</sup>}"/*17*/, "{v''H2''<sup>2</sup>}"/*18*/,
       "{w''H2''<sup>2</sup>}"/*19*/, "Dt2<sub>H2,x</sub>"/*20*/, "Dt2<sub>H2,y</sub>"/*21*/,
       "Dt2<sub>H2,z</sub>"/*22*/, "Dt2<sub>H2</sub>"/*23*/, "rhoDH2"/*24*/,
          //O2
       "{O2''O2''}"/*25*/, "<<greek>c</greek>><sub>O2</sub>"/*26*/, "{u''O2''}"/*27*/, "{v''O2''}"/*28*/,
       "{w''O2''}"/*29*/, "Dt<sub>O2,x</sub>"/*30*/, "Dt<sub>O2,y</sub>"/*31*/, "Dt<sub>O2,z</sub>"/*32*/,
       "Dt<sub>O2</sub>"/*33*/, "{u''O2''<sup>2</sup>}"/*34*/, "{v''O2''<sup>2</sup>}"/*35*/,
       "{w''O2''<sup>2</sup>}"/*36*/, "Dt2<sub>O2,x</sub>"/*37*/, "Dt2<sub>O2,y</sub>"/*38*/,
       "Dt2<sub>O2,z</sub>"/*39*/, "Dt2<sub>O2</sub>"/*40*/, "rhoDO2"/*41*/,
          //N2
       "{N2''N2''}"/*42*/, "<<greek>c</greek>><sub>N2</sub>"/*43*/, "{u''N2''}"/*44*/, "{v''N2''}"/*45*/,
       "{w''N2''}"/*46*/, "Dt<sub>N2,x</sub>"/*47*/, "Dt<sub>N2,y</sub>"/*48*/, "Dt<sub>N2,z</sub>"/*49*/,
       "Dt<sub>N2</sub>"/*50*/, "{u''N2''<sup>2</sup>}"/*51*/, "{v''N2''<sup>2</sup>}"/*52*/,
       "{w''N2''<sup>2</sup>}"/*53*/, "Dt2<sub>N2,x</sub>"/*54*/, "Dt2<sub>N2,y</sub>"/*55*/,
       "Dt2<sub>N2,z</sub>"/*56*/, "Dt2<sub>N2</sub>"/*57*/, "rhoDN2"/*58*/,
       "mul"/*59*/
      };
  static constexpr int n_stat = 8 + 22 * 3 + 1;
  static constexpr std::array<std::string_view, n_stat> namelistStat
      {"divV"/*0*/, "nut11"/*1*/, "nut22"/*2*/, "nut33"/*3*/, "nut12"/*4*/, "nut13"/*5*/, "nut23"/*6*/,
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
       "Sct2<sub>N2,y</sub>"/*71*/, "Sct2<sub>N2,z</sub>"/*72*/, "Sct2<sub>N2</sub>"/*73*/,
       "mul"/*74*/
      };

  __device__ static void
  collect(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k, int collect_idx);

  __device__ static void
  compute(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j, int k,
          int counter, int vol_stat_idx, int collected_idx);

  __device__ static void
  compute_spanwise_average(cfd::DZone *zone, cfd::DParameter *param, const int *counter_ud, int i, int j,
                           int mz, int counter, int span_stat_idx, int collected_idx, int vol_stat_idx);

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
  constexpr static int n_vol_stat_when_span_ave = (0 + ... + stats::n_vol_stat_when_span_ave);
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
    if (param->perform_spanwise_average) {
      l += stats::n_vol_stat_when_span_ave;
    } else {
      l += stats::n_stat;
    }
    collect_idx += stats::n_collect;
  }(), ...);
}

template<typename... stats>
__device__ void
compute_user_defined_statistical_data_with_spanwise_average(cfd::DZone *zone, cfd::DParameter *param,
                                                            const int *counter_ud, int i, int j, int mz, int counter) {
  auto l = 0, collect_idx = 0, vol_stat_idx = 0;
  ([&]() {
    stats::compute_spanwise_average(zone, param, counter_ud, i, j, mz, counter, l, collect_idx, vol_stat_idx);
    vol_stat_idx += stats::n_vol_stat_when_span_ave;
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
    if (param->perform_spanwise_average) {
      l += stats::n_vol_stat_when_span_ave;
    } else {
      l += stats::n_stat;
    }
    collect_idx += stats::n_collect;
  }(), ...);
}


using UserDefineStat = UDStat<USER_DEFINE_STATISTICS>;
}
