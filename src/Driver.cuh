#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "gxl_lib/Time.h"
#include "ChemData.h"
#include "Field.h"
#include "BoundCond.cuh"
#include "FlameletLib.cuh"
#include "SinglePointStat.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_);

  void initialize_computation();

  void deallocate();

public:
  int myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  Parameter &parameter;
  Species spec;
  Reaction reac;
  FlameletLib flameletLib;
  std::vector<Field> field; // The flowfield data of the simulation. Every block is a "Field" object
  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};
  // Statistical data
  SinglePointStat stat_collector;
//  StatisticsCollector stat_collector;
};

template<MixtureModel mix_model, class turb>
void Driver<mix_model, turb>::deallocate() {
  for (auto& f:field){
    f.deallocate_memory(parameter);
  }
}

void write_reference_state(Parameter &parameter, const Species &species);

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, int n_point_times3);
} // cfd