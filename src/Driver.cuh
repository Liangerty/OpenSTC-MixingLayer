#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "gxl_lib/Time.h"
#include "ChemData.h"
#include "Field.h"
#include "BoundCond.cuh"
#include "FlameletLib.cuh"
#include "StatisticsCollector.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_);

  void initialize_computation();

public:
  integer myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  Parameter &parameter;
  Species spec;
  Reaction reac;
  std::vector<cfd::Field> field; // The flowfield data of the simulation. Every block is a "Field" object
  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};
  // Statistical data
  StatisticsCollector stat_collector;
};

template<class turb>
struct Driver<MixtureModel::FL, turb> {
  Driver(Parameter &parameter, Mesh &mesh_);

  void initialize_computation();
//  void simulate();

  integer myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  Parameter &parameter;
  Species spec;
  FlameletLib flameletLib;
  std::vector<cfd::Field> field;
  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};
  // Statistical data
  StatisticsCollector stat_collector;
};

void write_reference_state(const Parameter &parameter, const Species &species);

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, integer n_point_times3);
} // cfd