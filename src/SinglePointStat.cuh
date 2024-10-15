#pragma once

#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"
#include <mpi.h>

namespace cfd {

__global__ void collect_single_point_statistics(DZone *zone, DParameter *param);

class SinglePointStat {
public:
  explicit SinglePointStat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field,
                           const Species &species);

  void initialize_statistics_collector(const Species &species);

  void collect_data(DParameter *param);

  void export_statistical_data(DParameter *param, bool perform_spanwise_average);

private:
  int n_reyAve = 2;
  std::vector<std::string> reyAveVar = {"rho", "p"};
  std::vector<int> reyAveVarIndex = {0, 4};
  std::vector<int> counter_rey1st;
  int n_favAve = 4;
  std::vector<std::string> favAveVar = {"rhoU", "rhoV", "rhoW", "rhoT"};
  std::vector<int> counter_fav1st;
  int n_rey2nd = 2;
  std::vector<std::string> rey2ndVar = {"RhoRho", "pp"};
  std::vector<int> counter_rey2nd;
  int n_fav2nd = 7;
  std::vector<std::string> fav2ndVar = {"rhoUU", "rhoVV", "rhoWW", "rhoUV", "rhoUW", "rhoVW", "rhoTT"};
  std::vector<int> counter_fav2nd;

  // available statistics
  bool tke_budget = false;
  std::vector<int> counter_tke_budget;
  MPI_Offset offset_tke_budget{0};

  int myid{0};
  // Data to be bundled
  Parameter &parameter;
  const Mesh &mesh;
  std::vector<Field> &field;

  int ngg = 1;
  MPI_Offset offset_unit[4]={0, 0, 0, 0};
  std::vector<MPI_Datatype> ty_1gg, ty_0gg;

private:
  void init_stat_name(const Species &species);

  void compute_offset_for_export_data();

  void read_previous_statistical_data();
};

} // cfd