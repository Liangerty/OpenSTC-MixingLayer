#include "BoundCond.cuh"
#include "Parallel.h"
#include <fstream>
#include "gxl_lib/MyString.h"
#include "gxl_lib/MyAlgorithm.h"
#include "MixingLayer.cuh"
// This file should not include "algorithm", which will result in an error for gcc.
// This persuades me to write a new function "exists" in MyAlgorithm.h and MyAlgorithm.cpp.

namespace cfd {
template<typename BCType>
void register_bc(BCType *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
                 Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(BCType));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      BCType bound_cond(bc_name, parameter);
      cudaMemcpy(&(bc[i]), &bound_cond, sizeof(BCType), cudaMemcpyHostToDevice);
      break;
    }
  }
}

template<>
void register_bc<Wall>(Wall *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
                       Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Wall));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      Wall wall(this_bc, parameter);
      bc_info[i].label = bc_label;
      cudaMemcpy(&(bc[i]), &wall, sizeof(Wall), cudaMemcpyHostToDevice);
    }
  }
}

template<>
void register_bc<Inflow>(Inflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
                         Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Inflow));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      Inflow inflow(bc_name, species, parameter);
      inflow.copy_to_gpu(&(bc[i]), species, parameter);
      break;
    }
  }
}

template<>
void
register_bc<FarField>(FarField *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
                      Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(FarField));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      FarField farfield(bc_name, species, parameter);
      farfield.copy_to_gpu(&(bc[i]), species, parameter);
      break;
    }
  }
}

template<>
void register_bc<SubsonicInflow>(SubsonicInflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
                                 Species &species, Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(SubsonicInflow));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      SubsonicInflow subsonic_inflow(bc_name, parameter);
      subsonic_inflow.copy_to_gpu(&(bc[i]), species, parameter);
      break;
    }
  }
}

template<>
void
register_bc<BackPressure>(BackPressure *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
                          Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(BackPressure));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      BackPressure back_pressure(bc_name, parameter);
      cudaMemcpy(&(bc[i]), &back_pressure, sizeof(BackPressure), cudaMemcpyHostToDevice);
      break;
    }
  }
}

void DBoundCond::initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter,
                                      DParameter *param) {
  std::vector<int> bc_labels;
  // Count the number of distinct boundary conditions
  for (auto i = 0; i < mesh.n_block; i++) {
    for (auto &b: mesh[i].boundary) {
      auto lab = b.type_label;
      bool has_this_bc = false;
      for (auto l: bc_labels) {
        if (l == lab) {
          has_this_bc = true;
          break;
        }
      }
      if (!has_this_bc) {
        bc_labels.push_back(lab);
      }
    }
  }
  // Initialize the inflow and wall conditions which are different among cases.
  std::vector<int> wall_idx, symmetry_idx, inflow_idx, outflow_idx, farfield_idx, subsonic_inflow_idx, back_pressure_idx
    , periodic_idx;
  auto &bcs = parameter.get_string_array("boundary_conditions");
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto label = std::get<int>(bc.at("label"));

    auto this_iter = bc_labels.end();
    for (auto iter = bc_labels.begin(); iter != bc_labels.end(); ++iter) {
      if (*iter == label) {
        this_iter = iter;
        break;
      }
    }
    if (this_iter != bc_labels.end()) {
      bc_labels.erase(this_iter);
      auto type = std::get<std::string>(bc.at("type"));
      if (type == "wall") {
        wall_idx.push_back(label);
        ++n_wall;
      } else if (type == "inflow") {
        inflow_idx.push_back(label);
        ++n_inflow;
      }
      // Note: Normally, this would not happen for outflow, symmetry, and periodic conditions.
      // Because the above-mentioned conditions normally do not need to specify special treatments.
      // If we need to add support for these conditions, then we add them here.
      else if (type == "outflow") {
        outflow_idx.push_back(label);
        ++n_outflow;
      } else if (type == "symmetry") {
        symmetry_idx.push_back(label);
        ++n_symmetry;
      } else if (type == "farfield") {
        farfield_idx.push_back(label);
        ++n_farfield;
      } else if (type == "subsonic_inflow") {
        subsonic_inflow_idx.push_back(label);
        ++n_subsonic_inflow;
      } else if (type == "back_pressure") {
        back_pressure_idx.push_back(label);
        ++n_back_pressure;
      } else if (type == "periodic") {
        periodic_idx.push_back(label);
        ++n_periodic;
      }
    }
  }
  for (int lab: bc_labels) {
    if (lab == 2) {
      wall_idx.push_back(lab);
      ++n_wall;
    } else if (lab == 3) {
      symmetry_idx.push_back(lab);
      ++n_symmetry;
    } else if (lab == 4) {
      farfield_idx.push_back(lab);
      ++n_farfield;
    } else if (lab == 5) {
      inflow_idx.push_back(lab);
      ++n_inflow;
    } else if (lab == 6) {
      outflow_idx.push_back(lab);
      ++n_outflow;
    } else if (lab == 7) {
      subsonic_inflow_idx.push_back(lab);
      ++n_subsonic_inflow;
    } else if (lab == 8) {
      periodic_idx.push_back(lab);
      ++n_periodic;
    } else if (lab == 9) {
      back_pressure_idx.push_back(lab);
      ++n_back_pressure;
    }
  }

  // Read specific conditions
  // We always first initialize the Farfield and Inflow conditions, because they may set the reference values.
  register_bc<FarField>(farfield, n_farfield, farfield_idx, farfield_info, species, parameter);
  register_bc<Inflow>(inflow, n_inflow, inflow_idx, inflow_info, species, parameter);
  register_bc<SubsonicInflow>(subsonic_inflow, n_subsonic_inflow, subsonic_inflow_idx, subsonic_inflow_info, species,
                              parameter);
  register_bc<Wall>(wall, n_wall, wall_idx, wall_info, species, parameter);
  register_bc<Symmetry>(symmetry, n_symmetry, symmetry_idx, symmetry_info, species, parameter);
  register_bc<Outflow>(outflow, n_outflow, outflow_idx, outflow_info, species, parameter);
  register_bc<BackPressure>(back_pressure, n_back_pressure, back_pressure_idx, back_pressure_info, species, parameter);
  register_bc<Periodic>(periodic, n_periodic, periodic_idx, periodic_info, species, parameter);

  link_bc_to_boundaries(mesh, field);

  MpiParallel::barrier();

  initialize_profile_and_rng(parameter, mesh, species, field, param);

  df_label.resize(n_inflow, -1);
  initialize_digital_filter(parameter, mesh);

  printf("\tProcess [[%d]] has finished setting up boundary conditions.\n", parameter.get_int("myid"));
}

void DBoundCond::link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const {
  const int n_block{mesh.n_block};
  auto **i_wall = new int *[n_wall];
  for (size_t i = 0; i < n_wall; i++) {
    i_wall[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_wall[i][j] = 0;
    }
  }
  auto **i_symm = new int *[n_symmetry];
  for (size_t i = 0; i < n_symmetry; i++) {
    i_symm[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_symm[i][j] = 0;
    }
  }
  auto **i_farfield = new int *[n_farfield];
  for (size_t i = 0; i < n_farfield; ++i) {
    i_farfield[i] = new int[n_block];
    for (int j = 0; j < n_block; ++j) {
      i_farfield[i][j] = 0;
    }
  }
  auto **i_inflow = new int *[n_inflow];
  for (size_t i = 0; i < n_inflow; i++) {
    i_inflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_inflow[i][j] = 0;
    }
  }
  auto **i_outflow = new int *[n_outflow];
  for (size_t i = 0; i < n_outflow; i++) {
    i_outflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_outflow[i][j] = 0;
    }
  }
  auto **i_subsonic_inflow = new int *[n_subsonic_inflow];
  for (size_t i = 0; i < n_subsonic_inflow; i++) {
    i_subsonic_inflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_subsonic_inflow[i][j] = 0;
    }
  }
  auto **i_back_pressure = new int *[n_back_pressure];
  for (size_t i = 0; i < n_back_pressure; i++) {
    i_back_pressure[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_back_pressure[i][j] = 0;
    }
  }
  auto **i_periodic = new int *[n_periodic];
  for (size_t i = 0; i < n_periodic; i++) {
    i_periodic[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_periodic[i][j] = 0;
    }
  }

  // We first count how many faces corresponds to a given boundary condition
  for (int i = 0; i < n_block; i++) {
    count_boundary_of_type_bc(mesh[i].boundary, n_wall, i_wall, i, n_block, wall_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_symmetry, i_symm, i, n_block, symmetry_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_farfield, i_farfield, i, n_block, farfield_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_inflow, i_inflow, i, n_block, inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_outflow, i_outflow, i, n_block, outflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_subsonic_inflow, i_subsonic_inflow, i, n_block,
                              subsonic_inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_back_pressure, i_back_pressure, i, n_block, back_pressure_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_periodic, i_periodic, i, n_block, periodic_info);
  }
  for (size_t l = 0; l < n_wall; l++) {
    wall_info[l].boundary = new int2[wall_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_symmetry; ++l) {
    symmetry_info[l].boundary = new int2[symmetry_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_farfield; ++l) {
    farfield_info[l].boundary = new int2[farfield_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_inflow; l++) {
    inflow_info[l].boundary = new int2[inflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_outflow; l++) {
    outflow_info[l].boundary = new int2[outflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_subsonic_inflow; ++l) {
    subsonic_inflow_info[l].boundary = new int2[subsonic_inflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_back_pressure; ++l) {
    back_pressure_info[l].boundary = new int2[back_pressure_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_periodic; ++l) {
    periodic_info[l].boundary = new int2[periodic_info[l].n_boundary];
  }

  const auto ngg{mesh[0].ngg};
  for (auto i = 0; i < n_block; i++) {
    link_boundary_and_condition(mesh[i].boundary, wall_info, n_wall, i_wall, i);
    link_boundary_and_condition(mesh[i].boundary, symmetry_info, n_symmetry, i_symm, i);
    link_boundary_and_condition(mesh[i].boundary, farfield_info, n_farfield, i_farfield, i);
    link_boundary_and_condition(mesh[i].boundary, inflow_info, n_inflow, i_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, outflow_info, n_outflow, i_outflow, i);
    link_boundary_and_condition(mesh[i].boundary, subsonic_inflow_info, n_subsonic_inflow, i_subsonic_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, back_pressure_info, n_back_pressure, i_back_pressure, i);
    link_boundary_and_condition(mesh[i].boundary, periodic_info, n_periodic, i_periodic, i);
  }
  //  for (auto i = 0; i < n_block; i++) {
  //    for (size_t l = 0; l < n_wall; l++) {
  //      const auto nb = wall_info[l].n_boundary;
  //      for (size_t m = 0; m < nb; m++) {
  //        auto i_zone = wall_info[l].boundary[m].x;
  //        if (i_zone != i) {
  //          continue;
  //        }
  //        auto &b = mesh[i].boundary[wall_info[l].boundary[m].y];
  //        for (int q = 0; q < 3; ++q) {
  //          if (q == b.face) continue;
  //          b.range_start[q] += ngg;
  //          b.range_end[q] -= ngg;
  //        }
  //      }
  //    }
  //    cudaMemcpy(field[i].h_ptr->boundary, mesh[i].boundary.data(), mesh[i].boundary.size() * sizeof(Boundary),
  //               cudaMemcpyHostToDevice);
  //  }
  for (int i = 0; i < n_wall; i++) {
    delete[]i_wall[i];
  }
  for (int i = 0; i < n_symmetry; i++) {
    delete[]i_symm[i];
  }
  for (int i = 0; i < n_farfield; ++i) {
    delete[]i_farfield[i];
  }
  for (int i = 0; i < n_inflow; i++) {
    delete[]i_inflow[i];
  }
  for (int i = 0; i < n_outflow; i++) {
    delete[]i_outflow[i];
  }
  for (int i = 0; i < n_subsonic_inflow; ++i) {
    delete[]i_subsonic_inflow[i];
  }
  for (int i = 0; i < n_back_pressure; ++i) {
    delete[]i_back_pressure[i];
  }
  for (int i = 0; i < n_periodic; ++i) {
    delete[]i_periodic[i];
  }
  delete[]i_wall;
  delete[]i_symm;
  delete[]i_farfield;
  delete[]i_inflow;
  delete[]i_outflow;
  delete[]i_subsonic_inflow;
  delete[]i_back_pressure;
  delete[]i_periodic;
}

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, int n_bc, int **sep, int blk_idx, int n_block,
                               BCInfo *bc_info) {
  if (n_bc <= 0) {
    return;
  }

  // Count how many faces correspond to the given bc
  const auto n_boundary{boundary.size()};
  auto *n = new int[n_bc];
  memset(n, 0, sizeof(int) * n_bc);
  for (size_t l = 0; l < n_bc; l++) {
    int label = bc_info[l].label; // This means every bc should have a member "label"
    for (size_t i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        ++bc_info[l].n_boundary;
        ++n[l];
      }
    }
  }
  if (blk_idx < n_block - 1) {
    for (size_t l = 0; l < n_bc; l++) {
      sep[l][blk_idx + 1] = n[l] + sep[l][blk_idx];
    }
  }
  delete[]n;
}

void link_boundary_and_condition(const std::vector<Boundary> &boundary, const BCInfo *bc, int n_bc, int **sep,
                                 int i_zone) {
  const auto n_boundary{boundary.size()};
  for (size_t l = 0; l < n_bc; l++) {
    int label = bc[l].label;
    int has_read{sep[l][i_zone]};
    for (auto i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        bc[l].boundary[has_read] = make_int2(i_zone, i);
        ++has_read;
      }
    }
  }
}

void Inflow::copy_to_gpu(Inflow *d_inflow, Species &spec, const Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  real *h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);
  if (inflow_type == 2) {
    // For mixing layer flow, there are another group of sv.
    real *h_sv_lower = new real[n_scalar];
    for (int l = 0; l < n_scalar; ++l) {
      h_sv_lower[l] = sv_lower[l];
    }
    delete[]sv_lower;
    cudaMalloc(&sv_lower, n_scalar * sizeof(real));
    cudaMemcpy(sv_lower, h_sv_lower, n_scalar * sizeof(real), cudaMemcpyHostToDevice);
  }
  if (fluctuation_type == 3) {
    // For the case of fluctuation_type == 3, we need to copy the fluctuation field to GPU.
    real *h_rand_values = new real[199];
    for (int l = 0; l < 199; ++l) {
      h_rand_values[l] = random_phase[l];
    }
    cudaMalloc(&random_phase, sizeof(real) * 199);
    cudaMemcpy(random_phase, h_rand_values, sizeof(real) * 199, cudaMemcpyHostToDevice);
  }

  cudaMemcpy(d_inflow, this, sizeof(Inflow), cudaMemcpyHostToDevice);
}

void FarField::copy_to_gpu(FarField *d_farfield, Species &spec, const Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  real *h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);

  cudaMemcpy(d_farfield, this, sizeof(FarField), cudaMemcpyHostToDevice);
}

void SubsonicInflow::copy_to_gpu(cfd::SubsonicInflow *d_inflow, cfd::Species &spec, const cfd::Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  real *h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);

  cudaMemcpy(d_inflow, this, sizeof(SubsonicInflow), cudaMemcpyHostToDevice);
}

void initialize_profile_with_inflow(const Boundary &boundary, const Block &block, const Parameter &parameter,
                                    const Species &species, ggxl::VectorField3D<real> &profile,
                                    const std::string &profile_related_bc_name) {
  const int direction = boundary.face;
  const int extent[3]{block.mx, block.my, block.mz};
  const int ngg = block.ngg;
  int range_0[2]{-ngg, block.mx + ngg - 1}, range_1[2]{-ngg, block.my + ngg - 1},
      range_2[2]{-ngg, block.mz + ngg - 1};
  int n0 = extent[0], n1 = extent[1], n2 = extent[2];
  //  if (direction == 1) {
  //    n1 = extent[0];
  //    range_1[1] = block.mx + ngg - 1;
  //  } else if (direction == 2) {
  //    n1 = extent[0];
  //    n2 = extent[1];
  //    range_1[1] = block.mx + ngg - 1;
  //    range_2[1] = block.my + ngg - 1;
  //  }
  if (direction == 0) {
    n0 = 1 + ngg;
    range_0[0] = 0;
    range_0[1] = ngg;
  } else if (direction == 1) {
    n1 = 1 + ngg;
    range_1[0] = 0;
    range_1[1] = ngg;
  } else if (direction == 2) {
    n2 = 1 + ngg;
    range_2[0] = 0;
    range_2[1] = ngg;
  }

  ggxl::VectorField3DHost<real> profile_host;
  const int n_var = parameter.get_int("n_var");
  profile_host.resize(n0, n1, n2, n_var + 1, ngg);
  Inflow inflow1(profile_related_bc_name, species, parameter);
  for (int i0 = range_0[0]; i0 <= range_0[1]; ++i0) {
    for (int i1 = range_1[0]; i1 <= range_1[1]; ++i1) {
      for (int i2 = range_2[0]; i2 <= range_2[1]; ++i2) {
        profile_host(i0, i1, i2, 0) = inflow1.density;
        profile_host(i0, i1, i2, 1) = inflow1.u;
        profile_host(i0, i1, i2, 2) = inflow1.v;
        profile_host(i0, i1, i2, 3) = inflow1.w;
        profile_host(i0, i1, i2, 4) = inflow1.pressure;
        profile_host(i0, i1, i2, 5) = inflow1.temperature;
        for (int i = 0; i < species.n_spec; ++i) {
          profile_host(i0, i1, i2, 6 + i) = inflow1.sv[i];
        }
      }
    }
  }
  profile.allocate_memory(n0, n1, n2, n_var + 1, ngg);
  cudaMemcpy(profile.data(), profile_host.data(), sizeof(real) * profile_host.size() * (n_var + 1),
             cudaMemcpyHostToDevice);
}


void init_mixingLayer_prof_compatible_cpu(const Boundary &boundary, Parameter &parameter, const Block &b,
                                          const Species &species,
                                          ggxl::VectorField3D<real> &profile) {
  const int direction = boundary.face;
  const int ngg = b.ngg;
  const int mx = b.mx, my = b.my, mz = b.mz;
  int range_x[2]{-ngg, mx + ngg - 1}, range_y[2]{-ngg, my + ngg - 1},
      range_z[2]{-ngg, mz + ngg - 1};
  int n0 = mx + 2 * ngg, n1 = my + 2 * ngg, n2 = mz + 2 * ngg;
  if (direction == 0) {
    n0 = 1 + ngg;
    range_x[0] = -ngg;
    range_x[1] = 0;
  } else if (direction == 1) {
    n1 = 1 + ngg;
    range_y[0] = -ngg;
    range_y[1] = 0;
  } else if (direction == 2) {
    n2 = 1 + ngg;
    range_z[0] = -ngg;
    range_z[1] = 0;
  }

  std::vector<real> var_info;
  get_mixing_layer_info(parameter, species, var_info);

  ggxl::VectorField3DHost<real> profile_host;
  int extent[3]{mx, my, mz};
  extent[direction] = 1;
  int nv = parameter.get_int("n_var");
  profile_host.resize(extent[0], extent[1], extent[2], nv + 1, ngg);

  const int n_spec = species.n_spec;
  const real delta_omega = parameter.get_real("delta_omega");
  int n_fl{0};
  if ((species.n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2)
    n_fl = 2;
  const int i_fl = parameter.get_int("i_fl");
  const int n_ps = parameter.get_int("n_ps");
  const int i_ps = parameter.get_int("i_ps");
  const int n_turb = parameter.get_int("n_turb");
  const int n_scalar = parameter.get_int("n_scalar");

  for (int j = range_y[0]; j <= range_y[1]; ++j) {
    real y = b.y(0, j, 0);
    const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
    real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / delta_omega);
    real p = var_info[4];
    real density, t;
    const real t_upper = var_info[5], t_lower = var_info[12 + n_spec];
    real yk[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER + 4] = {};

    real y_upper = (u - u_lower) / (u_upper - u_lower);
    real y_lower = 1 - y_upper;

    if (n_spec > 0) {
      // multi-species
      auto sv_upper = &var_info[6];
      auto sv_lower = &var_info[7 + n_spec + 6];

      for (int l = 0; l < n_spec; ++l) {
        yk[l] = y_upper * sv_upper[l] + y_lower * sv_lower[l];
      }

      // compute the total enthalpy of upper and lower streams
      real h0_upper{0.5 * u_upper * u_upper}, h0_lower{0.5 * u_lower * u_lower};

      real h_upper[MAX_SPEC_NUMBER], h_lower[MAX_SPEC_NUMBER];
      species.compute_enthalpy(t_upper, h_upper);
      species.compute_enthalpy(t_lower, h_lower);

      real mw_inv = 0;
      for (int l = 0; l < n_spec; ++l) {
        h0_upper += yk[l] * h_upper[l];
        h0_lower += yk[l] * h_lower[l];
        mw_inv += yk[l] / species.mw[l];
      }

      real h = y_upper * h0_upper + y_lower * h0_lower;
      h -= 0.5 * u * u;

      auto hs = h_upper, cps = h_lower;
      real err{1};
      t = t_upper * y_upper + t_lower * y_lower;
      constexpr int max_iter{1000};
      constexpr real eps{1e-3};
      int iter = 0;

      while (err > eps && iter++ < max_iter) {
        species.compute_enthalpy_and_cp(t, hs, cps);
        real cp{0}, h_new{0};
        for (int l = 0; l < n_spec; ++l) {
          cp += yk[l] * cps[l];
          h_new += yk[l] * hs[l];
        }
        const real t1 = t - (h_new - h) / cp;
        err = abs(1 - t1 / t);
        t = t1;
      }
      density = p / (R_u * mw_inv * t);

      if (n_fl > 0) {
        yk[i_fl] = var_info[6 + n_spec] * y_upper + var_info[13 + n_spec + n_spec] * y_lower;
        yk[i_fl + 1] = 0;
      }
      if (n_ps > 0) {
        for (int l = 0; l < n_ps; ++l) {
          yk[i_ps + l] =
              var_info[14 + 2 * n_spec + 4 + 2 * l] * y_upper + var_info[14 + 2 * n_spec + 4 + 2 * l + 1] * y_lower;
        }
      }
    } else {
      // Air
      constexpr real cp = gamma_air * R_air / (gamma_air - 1);
      real h0_upper = 0.5 * u_upper * u_upper + cp * var_info[5];
      real h0_lower = 0.5 * u_lower * u_lower + cp * var_info[12 + n_spec];
      real h = y_upper * h0_upper + y_lower * h0_lower - 0.5 * u * u;
      t = h / cp;
      density = p / (R_air * t);

      if (n_ps > 0) {
        if (y > 0) {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l];
          }
        } else {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l + 1];
          }
        }
      }
    }
    if (n_turb > 0) {
      if (y > 0) {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + 1 + l];
        }
      } else {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + n_turb + 1 + l];
        }
      }
    }

    auto &prof = profile_host;
    for (int k = range_z[0]; k <= range_z[1]; ++k) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        prof(i, j, k, 0) = density;
        prof(i, j, k, 1) = u;
        prof(i, j, k, 2) = 0;
        prof(i, j, k, 3) = 0;
        prof(i, j, k, 4) = p;
        prof(i, j, k, 5) = t;
        for (int l = 0; l < n_scalar; ++l) {
          prof(i, j, k, 6 + l) = yk[l];
        }
      }
    }
  }

  profile.allocate_memory(extent[0], extent[1], extent[2], nv + 1, ngg);
  cudaMemcpy(profile.data(), profile_host.data(), profile_host.size() * (nv + 1) * sizeof(real),
             cudaMemcpyHostToDevice);

  // Write to file
  FILE *fp = fopen("./mixingLayerProfile.dat", "w");
  if (fp == nullptr) {
    printf("Cannot open file %s\n", "./mixingLayerProfile.dat");
    MpiParallel::exit();
  }
  fprintf(fp, "VARIABLES = \"X\"\n\"Y\"\n\"Z\"\n\"RHO\"\n\"U\"\n\"V\"\n\"W\"\n\"p\"\n\"T\"\n");
  if (n_spec > 0) {
    for (int l = 0; l < n_spec; ++l) {
      fprintf(fp, "\"%s\"\n", species.spec_name[l].c_str());
    }
  }
  if (n_turb == 2) {
    fprintf(fp, "\"TKE\"\n\"omega\"\n");
  }
  if (n_fl > 0) {
    fprintf(fp, "\"MixtureFraction\"\n\"MixtureFractionVariance\"\n");
  }
  fprintf(fp, "ZONE T=\"INFLOW\"\n");
  fprintf(fp, "I=%d, J=%d, K=%d, f=BLOCK\n", n0, n1, n2);
  // Print nv DOUBLE in DT=(...) into a char array
  std::string temp = "DT=(";
  for (int l = 0; l < nv + 4; ++l) {
    temp += "DOUBLE ";
  }
  temp += ")\n";
  fprintf(fp, "%s", temp.c_str());
  // First, x, y, z
  int everyFour = 0;
  constexpr int numberToChangeLine = 4;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.x(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  everyFour = 0;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.y(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  everyFour = 0;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.z(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  // Then, the variables
  for (int l = 0; l < nv + 1; ++l) {
    everyFour = 0;
    for (int k = range_z[0]; k <= range_z[1]; ++k) {
      for (int j = range_y[0]; j <= range_y[1]; ++j) {
        for (int i = range_x[0]; i <= range_x[1]; ++i) {
          fprintf(fp, " %.15e", profile_host(i, j, k, l));
          everyFour++;
          if (everyFour == numberToChangeLine) {
            fprintf(fp, "\n");
            everyFour = 0;
          }
        }
      }
    }
    if (everyFour != 0) {
      fprintf(fp, "\n");
    }
  }
  fclose(fp);

  profile_host.deallocate_memory();
}

int read_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
                 const Species &species, ggxl::VectorField3D<real> &profile,
                 const std::string &profile_related_bc_name) {
  if (file == "MYSELF") {
    // The profile is initialized by the inflow condition.
    initialize_profile_with_inflow(boundary, block, parameter, species, profile, profile_related_bc_name);
    return 0;
  }
  if (file == "mixingLayerProfile.dat") {
    // The profile is initialized by the inflow condition.
    init_mixingLayer_prof_compatible_cpu(boundary, parameter, block, species, profile);
    return 1;
  }
  auto dot = file.find_last_of('.');
  auto suffix = file.substr(dot + 1, file.size());
  if (suffix == "dat") {
    read_dat_profile(boundary, file, block, parameter, species, profile, profile_related_bc_name);
  } else if (suffix == "plt") {
    //    read_plt_profile();
  }
  return 0;
}

std::vector<int>
identify_variable_labels(const cfd::Parameter &parameter, std::vector<std::string> &var_name, const Species &species,
                         bool &has_pressure, bool &has_temperature, bool &has_tke) {
  std::vector<int> labels;
  const int n_spec = species.n_spec;
  const int n_turb = parameter.get_int("n_turb");
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0, and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0;
    } else if (n == "U") {
      l = 1;
    } else if (n == "V") {
      l = 2;
    } else if (n == "W") {
      l = 3;
    } else if (n == "P" || n == "PRESSURE") {
      l = 4;
      has_pressure = true;
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5;
      has_temperature = true;
    } else {
      if (n_spec > 0) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label;
            break;
          }
        }
        if (n == "MIXTUREFRACTION") {
          // Mixture fraction
          l = 6 + n_spec + n_turb;
        } else if (n == "MIXTUREFRACTIONVARIANCE") {
          // Mixture fraction variance
          l = 6 + n_spec + n_turb + 1;
        }
      }
      if (n_turb > 0) {
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K" || n == "TKE") { // turbulent kinetic energy
          if (n_turb == 2) {
            l = 6 + n_spec;
            has_tke = true;
          }
        } else if (n == "OMEGA") { // specific dissipation rate
          if (n_turb == 2) {
            l = 6 + n_spec + 1;
          }
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          if (n_turb == 1) {
            l = 6 + n_spec;
          }
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

void
read_dat_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
                 const Species &species, ggxl::VectorField3D<real> &profile,
                 const std::string &profile_related_bc_name) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    MpiParallel::exit();
  }
  const int direction = boundary.face;
  const int n_spec = species.n_spec;

  std::string input;
  std::vector<std::string> var_name;
  gxl::read_until(file_in, input, "VARIABLES", gxl::Case::upper);
  while (!(input.substr(0, 4) == "ZONE" || input.substr(0, 5) == " zone")) {
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    auto equal = input.find('=');
    if (equal != std::string::npos)
      input.erase(0, equal + 1);
    std::istringstream line(input);
    std::string v_name;
    while (line >> v_name) {
      var_name.emplace_back(v_name);
    }
    gxl::getline(file_in, input, gxl::Case::upper);
  }
  auto label_order = parameter.identify_variable_labels(var_name, species);

  bool has_pressure = gxl::exists(label_order, 4);
  bool has_temperature = gxl::exists(label_order, 5);
  bool has_tke = gxl::exists(label_order, 6 + n_spec);

  if ((!has_temperature) && (!has_pressure)) {
    printf("The temperature or pressure is not given in the profile, please provide at least one of them!\n");
    MpiParallel::exit();
  }
  real turb_viscosity_ratio{0}, turb_intensity{0};
  if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
    auto &info = parameter.get_struct(profile_related_bc_name);
    if (info.find("turb_viscosity_ratio") == info.end() || info.find("turbulence_intensity") == info.end()) {
      printf(
        "The turbulence intensity or turbulent viscosity ratio is not given in the profile, please provide both of them!\n");
      MpiParallel::exit();
    }
    turb_viscosity_ratio = std::get<real>(info.at("turb_viscosity_ratio"));
    turb_intensity = std::get<real>(info.at("turbulence_intensity"));
  }

  int mx, my, mz;
  bool i_read{false}, j_read{false}, k_read{false}, packing_read{false};
  std::string key;
  std::string data_packing{"POINT"};
  while (!(i_read && j_read && k_read && packing_read)) {
    std::getline(file_in, input);
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    gxl::replace(input, '=', ' ');
    std::istringstream line(input);
    while (line >> key) {
      if (key == "i" || key == "I") {
        line >> mx;
        i_read = true;
      } else if (key == "j" || key == "J") {
        line >> my;
        j_read = true;
      } else if (key == "k" || key == "K") {
        line >> mz;
        k_read = true;
      } else if (key == "f" || key == "DATAPACKING" || key == "datapacking") {
        line >> data_packing;
        data_packing = gxl::to_upper(data_packing);
        packing_read = true;
      }
    }
  }

  // This line is the DT=(double ...) line, which must exist if we output the data from Tecplot.
  std::getline(file_in, input);

  int extent[3]{mx, my, mz};

  // Then we read the variables.
  auto nv_read = static_cast<int>(var_name.size());
  gxl::VectorField3D<real> profile_read;
  profile_read.resize(extent[0], extent[1], extent[2], nv_read, 0);

  if (data_packing == "POINT") {
    for (int k = 0; k < extent[2]; ++k) {
      for (int j = 0; j < extent[1]; ++j) {
        for (int i = 0; i < extent[0]; ++i) {
          for (int l = 0; l < nv_read; ++l) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  } else if (data_packing == "BLOCK") {
    for (int l = 0; l < nv_read; ++l) {
      for (int k = 0; k < extent[2]; ++k) {
        for (int j = 0; j < extent[1]; ++j) {
          for (int i = 0; i < extent[0]; ++i) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  }

  const int n_var = parameter.get_int("n_var");
  const int n_scalar = parameter.get_int("n_scalar");

  const auto ngg = block.ngg;
  int range_i[2]{-ngg, block.mx + ngg - 1},
      range_j[2]{-ngg, block.my + ngg - 1},
      range_k[2]{-ngg, block.mz + ngg - 1};
  if (direction == 0) {
    // i direction
    if (boundary.direction == 1) {
      range_i[0] = block.mx - 1;
      range_i[1] = block.mx + ngg - 1;
    } else {
      range_i[0] = -ngg;
      range_i[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    // The 2*ngg ghost layers in x direction are not used.
    profile_to_match.resize(1 + ngg, block.my, block.mz, n_var + 1, ngg);
    //    ggxl::VectorField2DHost<real> profile_to_match;
    //    profile_to_match.allocate_memory(block.my, block.mz, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        for (int ic = 0; ic <= ngg; ++ic) {
          int i = range_i[0] + ic;
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(ic, j, k, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(ic, j, k, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(ic, j, k, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(ic, j, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(ic, j, k, 5) = profile_to_match(ic, j, k, 4) * mw / (R_u * profile_to_match(ic, j, k, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(ic, j, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(ic, j, k, 4) = profile_to_match(ic, j, k, 5) * R_u * profile_to_match(ic, j, k, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu{};
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(ic, j, k, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(ic, j, k, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(ic, j, k, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(ic, j, k, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(ic, j, k, 1) * profile_to_match(ic, j, k, 1) +
                              profile_to_match(ic, j, k, 2) * profile_to_match(ic, j, k, 2) +
                              profile_to_match(ic, j, k, 3) * profile_to_match(ic, j, k, 3);
            profile_to_match(ic, j, k, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(ic, j, k, 6 + n_spec + 1) =
                profile_to_match(ic, j, k, 0) * profile_to_match(ic, j, k, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(1 + ngg, block.my, block.mz, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 1) {
    // j direction
    auto face = boundary.direction;
    if (face == 1) {
      range_j[0] = block.my - 1;
      range_j[1] = block.my + ngg - 1;
    } else {
      range_j[0] = -ngg;
      range_j[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, 1 + ngg, block.mz, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int jc = 0; jc <= ngg; ++jc) {
        for (int i = range_i[0]; i <= range_i[1]; ++i) {
          int j = range_j[0] + jc;
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(i, jc, k, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(i, jc, k, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(i, jc, k, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, jc, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, jc, k, 5) = profile_to_match(i, jc, k, 4) * mw / (R_u * profile_to_match(i, jc, k, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, jc, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, jc, k, 4) = profile_to_match(i, jc, k, 5) * R_u * profile_to_match(i, jc, k, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu;
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(i, jc, k, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(i, jc, k, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(i, jc, k, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(i, jc, k, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(i, jc, k, 1) * profile_to_match(i, jc, k, 1) +
                              profile_to_match(i, jc, k, 2) * profile_to_match(i, jc, k, 2) +
                              profile_to_match(i, jc, k, 3) * profile_to_match(i, jc, k, 3);
            profile_to_match(i, jc, k, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(i, jc, k, 6 + n_spec + 1) =
                profile_to_match(i, jc, k, 0) * profile_to_match(i, jc, k, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, 1 + ngg, block.mz, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 2) {
    // k direction
    auto face = boundary.direction;
    if (face == 1) {
      range_k[0] = block.mz - 1;
      range_k[1] = block.mz + ngg - 1;
    } else {
      range_k[0] = -ngg;
      range_k[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, block.my, 1 + ngg, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int kc = 0; kc <= ngg; ++kc) {
      int k = range_k[0] + kc;
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        for (int i = range_i[0]; i <= range_i[1]; ++i) {
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(i, j, kc, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(i, j, kc, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(i, j, kc, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, j, kc, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, j, kc, 5) = profile_to_match(i, j, kc, 4) * mw / (R_u * profile_to_match(i, j, kc, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, j, kc, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, j, kc, 4) = profile_to_match(i, j, kc, 5) * R_u * profile_to_match(i, j, kc, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu;
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(i, j, kc, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(i, j, kc, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(i, j, kc, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(i, j, kc, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(i, j, kc, 1) * profile_to_match(i, j, kc, 1) +
                              profile_to_match(i, j, kc, 2) * profile_to_match(i, j, kc, 2) +
                              profile_to_match(i, j, kc, 3) * profile_to_match(i, j, kc, 3);
            profile_to_match(i, j, kc, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(i, j, kc, 6 + n_spec + 1) =
                profile_to_match(i, j, kc, 0) * profile_to_match(i, j, kc, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, block.my, 1 + ngg, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  }
}

__global__ void
initialize_rng(curandState *rng_states, int size, int64_t time_stamp) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= size)
    return;

  curand_init(time_stamp + i, i, 0, &rng_states[i]);
}

void read_lst_profile(const Boundary &boundary, const std::string &file, const Block &block, const Parameter &parameter,
                      const Species &species, ggxl::VectorField3D<real> &profile,
                      const std::string &profile_related_bc_name) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    MpiParallel::exit();
  }
  const int direction = boundary.face;

  std::string input;
  std::vector<std::string> var_name;
  gxl::read_until(file_in, input, "VARIABLES", gxl::Case::upper);
  while (!(input.substr(0, 4) == "ZONE" || input.substr(0, 5) == " zone")) {
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    auto equal = input.find('=');
    if (equal != std::string::npos)
      input.erase(0, equal + 1);
    std::istringstream line(input);
    std::string v_name;
    while (line >> v_name) {
      var_name.emplace_back(v_name);
    }
    gxl::getline(file_in, input, gxl::Case::upper);
  }
  bool has_pressure{false}, has_temperature{false}, has_rho{false};
  // Identify the labels of the variables
  std::vector<int> label_order;
  const int n_spec = species.n_spec;
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0, and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "RHOR") {
      l = 0;
      has_rho = true;
    } else if (n == "UR") {
      l = 1;
    } else if (n == "VR") {
      l = 2;
    } else if (n == "WR") {
      l = 3;
    } else if (n == "PR") {
      l = 4;
      has_pressure = true;
    } else if (n == "TR") {
      l = 5;
      has_temperature = true;
    } else if (n == "RHOI") {
      l = 6;
    } else if (n == "UI") {
      l = 7;
    } else if (n == "VI") {
      l = 8;
    } else if (n == "WI") {
      l = 9;
    } else if (n == "PI") {
      l = 10;
    } else if (n == "TI") {
      l = 11;
    } else {
      if (n_spec > 0) {
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec) + "R") {
            l = 12 + sp_label;
            break;
          } else if (n == gxl::to_upper(spec) + "I") {
            l = 12 + n_spec + sp_label;
            break;
          }
        }
      }
    }
    label_order.emplace_back(l);
  }
  // At least 2 of the 3 variables, pressure, temperature and density should be given.
  if (!has_rho) {
    if (!has_pressure || !has_temperature) {
      printf(
        "The fluctuation of density, temperature or pressure is not given in the profile, please provide at least two of them!\n");
      MpiParallel::exit();
    }
  } else if (!has_pressure) {
    if (!has_temperature) {
      printf(
        "The fluctuation of temperature or pressure is not given in the profile when rho is given, please provide at least one of them!\n");
      MpiParallel::exit();
    }
  }

  int mx, my, mz;
  bool i_read{false}, j_read{false}, k_read{false}, packing_read{false};
  std::string key;
  std::string data_packing{"POINT"};
  while (!(i_read && j_read && k_read && packing_read)) {
    std::getline(file_in, input);
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    gxl::replace(input, '=', ' ');
    std::istringstream line(input);
    while (line >> key) {
      if (key == "i" || key == "I") {
        line >> mx;
        i_read = true;
      } else if (key == "j" || key == "J") {
        line >> my;
        j_read = true;
      } else if (key == "k" || key == "K") {
        line >> mz;
        k_read = true;
      } else if (key == "f" || key == "DATAPACKING" || key == "datapacking") {
        line >> data_packing;
        data_packing = gxl::to_upper(data_packing);
        packing_read = true;
      }
    }
  }

  // This line is the DT=(double ...) line, which must exist if we output the data from Tecplot.
  std::getline(file_in, input);

  int extent[3]{mx, my, mz};

  // Then we read the variables.
  auto nv_read = static_cast<int>(var_name.size());
  gxl::VectorField3D<real> profile_read;
  profile_read.resize(extent[0], extent[1], extent[2], nv_read, 0);

  if (data_packing == "POINT") {
    for (int k = 0; k < extent[2]; ++k) {
      for (int j = 0; j < extent[1]; ++j) {
        for (int i = 0; i < extent[0]; ++i) {
          for (int l = 0; l < nv_read; ++l) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  } else if (data_packing == "BLOCK") {
    for (int l = 0; l < nv_read; ++l) {
      for (int k = 0; k < extent[2]; ++k) {
        for (int j = 0; j < extent[1]; ++j) {
          for (int i = 0; i < extent[0]; ++i) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  }

  const int n_var = parameter.get_int("n_var");
  const auto ngg = block.ngg;
  int range_i[2]{-ngg, block.mx + ngg - 1},
      range_j[2]{-ngg, block.my + ngg - 1},
      range_k[2]{-ngg, block.mz + ngg - 1};
  if (direction == 0) {
    // i direction
    ggxl::VectorField3DHost<real> profile_to_match;
    // The 2*ngg ghost layers in x direction are not used.
    profile_to_match.resize(1, block.my, block.mz, 2 * (n_var + 1), ngg);
    const int i = boundary.direction == 1 ? block.mx - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1 + 3)) {
            profile_to_match(0, j, k, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(1, block.my, block.mz, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 1) {
    // j direction
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, 1, block.mz, 2 * (n_var + 1), ngg);
    const int j = boundary.direction == 1 ? block.my - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int i = range_i[0]; i <= range_i[1]; ++i) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1)) {
            profile_to_match(i, 0, k, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, 1, block.mz, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 2) {
    // k direction
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, block.my, 1, 2 * (n_var + 1), ngg);
    const int k = boundary.direction == 1 ? block.mz - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int j = range_j[0]; j <= range_j[1]; ++j) {
      for (int i = range_i[0]; i <= range_i[1]; ++i) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1)) {
            profile_to_match(i, j, 0, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, block.my, 1, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  }
}

void init_mixingLayer_prof_compatible_cpu(Parameter &parameter, Block &b, Species &species,
                                          ggxl::VectorField3DHost<real> &profile,
                                          const int *range_x, const int *range_y, const int *range_z,
                                          const real *var_info) {
  const int n_spec = species.n_spec;
  const real delta_omega = parameter.get_real("delta_omega");
  int n_fl{0};
  if ((species.n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2)
    n_fl = 2;
  const int i_fl = parameter.get_int("i_fl");
  const int n_ps = parameter.get_int("n_ps");
  const int i_ps = parameter.get_int("i_ps");
  const int n_turb = parameter.get_int("n_turb");
  const int n_scalar = parameter.get_int("n_scalar");

  for (int j = range_y[0]; j <= range_y[1]; ++j) {
    real y = b.y(0, j, 0);
    const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
    real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / delta_omega);
    real p = var_info[4];
    real density, t;
    const real t_upper = var_info[5], t_lower = var_info[12 + n_spec];
    real yk[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER + 4] = {};

    real y_upper = (u - u_lower) / (u_upper - u_lower);
    real y_lower = 1 - y_upper;

    if (n_spec > 0) {
      // multi-species
      auto sv_upper = &var_info[6];
      auto sv_lower = &var_info[7 + n_spec + 6];

      for (int l = 0; l < n_spec; ++l) {
        yk[l] = y_upper * sv_upper[l] + y_lower * sv_lower[l];
      }

      // compute the total enthalpy of upper and lower streams
      real h0_upper{0.5 * u_upper * u_upper}, h0_lower{0.5 * u_lower * u_lower};

      real h_upper[MAX_SPEC_NUMBER], h_lower[MAX_SPEC_NUMBER];
      species.compute_enthalpy(t_upper, h_upper);
      species.compute_enthalpy(t_lower, h_lower);

      real mw_inv = 0;
      for (int l = 0; l < n_spec; ++l) {
        h0_upper += yk[l] * h_upper[l];
        h0_lower += yk[l] * h_lower[l];
        mw_inv += yk[l] / species.mw[l];
      }

      real h = y_upper * h0_upper + y_lower * h0_lower;
      h -= 0.5 * u * u;

      auto hs = h_upper, cps = h_lower;
      real err{1};
      t = t_upper * y_upper + t_lower * y_lower;
      constexpr int max_iter{1000};
      constexpr real eps{1e-3};
      int iter = 0;

      while (err > eps && iter++ < max_iter) {
        species.compute_enthalpy_and_cp(t, hs, cps);
        real cp{0}, h_new{0};
        for (int l = 0; l < n_spec; ++l) {
          cp += yk[l] * cps[l];
          h_new += yk[l] * hs[l];
        }
        const real t1 = t - (h_new - h) / cp;
        err = abs(1 - t1 / t);
        t = t1;
      }
      density = p / (R_u * mw_inv * t);

      if (n_fl > 0) {
        yk[i_fl] = var_info[6 + n_spec] * y_upper + var_info[13 + n_spec + n_spec] * y_lower;
        yk[i_fl + 1] = 0;
      }
      if (n_ps > 0) {
        for (int l = 0; l < n_ps; ++l) {
          yk[i_ps + l] =
              var_info[14 + 2 * n_spec + 4 + 2 * l] * y_upper + var_info[14 + 2 * n_spec + 4 + 2 * l + 1] * y_lower;
        }
      }
    } else {
      // Air
      constexpr real cp = gamma_air * R_air / (gamma_air - 1);
      real h0_upper = 0.5 * u_upper * u_upper + cp * var_info[5];
      real h0_lower = 0.5 * u_lower * u_lower + cp * var_info[12 + n_spec];
      real h = y_upper * h0_upper + y_lower * h0_lower - 0.5 * u * u;
      t = h / cp;
      density = p / (R_air * t);

      if (n_ps > 0) {
        if (y > 0) {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l];
          }
        } else {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l + 1];
          }
        }
      }
    }
    if (n_turb > 0) {
      if (y > 0) {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + 1 + l];
        }
      } else {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + n_turb + 1 + l];
        }
      }
    }

    auto &prof = profile;
    for (int k = range_z[0]; k <= range_z[1]; ++k) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        prof(i, j, k, 0) = density;
        prof(i, j, k, 1) = u;
        prof(i, j, k, 2) = 0;
        prof(i, j, k, 3) = 0;
        prof(i, j, k, 4) = p;
        prof(i, j, k, 5) = t;
        for (int l = 0; l < n_scalar; ++l) {
          prof(i, j, k, 6 + l) = yk[l];
        }
      }
    }
  }
}

void
DBoundCond::initialize_profile_and_rng(Parameter &parameter, Mesh &mesh, Species &species, std::vector<Field> &field,
                                       DParameter *param) {
  if (const int n_profile = parameter.get_int("n_profile"); n_profile > 0) {
    profile_hPtr_withGhost.resize(n_profile);
    // std::vector<int> special_treatment;
    for (int i = 0; i < n_profile; ++i) {
      const auto file_name = parameter.get_string_array("profile_file_names")[i];
      const auto profile_related_bc_name = parameter.get_string_array("profile_related_bc_names")[i];
      const auto &nn = parameter.get_struct(profile_related_bc_name);
      const auto label = std::get<int>(nn.at("label"));
      for (int blk = 0; blk < mesh.n_block; ++blk) {
        auto &bs = mesh[blk].boundary;
        for (auto &b: bs) {
          if (b.type_label == label) {
            int type = read_profile(b, file_name, mesh[blk], parameter, species, profile_hPtr_withGhost[i],
                                    profile_related_bc_name);
            // if (type == 1) {
            //   // mixing layer
            //   special_treatment.push_back(i);
            // }
            break;
          }
        }
      }
    }
    // for (int ll: special_treatment) {
    //   // Currently, only mixing layer is treated here.
    //   // This part allocates the memory for the d_ptr of the profile.
    //   const auto profile_related_bc_name = parameter.get_string_array("profile_related_bc_names")[ll];
    //   const auto &nn = parameter.get_struct(profile_related_bc_name);
    //   const auto label = std::get<int>(nn.at("label"));
    //   for (int blk = 0; blk < mesh.n_block; ++blk) {
    //     auto &bs = mesh[blk].boundary;
    //     for (auto &b: bs) {
    //       if (b.type_label == label) {
    //         int extent[3]{mesh[blk].mx, mesh[blk].my, mesh[blk].mz};
    //         extent[b.face] = 1;
    //         profile_hPtr_withGhost[ll].allocate_memory(extent[0], extent[1], extent[2], parameter.get_int("n_var") + 1,
    //                                                    mesh[blk].ngg);
    //       }
    //     }
    //   }
    // }
    cudaMalloc(&profile_dPtr_withGhost, sizeof(ggxl::VectorField3D<real>) * n_profile);
    cudaMemcpy(profile_dPtr_withGhost, profile_hPtr_withGhost.data(), sizeof(ggxl::VectorField3D<real>) * n_profile,
               cudaMemcpyHostToDevice);
    // for (int ll: special_treatment) {
    //   // Currently, only mixing layer is treated here.
    //   // This part computes the profile.
    //   const auto profile_related_bc_name = parameter.get_string_array("profile_related_bc_names")[ll];
    //   const auto &nn = parameter.get_struct(profile_related_bc_name);
    //   const auto label = std::get<int>(nn.at("label"));
    //   for (int blk = 0; blk < mesh.n_block; ++blk) {
    //     auto &bs = mesh[blk].boundary;
    //     for (auto &b: bs) {
    //       if (b.type_label == label) {
    //         const int direction = b.face;
    //         const int ngg = mesh[blk].ngg;
    //         const int mx = mesh[blk].mx, my = mesh[blk].my, mz = mesh[blk].mz;
    //         int range_0[2]{-ngg, mx + ngg - 1}, range_1[2]{-ngg, my + ngg - 1},
    //             range_2[2]{-ngg, mz + ngg - 1};
    //         int n0 = mx + 2 * ngg, n1 = my + 2 * ngg, n2 = mz + 2 * ngg;
    //         if (direction == 0) {
    //           n0 = 1 + ngg;
    //           range_0[0] = -ngg;
    //           range_0[1] = 0;
    //         } else if (direction == 1) {
    //           n1 = 1 + ngg;
    //           range_1[0] = -ngg;
    //           range_1[1] = 0;
    //         } else if (direction == 2) {
    //           n2 = 1 + ngg;
    //           range_2[0] = -ngg;
    //           range_2[1] = 0;
    //         }
    //
    //         std::vector<real> var_info;
    //         get_mixing_layer_info(parameter, species, var_info);
    //
    //         ggxl::VectorField3DHost<real> profile_hPtr;
    //         int extent[3]{mx, my, mz};
    //         extent[b.face] = 1;
    //         int nv = parameter.get_int("n_var");
    //         profile_hPtr.resize(extent[0], extent[1], extent[2], nv + 1, ngg);
    //         init_mixingLayer_prof_compatible_cpu(parameter, mesh[blk], species, profile_hPtr, range_0,
    //                                              range_1, range_2, var_info.data());
    //
    //         // cudaDeviceSynchronize();
    //         // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //         //   printf("Error in proc %d before var_info_dev: %s\n", parameter.get_int("myid"), cudaGetErrorString(err));
    //         //   MpiParallel::exit();
    //         // }
    //         // real *var_info_dev;
    //         // cudaMalloc(&var_info_dev, sizeof(real) * var_info.size());
    //         // cudaMemcpy(var_info_dev, var_info.data(), sizeof(real) * var_info.size(), cudaMemcpyHostToDevice);
    //         // cudaDeviceSynchronize();
    //         // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //         //   printf("Error in proc %d after var_info_dev: %s\n", parameter.get_int("myid"), cudaGetErrorString(err));
    //         //   MpiParallel::exit();
    //         // }
    //         //
    //         // uint tpb[3], bpg[3];
    //         // tpb[0] = n0 <= (2 * ngg + 1) ? 1 : 16;
    //         // tpb[1] = n1 <= (2 * ngg + 1) ? 1 : 16;
    //         // tpb[2] = n2 <= (2 * ngg + 1) ? 1 : 16;
    //         // bpg[0] = (n0 - 1) / tpb[0] + 1;
    //         // bpg[1] = (n1 - 1) / tpb[1] + 1;
    //         // bpg[2] = (n2 - 1) / tpb[2] + 1;
    //         // printf("proc %d, tpb: %d,%d,%d, bpg: %d,%d,%d, n0:n2= %d,%d,%d\n", parameter.get_int("myid"), tpb[0],
    //         //        tpb[1], tpb[2],
    //         //        bpg[0], bpg[1], bpg[2], n0, n1, n2);
    //         // dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
    //         int n_fl{0};
    //         if ((species.n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2)
    //           n_fl = 2;
    //         // printf("6\n");
    //         // cudaDeviceSynchronize();
    //         // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //         //   printf("Error in proc %d before init_mixingLayer_prof_compatible: %s\n", parameter.get_int("myid"),
    //         //          cudaGetErrorString(err));
    //         //   MpiParallel::exit();
    //         // }
    //         // int *range_x, *range_y, *range_z;
    //         // cudaMalloc(&range_x, sizeof(int) * 2);
    //         // cudaMalloc(&range_y, sizeof(int) * 2);
    //         // cudaMalloc(&range_z, sizeof(int) * 2);
    //         // cudaMemcpy(range_x, range_0, sizeof(int) * 2, cudaMemcpyHostToDevice);
    //         // cudaMemcpy(range_y, range_1, sizeof(int) * 2, cudaMemcpyHostToDevice);
    //         // cudaMemcpy(range_z, range_2, sizeof(int) * 2, cudaMemcpyHostToDevice);
    //         // printf("range_x=%d,%d, range_y=%d,%d, range_z=%d,%d\n", range_0[0], range_0[1], range_1[0], range_1[1],
    //         //        range_2[0], range_2[1]);
    //         // init_mixingLayer_prof_compatible<<<BPG, TPB>>>(field[blk].d_ptr, range_x, range_y, range_z, var_info_dev,
    //         //                                                parameter.get_real("delta_omega"),
    //         //                                                parameter.get_int("n_spec"),
    //         //                                                parameter.get_int("n_turb"),
    //         //                                                n_fl, parameter.get_int("n_ps"),
    //         //                                                profile_dPtr_withGhost, ll);
    //         // cudaDeviceSynchronize();
    //         // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //         //   printf("Error in proc %d after init_mixingLayer_prof_compatible: %s\n", parameter.get_int("myid"),
    //         //          cudaGetErrorString(err));
    //         //   MpiParallel::exit();
    //         // }
    //
    //         // print the profile out to file
    //         // Here, we assume the profile is 1D in y direction, that is, the same for x and z.
    //         // ggxl::VectorField3DHost<real> profile_hPtr;
    //         // int extent[3]{mx, my, mz};
    //         // extent[b.face] = 1;
    //         // int nv = parameter.get_int("n_var");
    //         // cudaDeviceSynchronize();
    //         // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //         //   printf("Error in proc %d before profile_hPtr: %s\n", parameter.get_int("myid"), cudaGetErrorString(err));
    //         //   MpiParallel::exit();
    //         // }
    //         // profile_hPtr.resize(extent[0], extent[1], extent[2], nv + 1, ngg);
    //         cudaDeviceSynchronize();
    //         if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //           printf("Error in proc %d before profile_hPtr cudaMemcpy: %s\n", parameter.get_int("myid"),
    //                  cudaGetErrorString(err));
    //           MpiParallel::exit();
    //         }
    //         cudaMemcpy(profile_hPtr_withGhost[ll].data(), profile_hPtr.data(),
    //                    sizeof(real) * profile_hPtr.size() * (nv + 1), cudaMemcpyHostToDevice);
    //         // cudaMemcpy(profile_hPtr.data(), profile_hPtr_withGhost[ll].data(),
    //         //            sizeof(real) * profile_hPtr.size() * (nv + 1), cudaMemcpyDeviceToHost);
    //         cudaDeviceSynchronize();
    //         if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //           printf("Error in proc %d after profile_hPtr cudaMemcpy: %s\n", parameter.get_int("myid"),
    //                  cudaGetErrorString(err));
    //           MpiParallel::exit();
    //         }
    //         printf("Before writing\n");
    //         FILE *fp = fopen("./mixingLayerProfile.dat", "w");
    //         if (fp == nullptr) {
    //           printf("Cannot open file %s\n", "./mixingLayerProfile.dat");
    //           MpiParallel::exit();
    //         }
    //         fprintf(fp, "VARIABLES = \"X\"\n\"Y\"\n\"Z\"\n\"RHO\"\n\"U\"\n\"V\"\n\"W\"\n\"p\"\n\"T\"\n");
    //         if (species.n_spec > 0) {
    //           for (int l = 0; l < species.n_spec; ++l) {
    //             fprintf(fp, "\"%s\"\n", species.spec_name[l].c_str());
    //           }
    //         }
    //         if (parameter.get_int("n_turb") == 2) {
    //           fprintf(fp, "\"TKE\"\n\"omega\"\n");
    //         }
    //         if (n_fl > 0) {
    //           fprintf(fp, "\"MixtureFraction\"\n\"MixtureFractionVariance\"\n");
    //         }
    //         fprintf(fp, "ZONE T=\"INFLOW\"\n");
    //         fprintf(fp, "I=%d, J=%d, K=%d, f=BLOCK\n", n0, n1, n2);
    //         // Print nv DOUBLE in DT=(...) into a char array
    //         std::string temp = "DT=(";
    //         for (int l = 0; l < nv + 4; ++l) {
    //           temp += "DOUBLE ";
    //         }
    //         temp += ")\n";
    //         fprintf(fp, "%s", temp.c_str());
    //         // First, x, y, z
    //         int everyFour = 0;
    //         constexpr int numberToChangeLine = 4;
    //         if (direction == 0) {
    //           range_0[0] = -ngg;
    //           range_0[1] = 0;
    //         } else if (direction == 1) {
    //           range_1[0] = -ngg;
    //           range_1[1] = 0;
    //         } else if (direction == 2) {
    //           range_2[0] = -ngg;
    //           range_2[1] = 0;
    //         }
    //         for (int k = range_2[0]; k <= range_2[1]; ++k) {
    //           for (int j = range_1[0]; j <= range_1[1]; ++j) {
    //             for (int i = range_0[0]; i <= range_0[1]; ++i) {
    //               fprintf(fp, " %.15e", mesh[blk].x(i, j, k));
    //               everyFour++;
    //               if (everyFour == numberToChangeLine) {
    //                 fprintf(fp, "\n");
    //                 everyFour = 0;
    //               }
    //             }
    //           }
    //         }
    //         if (everyFour != 0) {
    //           fprintf(fp, "\n");
    //         }
    //         everyFour = 0;
    //         for (int k = range_2[0]; k <= range_2[1]; ++k) {
    //           for (int j = range_1[0]; j <= range_1[1]; ++j) {
    //             for (int i = range_0[0]; i <= range_0[1]; ++i) {
    //               fprintf(fp, " %.15e", mesh[blk].y(i, j, k));
    //               everyFour++;
    //               if (everyFour == numberToChangeLine) {
    //                 fprintf(fp, "\n");
    //                 everyFour = 0;
    //               }
    //             }
    //           }
    //         }
    //         if (everyFour != 0) {
    //           fprintf(fp, "\n");
    //         }
    //         everyFour = 0;
    //         for (int k = range_2[0]; k <= range_2[1]; ++k) {
    //           for (int j = range_1[0]; j <= range_1[1]; ++j) {
    //             for (int i = range_0[0]; i <= range_0[1]; ++i) {
    //               fprintf(fp, " %.15e", mesh[blk].z(i, j, k));
    //               everyFour++;
    //               if (everyFour == numberToChangeLine) {
    //                 fprintf(fp, "\n");
    //                 everyFour = 0;
    //               }
    //             }
    //           }
    //         }
    //         if (everyFour != 0) {
    //           fprintf(fp, "\n");
    //         }
    //         // Then, the variables
    //         for (int l = 0; l < nv + 1; ++l) {
    //           everyFour = 0;
    //           for (int k = range_2[0]; k <= range_2[1]; ++k) {
    //             for (int j = range_1[0]; j <= range_1[1]; ++j) {
    //               for (int i = range_0[0]; i <= range_0[1]; ++i) {
    //                 fprintf(fp, " %.15e", profile_hPtr(i, j, k, l));
    //                 everyFour++;
    //                 if (everyFour == numberToChangeLine) {
    //                   fprintf(fp, "\n");
    //                   everyFour = 0;
    //                 }
    //               }
    //             }
    //           }
    //           if (everyFour != 0) {
    //             fprintf(fp, "\n");
    //           }
    //         }
    //         fclose(fp);
    //         printf("After writing\nBefore updating file name\n");
    //         // auto arr = parameter.get_string_array("profile_file_names");
    //         // for (auto &s: arr) {
    //         //   printf("name: %s\n", s.c_str());
    //         // }
    //         // arr[ll] = "mixingLayerProfile.dat";
    //         // printf("After updating file name\n");
    //         // parameter.update_parameter("profile_file_names", arr);
    //         // for (auto &s: arr) {
    //         //   printf("name: %s\n", s.c_str());
    //         // }
    //         cudaDeviceSynchronize();
    //         if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //           printf("Error in proc %d before profile_hPtr deallocate_memory: %s\n", parameter.get_int("myid"),
    //                  cudaGetErrorString(err));
    //           MpiParallel::exit();
    //         }
    //         profile_hPtr.deallocate_memory();
    //         cudaDeviceSynchronize();
    //         if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //           printf("Error in proc %d after profile_hPtr deallocate_memory: %s\n", parameter.get_int("myid"),
    //                  cudaGetErrorString(err));
    //           MpiParallel::exit();
    //         }
    //         printf("7\n");
    //         break;
    //       }
    //     }
    //   }
    // }
  }

  // Count the max number of rng needed
  auto size{0};
  const auto need_rng = parameter.get_int_array("need_rng");
  if (!need_rng.empty()) {
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (auto &b: bs) {
        if (gxl::exists(need_rng, b.type_label)) {
          int n1{mesh[blk].my}, n2{mesh[blk].mz};
          if (b.face == 1) {
            n1 = mesh[blk].mx;
          } else if (b.face == 2) {
            n1 = mesh[blk].mx;
            n2 = mesh[blk].my;
          }
          const auto ngg = mesh[blk].ngg;
          auto this_size = (n1 + 2 * ngg) * (n2 + 2 * ngg);
          if (this_size > size)
            size = this_size;
        }
      }
    }
  }
  if (size > 0) {
    cudaMalloc(&rng_d_ptr, sizeof(curandState) * size);
    dim3 TPB = {128, 1, 1};
    dim3 BPG = {(size - 1) / TPB.x + 1, 1, 1};
    // Get the current time
    time_t time_curr;
    initialize_rng<<<BPG, TPB>>>(rng_d_ptr, size, time(&time_curr));
  }

  // Read the fluctuation profiles if needed
  const auto need_fluctuation_profile = parameter.get_int_array("need_fluctuation_profile");
  if (!need_fluctuation_profile.empty()) {
    std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr;
    const int n_fluc_profile = static_cast<int>(need_fluctuation_profile.size());
    fluctuation_hPtr.resize(n_fluc_profile);
    for (int i = 0; i < n_fluc_profile; ++i) {
      const auto file_name = parameter.get_string_array("fluctuation_profile_file")[i];
      auto bc_name = parameter.get_string_array("fluctuation_profile_related_bc_name")[i];
      const auto &nn = parameter.get_struct(bc_name);
      const auto label = std::get<int>(nn.at("label"));
      for (int blk = 0; blk < mesh.n_block; ++blk) {
        auto &bs = mesh[blk].boundary;
        for (auto &b: bs) {
          if (b.type_label == label) {
            read_lst_profile(b, file_name, mesh[blk], parameter, species, fluctuation_hPtr[i], bc_name);
            break;
          }
        }
      }
    }
    cudaMalloc(&fluctuation_dPtr, sizeof(ggxl::VectorField3D<real>) * n_fluc_profile);
    cudaMemcpy(fluctuation_dPtr, fluctuation_hPtr.data(), sizeof(ggxl::VectorField3D<real>) * n_fluc_profile,
               cudaMemcpyHostToDevice);
  }
}

__global__ void
initialize_rest_rng(ggxl::VectorField2D<curandState> *rng_states, int iFace, int64_t time_stamp, int dy, int dz,
                    int ngg, int my, int mz) {
  int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg || k >= mz + DBoundCond::DF_N + ngg) {
    return;
  }
  if (j < -ngg - DBoundCond::DF_N + dy || j > my + ngg + DBoundCond::DF_N - 1 - dy ||
      k < -ngg - DBoundCond::DF_N + dz || k > mz + ngg + DBoundCond::DF_N - 1 - dz) {
    int sz = (my + 2 * ngg + 2 * DBoundCond::DF_N) * (mz + 2 * ngg + 2 * DBoundCond::DF_N);
    int i = k * (my + 2 * ngg + 2 * DBoundCond::DF_N) + j +
            (my + 2 * ngg + 2 * DBoundCond::DF_N + 1) * (ngg + DBoundCond::DF_N);
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 0));
    i += sz;
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 1));
    i += sz;
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 2));
  }
}
}
