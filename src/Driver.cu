#include "Driver.cuh"
#include "Initialize.cuh"
#include "DataCommunication.cuh"
#include "TimeAdvanceFunc.cuh"
#include "WallDistance.cuh"
#include "MixingLayer.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
Driver<mix_model, turb>::Driver(Parameter &parameter, Mesh &mesh_):
    myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter),
    spec(parameter), reac(parameter, spec), stat_collector(parameter, mesh, field) {
  printf("Initialize driver on process %d\n", myid);
  // Allocate the memory for every block
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  if (parameter.get_int("initial") == 1) {
    // If continue from previous results, then we need the residual scales
    // If the file does not exist, then we have a trouble
    std::ifstream res_scale_in("output/message/residual_scale.txt");
    res_scale_in >> res_scale[0] >> res_scale[1] >> res_scale[2] >> res_scale[3];
    res_scale_in.close();
  }

  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].setup_device_memory(parameter);
  }
  bound_cond.initialize_bc_on_GPU(mesh_, field, spec, parameter);

  initialize_basic_variables<mix_model, turb>(parameter, mesh, field, spec);

  DParameter d_param(parameter, spec, &reac);
  cudaMalloc(&param, sizeof(DParameter));
  cudaMemcpy(param, &d_param, sizeof(DParameter), cudaMemcpyHostToDevice);

  if (parameter.get_bool("steady") == 0 && parameter.get_bool("if_collect_statistics")) {
    stat_collector.initialize_statistics_collector<mix_model, turb>(spec);
  }

  write_reference_state(parameter, spec);
}

template<MixtureModel mix_model, class turb>
void Driver<mix_model, turb>::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // If we use k-omega SST model, we need the wall distance, thus we need to compute or read it here.
  if constexpr (TurbMethod<turb>::needWallDistance == true) {
    // SST method
    acquire_wall_distance<mix_model, turb>(*this);
  } else {
    if (parameter.get_int("if_compute_wall_distance") == 1) {
      acquire_wall_distance<mix_model, turb>(*this);
    }
  }

  if (mesh.dimension == 2) {
    for (auto b = 0; b < mesh.n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient <<<BPG, tpb >>>(field[b].d_ptr, param);
    }
  }

  // Second, apply boundary conditions to all boundaries, including face communication between faces
  for (integer b = 0; b < mesh.n_block; ++b) {
    bound_cond.apply_boundary_conditions<mix_model, turb>(mesh[b], field[b], param);
  }
  printf("Boundary conditions are applied successfully for initialization on process %d\n", myid);


  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_velocity<<<bpg, tpb>>>(field[i].d_ptr);
    if constexpr (TurbMethod<turb>::hasMut == true) {
      initialize_mut<mix_model, turb><<<bpg, tpb>>>(field[i].d_ptr, param);
    }
  }
  cudaDeviceSynchronize();
  // Third, communicate values between processes
  printf("Before data communication on process %d\n", myid);
  data_communication<mix_model, turb>(mesh, field, parameter, 0, param);

  printf("Finish data transfer on process %d.\n", myid);
  cudaDeviceSynchronize();

  for (auto b = 0; b < mesh.n_block; ++b) {
    integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<mix_model><<<bpg, tpb>>>(field[b].d_ptr, param);
  }
  cudaDeviceSynchronize();
  if (myid == 0) {
    printf("The flowfield is completely initialized on GPU.\n");
  }
}

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, integer n_point_times3) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real x{zone->x(i, j, k)}, y{zone->y(i, j, k)}, z{zone->z(i, j, k)};
  const integer n_wall_point = n_point_times3 / 3;
  auto &wall_dist = zone->wall_distance(i, j, k);
  wall_dist = 1e+6;
  for (integer l = 0; l < n_wall_point; ++l) {
    const integer idx = 3 * l;
    real d = (x - wall_point_coor[idx]) * (x - wall_point_coor[idx]) +
             (y - wall_point_coor[idx + 1]) * (y - wall_point_coor[idx + 1]) +
             (z - wall_point_coor[idx + 2]) * (z - wall_point_coor[idx + 2]);
    if (wall_dist > d) {
      wall_dist = d;
    }
  }
  wall_dist = std::sqrt(wall_dist);
}

void write_reference_state(const Parameter &parameter, const Species &species) {
  if (parameter.get_int("myid") == 0) {
    const std::filesystem::path out_dir("output/message");
    if (!exists(out_dir)) {
      create_directories(out_dir);
    }
    FILE *ref_state = fopen("output/message/reference_state.txt", "w");
    if (parameter.get_int("problem_type") == 1) {
      // For mixing layers, we need to output info about both streams.
      std::vector<real> var_info;
      cfd::get_mixing_layer_info(parameter, species, var_info);
      fprintf(ref_state, "Upper stream:\ndensity = %16.10e\n", var_info[0]);
      fprintf(ref_state, "u = %16.10e\n", var_info[1]);
      fprintf(ref_state, "v = %16.10e\n", var_info[2]);
      fprintf(ref_state, "w = %16.10e\n", var_info[3]);
      real velocity{std::sqrt(var_info[1] * var_info[1] + var_info[2] * var_info[2] + var_info[3] * var_info[3])};
      fprintf(ref_state, "velocity = %16.10e\n", velocity);
      fprintf(ref_state, "pressure = %16.10e\n", var_info[4]);
      fprintf(ref_state, "temperature = %16.10e\n", var_info[5]);
      int ns{species.n_spec};
      real mu, gamma{gamma_air}, mw{mw_air};
      if (ns > 0) {
        real cp_i[MAX_SPEC_NUMBER];
        species.compute_cp(var_info[5], cp_i);
        real cp{0};
        mw = 0;
        for (const auto& [name, i]: species.spec_list) {
          if (var_info[6 + i] > 0)
            fprintf(ref_state, "Y_%s = %16.10e\n", name.c_str(), var_info[6 + i]);
          mw += var_info[6 + i] / species.mw[i];
          cp += cp_i[i] * var_info[6 + i];
        }
        gamma = cp / (cp - R_u * mw);
        mw = 1 / mw;
        mu = compute_viscosity(var_info[5], mw, &var_info[6], species);
      } else {
        mu = Sutherland(var_info[5]);
      }
      real c{std::sqrt(gamma * R_u / mw * var_info[5])};
      fprintf(ref_state, "speed_of_sound = %16.10e\n", c);
      fprintf(ref_state, "specific_heat_ratio = %16.10e\n", gamma);
      fprintf(ref_state, "Ma = %16.10e\n", velocity / c);
      fprintf(ref_state, "Re_unit = %16.10e\n", var_info[0] * velocity / mu);
      fprintf(ref_state, "mu = %16.10e\n", mu);
      // Next, the lower stream
      fprintf(ref_state, "\nLower stream:\ndensity = %16.10e\n", var_info[7 + ns]);
      fprintf(ref_state, "u = %16.10e\n", var_info[8 + ns]);
      fprintf(ref_state, "v = %16.10e\n", var_info[9 + ns]);
      fprintf(ref_state, "w = %16.10e\n", var_info[10 + ns]);
      velocity = std::sqrt(var_info[8 + ns] * var_info[8 + ns] + var_info[9 + ns] * var_info[9 + ns] +
                           var_info[10 + ns] * var_info[10 + ns]);
      fprintf(ref_state, "velocity = %16.10e\n", velocity);
      fprintf(ref_state, "pressure = %16.10e\n", var_info[11 + ns]);
      fprintf(ref_state, "temperature = %16.10e\n", var_info[12 + ns]);
      if (ns > 0) {
        real cp_i[MAX_SPEC_NUMBER];
        species.compute_cp(var_info[12 + ns], cp_i);
        real cp{0};
        mw = 0;
        for (const auto& [name, i]: species.spec_list) {
          if (var_info[13 + ns + i] > 0)
            fprintf(ref_state, "Y_%s = %16.10e\n", name.c_str(), var_info[13 + ns + i]);
          mw += var_info[13 + ns + i] / species.mw[i];
          cp += cp_i[i] * var_info[13 + ns + i];
        }
        gamma = cp / (cp - R_u * mw);
        mw = 1 / mw;
        mu = compute_viscosity(var_info[12 + ns], mw, &var_info[13 + ns], species);
      } else {
        mu = Sutherland(var_info[12 + ns]);
      }
      c = std::sqrt(gamma * R_u / mw * var_info[12 + ns]);
      fprintf(ref_state, "speed_of_sound = %16.10e\n", c);
      fprintf(ref_state, "specific_heat_ratio = %16.10e\n", gamma);
      fprintf(ref_state, "Ma = %16.10e\n", velocity / c);
      fprintf(ref_state, "Re_unit = %16.10e\n", var_info[7 + ns] * velocity / mu);
      fprintf(ref_state, "mu = %16.10e\n", mu);
    } else {
      fprintf(ref_state, "Reference state\nrho_ref = %16.10e\n", parameter.get_real("rho_inf"));
      fprintf(ref_state, "v_ref = %16.10e\n", parameter.get_real("v_inf"));
      fprintf(ref_state, "p_ref = %16.10e\n", parameter.get_real("p_inf"));
      fprintf(ref_state, "T_ref = %16.10e\n", parameter.get_real("T_inf"));
      auto &sv_ref = parameter.get_real_array("sv_inf");
      for (const auto& [name, i]: species.spec_list) {
        if (sv_ref[i] > 0)
          fprintf(ref_state, "Y_%s = %16.10e\n", name.c_str(), sv_ref[i]);
      }
      fprintf(ref_state, "Ma_ref = %16.10e\n", parameter.get_real("M_inf"));
      fprintf(ref_state, "Re_unit = %16.10e\n", parameter.get_real("Re_unit"));
      fprintf(ref_state, "mu_ref = %16.10e\n", parameter.get_real("mu_inf"));
      fprintf(ref_state, "acoustic_speed_ref = %16.10e\n", parameter.get_real("speed_of_sound"));
      fprintf(ref_state, "specific_heat_ratio = %16.10e\n", parameter.get_real("specific_heat_ratio_inf"));
    }
    fclose(ref_state);
  }
}

// Instantiate all possible drivers
template
struct Driver<MixtureModel::Air, Laminar>;
template
struct Driver<MixtureModel::Air, SST<TurbSimLevel::RANS>>;
template
struct Driver<MixtureModel::Air, SST<TurbSimLevel::DES>>;
template
struct Driver<MixtureModel::Mixture, Laminar>;
template
struct Driver<MixtureModel::Mixture, SST<TurbSimLevel::RANS>>;
template
struct Driver<MixtureModel::Mixture, SST<TurbSimLevel::DES>>;
template
struct Driver<MixtureModel::FR, Laminar>;
template
struct Driver<MixtureModel::FR, SST<TurbSimLevel::RANS>>;
template
struct Driver<MixtureModel::FR, SST<TurbSimLevel::DES>>;
template
struct Driver<MixtureModel::MixtureFraction, Laminar>;
template
struct Driver<MixtureModel::MixtureFraction, SST<TurbSimLevel::RANS>>;

} // cfd