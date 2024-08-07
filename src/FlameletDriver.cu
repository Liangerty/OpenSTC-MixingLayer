#include "Driver.cuh"
#include <fstream>
#include "Initialize.cuh"
#include "DataCommunication.cuh"
#include "WallDistance.cuh"

namespace cfd {
template<class turb>
Driver<MixtureModel::FL, turb>::Driver(Parameter &parameter, Mesh &mesh_):
    myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter),
    spec(parameter), flameletLib(parameter), stat_collector(parameter, mesh, field) {
  // Allocate the memory for every block
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  if (parameter.get_int("initial") == 1) {
    // If continue from previous results, then we need the residual scales
    // If the file does not exist, then we have a trouble
    std::ifstream res_scale_in("output/message/residual_scale.txt");
    res_scale_in >> res_scale[0] >> res_scale[1] >> res_scale[2] >> res_scale[3];
    res_scale_in.close();
  }

  for (int blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].setup_device_memory(parameter);
  }
  bound_cond.initialize_bc_on_GPU(mesh_, field, spec, parameter);
  initialize_basic_variables<MixtureModel::FL, turb>(parameter, mesh, field, spec);
  DParameter d_param(parameter, spec, nullptr, &flameletLib);
  cudaMalloc(&param, sizeof(DParameter));
  cudaMemcpy(param, &d_param, sizeof(DParameter), cudaMemcpyHostToDevice);
  if (parameter.get_bool("steady") == 0 && parameter.get_bool("if_collect_statistics")) {
    stat_collector.initialize_statistics_collector<MixtureModel::FL, turb>(spec);
  }

  write_reference_state(parameter, spec);
}

template<class turb>
void Driver<MixtureModel::FL, turb>::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // If we use k-omega SST model, we need the wall distance, thus we need to compute or read it here.
  if constexpr (TurbMethod<turb>::needWallDistance == true) {
    // SST method
    acquire_wall_distance<MixtureModel::FL, turb>(*this);
  }

  if (mesh.dimension == 2) {
    for (auto b = 0; b < mesh.n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient <<<BPG, tpb >>>(field[b].d_ptr, param);
    }
  }

  // Second, apply boundary conditions to all boundaries, including face communication between faces
  for (int b = 0; b < mesh.n_block; ++b) {
    bound_cond.apply_boundary_conditions<MixtureModel::FL, turb>(mesh[b], field[b], param);
  }
  if (myid == 0) {
    printf("Boundary conditions are applied successfully for initialization\n");
  }

  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    int mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    if constexpr (TurbMethod<turb>::hasMut == true) {
      initialize_mut<MixtureModel::FL, turb><<<bpg, tpb >>>(field[i].d_ptr, param);
    }
  }
  cudaDeviceSynchronize();
  // Third, communicate values between processes
  data_communication<MixtureModel::FL, turb>(mesh, field, parameter, 0, param);

  if (myid == 0) {
    printf("Finish data transfer.\n");
  }
  cudaDeviceSynchronize();

  for (auto b = 0; b < mesh.n_block; ++b) {
    int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<MixtureModel::FL><<<bpg, tpb>>>(field[b].d_ptr, param);
  }
  cudaDeviceSynchronize();
  if (myid == 0) {
    printf("The flowfield is completely initialized on GPU.\n");
  }
}

// Explicitly instantiate the template, which means the flamelet model can only be used with RANS and LES.
template
struct Driver<MixtureModel::FL, SST<TurbSimLevel::RANS>>;
//template<> struct Driver<MixtureModel::FL,TurbulenceMethod::LES>;
}