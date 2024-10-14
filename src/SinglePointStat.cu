#include "SinglePointStat.cuh"
#include <filesystem>
#include "gxl_lib/MyString.h"
#include "DParameter.cuh"

namespace cfd {
SinglePointStat::SinglePointStat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field,
                                 const Species &species) : parameter(_parameter), mesh(_mesh), field(_field) {
  if (!_parameter.get_bool("if_collect_statistics")) {
    return;
  }
  myid = _parameter.get_int("myid");

  const std::filesystem::path out_dir("output/stat");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }

  init_stat_name(species);
  // update the variables in parameter, these are used to initialize the memory in the field.
  counter_rey1st.resize(n_reyAve, 0);
  parameter.update_parameter("n_stat_reynolds_1st", n_reyAve);
  counter_fav1st.resize(n_favAve, 0);
  parameter.update_parameter("n_stat_favre_1st", n_favAve);
  counter_rey2nd.resize(n_rey2nd, 0);
  parameter.update_parameter("n_stat_reynolds_2nd", n_rey2nd);
  counter_fav2nd.resize(n_fav2nd, 0);
  parameter.update_parameter("n_stat_favre_2nd", n_fav2nd);
  parameter.update_parameter("reyAveVarIndex", reyAveVarIndex);
}

void SinglePointStat::init_stat_name(const Species &species) {
  // The default average is the favre one.
  // All variables computed will be Favre averaged.
  int n_spec = species.n_spec;
  if (n_spec > 0) {
    n_favAve += n_spec;
    n_fav2nd += n_spec;
    for (int l = 0; l < n_spec; ++l) {
      favAveVar.push_back(species.spec_name[l]);
      fav2ndVar.push_back(species.spec_name[l] + species.spec_name[l]);
    }
  }
  if (int n_ps = parameter.get_int("n_passive_scalar"); n_ps > 0) {
    n_favAve += n_ps;
    n_fav2nd += n_ps;
    for (int l = 0; l < n_ps; ++l) {
      favAveVar.push_back("ps" + std::to_string(l + 1));
      fav2ndVar.push_back("ps" + std::to_string(l + 1) + "ps" + std::to_string(l + 1));
    }
  }

  // Next, see if there are some basic variables except rho, p are to be averaged.
  if (auto &stat_rey_1st = parameter.get_string_array("stat_rey_1st"); !stat_rey_1st.empty()) {
    int n_rey_1st = (int) stat_rey_1st.size();
    for (int i = 0; i < n_rey_1st; ++i) {
      auto var = gxl::to_upper(stat_rey_1st[i]);
      if (var == "U") {
        reyAveVar.emplace_back("u");
        reyAveVarIndex.push_back(1);
        ++n_reyAve;
      } else if (var == "V") {
        reyAveVar.emplace_back("v");
        reyAveVarIndex.push_back(2);
        ++n_reyAve;
      } else if (var == "W") {
        reyAveVar.emplace_back("w");
        reyAveVarIndex.push_back(3);
        ++n_reyAve;
      } else if (var == "T") {
        reyAveVar.emplace_back("T");
        reyAveVarIndex.push_back(5);
        ++n_reyAve;
      } else {
        printf("Error: Variable %s for Reynolds average is not supported.\n", var.c_str());
      }
    }
  }
  if (parameter.get_bool("rho_p_correlation")) {
    ++n_rey2nd;
    rey2ndVar.emplace_back("rhoP");
  }
}

void SinglePointStat::collect_data(DParameter *param) {
  for (auto &c: counter_rey1st) {
    ++c;
  }
  for (auto &c: counter_fav1st) {
    ++c;
  }
  for (auto &c: counter_rey2nd) {
    ++c;
  }
  for (auto &c: counter_fav2nd) {
    ++c;
  }
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx + 2}, my{mesh[b].my + 2}, mz{mesh[b].mz + 2};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    collect_single_point_statistics<<<bpg, tpb>>>(field[b].d_ptr, param);
  }

}

void SinglePointStat::export_statistical_data(DParameter *param, bool perform_spanwise_average) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey2);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav2);
  MPI_Status status;

  MPI_Offset offset[4]{0, 0, 0, 0};
  memcpy(offset, offset_unit, sizeof(offset_unit));
  if (myid == 0) {
    MPI_File_write_at(fp_rey1, offset[0], counter_rey1st.data(), n_reyAve, MPI_INT32_T, &status);
    offset[0] += 4 * n_reyAve;
  }
  for (int b = 0; b < mesh.n_block; ++b) {
    auto &zone = field[b].h_ptr;
    const int mx = mesh[b].mx, my = mesh[b].my, mz = mesh[b].mz;
    const int64_t N = (int64_t) mx * my * mz;
    const auto sz = (long long) (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;

    cudaMemcpy(field[b].collect_reynolds_1st.data(), zone->collect_reynolds_1st.data(), sz * n_reyAve,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].collect_favre_1st.data(), zone->collect_favre_1st.data(), sz * n_favAve,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].collect_reynolds_2nd.data(), zone->collect_reynolds_2nd.data(), sz * n_rey2nd,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].collect_favre_2nd.data(), zone->collect_favre_2nd.data(), sz * n_fav2nd,
               cudaMemcpyDeviceToHost);

    // We create this datatype because in the original MPI_File_write_at, the number of elements is a 64-bit integer.
    // However, the nx*ny*nz may be larger than 2^31, so we need to use MPI_Type_create_subarray to create a datatype
    MPI_Datatype ty;
    int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    MPI_File_write_at(fp_rey1, offset[0], &mx, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], &my, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], &mz, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], field[b].collect_reynolds_1st.data(), n_reyAve, ty, &status);
    offset[0] += sz * n_reyAve;

    MPI_File_write_at(fp_fav1, offset[1], &mx, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], &my, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], &mz, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], field[b].collect_favre_1st.data(), n_favAve, ty, &status);
    offset[1] += sz * n_favAve;

    MPI_File_write_at(fp_rey2, offset[2], &mx, 1, MPI_INT32_T, &status);
    offset[2] += 4;
    MPI_File_write_at(fp_rey2, offset[2], &my, 1, MPI_INT32_T, &status);
    offset[2] += 4;
    MPI_File_write_at(fp_rey2, offset[2], &mz, 1, MPI_INT32_T, &status);
    offset[2] += 4;
    MPI_File_write_at(fp_rey2, offset[2], field[b].collect_reynolds_2nd.data(), n_rey2nd, ty, &status);
    offset[2] += sz * n_rey2nd;

    MPI_File_write_at(fp_fav2, offset[3], &mx, 1, MPI_INT32_T, &status);
    offset[3] += 4;
    MPI_File_write_at(fp_fav2, offset[3], &my, 1, MPI_INT32_T, &status);
    offset[3] += 4;
    MPI_File_write_at(fp_fav2, offset[3], &mz, 1, MPI_INT32_T, &status);
    offset[3] += 4;
    MPI_File_write_at(fp_fav2, offset[3], field[b].collect_favre_2nd.data(), n_fav2nd, ty, &status);
    offset[3] += sz * n_fav2nd;

    MPI_Type_free(&ty);
  }
}

void SinglePointStat::initialize_statistics_collector(const Species &species) {
  compute_offset_for_export_data();

  if (parameter.get_bool("if_continue_collect_statistics")) {
    read_previous_statistical_data();
  }

//  if (parameter.get_bool("output_statistics_plt"))
//    prepare_for_statistical_data_plot<mix_model, turb>(species);
//  cudaMalloc(&counter_ud_device, sizeof(int) * UserDefineStat::n_collect);
}

void SinglePointStat::compute_offset_for_export_data() {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey2);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav2);
  MPI_Status status;
  MPI_Offset offset[4]{0, 0, 0, 0};

  int n_stat_reynolds_1st = parameter.get_int("n_stat_reynolds_1st");
  int n_stat_favre_1st = parameter.get_int("n_stat_favre_1st");
  int n_stat_reynolds_2nd = parameter.get_int("n_stat_reynolds_2nd");
  int n_stat_favre_2nd = parameter.get_int("n_stat_favre_2nd");
  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    // collect reynolds 1st order statistics
    MPI_File_write_at(fp_rey1, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey1, 4, &n_stat_reynolds_1st, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey1, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[0] = 4 * 3;
    for (const auto &var: reyAveVar) {
      gxl::write_str(var.c_str(), fp_rey1, offset[0]);
    }
//    MPI_File_write_at(fp_rey1, offset[0], counter_rey1st.data(), n_stat_reynolds_1st, MPI_INT32_T, &status);
//    offset[0] += 4 * n_stat_reynolds_1st;

    // collect favre 1st order statistics
    MPI_File_write_at(fp_fav1, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav1, 4, &n_stat_favre_1st, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav1, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[1] = 4 * 3;
    for (const auto &var: favAveVar) {
      gxl::write_str(var.c_str(), fp_fav1, offset[1]);
    }

    // collect reynolds 2nd order statistics
    MPI_File_write_at(fp_rey2, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey2, 4, &n_stat_reynolds_2nd, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey2, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[2] = 4 * 3;
    for (const auto &var: rey2ndVar) {
      gxl::write_str(var.c_str(), fp_rey2, offset[2]);
    }

    // collect favre 2nd order statistics
    MPI_File_write_at(fp_fav2, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav2, 4, &n_stat_favre_2nd, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav2, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[3] = 4 * 3;
    for (const auto &var: fav2ndVar) {
      gxl::write_str(var.c_str(), fp_fav2, offset[3]);
    }
  }
  MPI_Bcast(offset, 4, MPI_OFFSET, 0, MPI_COMM_WORLD);

  int n_block = 0;
  for (int p = 0; p < parameter.get_int("myid"); ++p) {
    n_block += mesh.nblk[p];
  }
  offset_unit[0] = offset[0];
  offset_unit[1] = offset[1];
  offset_unit[2] = offset[2];
  offset_unit[3] = offset[3];
  if (myid == 0) {
    offset_unit[0] += 4 * n_stat_reynolds_1st;
  }
  for (int b = 0; b < n_block; ++b) {
    MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
    offset_unit[0] += sz * n_stat_reynolds_1st + 4 * 3;
    offset_unit[1] += sz * n_stat_favre_1st + 4 * 3;
    offset_unit[2] += sz * n_stat_reynolds_2nd + 4 * 3;
    offset_unit[3] += sz * n_stat_favre_2nd + 4 * 3;
  }
}

void SinglePointStat::read_previous_statistical_data() {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_fav1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey2);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_fav2);
  MPI_Status status;

  MPI_Offset offset_read[4]{0, 0, 0, 0};
  // Reynolds 1st order statistics
  int nBlock = 0;
  MPI_File_read_at(fp_rey1, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_rey_1st.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_rey1 = 0;
  MPI_File_read_at(fp_rey1, 4, &n_read_rey1, 1, MPI_INT32_T, &status);
  int ngg_read = 0;
  MPI_File_read_at(fp_rey1, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_rey_1st.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> rey1Var(n_read_rey1);
  offset_read[0] = 4 * 3;
  for (int l = 0; l < n_read_rey1; ++l) {
    rey1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey1, offset_read[0]);
  }
  std::vector<int> counter_read(n_read_rey1);
  MPI_File_read_at(fp_rey1, offset_read[0], counter_read.data(), n_read_rey1, MPI_INT32_T, &status);
  offset_read[0] += 4 * n_read_rey1;
  std::vector<int> read_rey1_index(n_read_rey1, -1);
  for (int l = 0; l < n_read_rey1; ++l) {
    for (int i = 0; i < n_reyAve; ++i) {
      if (rey1Var[l] == reyAveVar[i]) {
        read_rey1_index[l] = i;
        counter_rey1st[i] = counter_read[l];
        break;
      }
    }
  }
  // Favre 1st order statistics
  nBlock = 0;
  MPI_File_read_at(fp_fav1, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_fav_1st.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_fav1 = 0;
  MPI_File_read_at(fp_fav1, 4, &n_read_fav1, 1, MPI_INT32_T, &status);
  ngg_read = 0;
  MPI_File_read_at(fp_fav1, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_fav_1st.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> fav1Var(n_read_fav1);
  offset_read[1] = 4 * 3;
  for (int l = 0; l < n_read_fav1; ++l) {
    fav1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_fav1, offset_read[1]);
  }
  counter_read.resize(n_read_fav1);
  MPI_File_read_at(fp_fav1, offset_read[1], counter_read.data(), n_read_fav1, MPI_INT32_T, &status);
  offset_read[1] += 4 * n_read_fav1;
  std::vector<int> read_fav1_index(n_read_fav1, -1);
  for (int l = 0; l < n_read_fav1; ++l) {
    for (int i = 0; i < n_favAve; ++i) {
      if (fav1Var[l] == favAveVar[i]) {
        read_fav1_index[l] = i;
        counter_fav1st[i] = counter_read[l];
        break;
      }
    }
  }
  // Reynolds 2nd order statistics
  nBlock = 0;
  MPI_File_read_at(fp_rey2, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_rey_2nd.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_rey2 = 0;
  MPI_File_read_at(fp_rey2, 4, &n_read_rey2, 1, MPI_INT32_T, &status);
  ngg_read = 0;
  MPI_File_read_at(fp_rey2, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_rey_2nd.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> rey2Var(n_read_rey2);
  offset_read[2] = 4 * 3;
  for (int l = 0; l < n_read_rey2; ++l) {
    rey2Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey2, offset_read[2]);
  }
  counter_read.resize(n_read_rey2);
  MPI_File_read_at(fp_rey2, offset_read[2], counter_read.data(), n_read_rey2, MPI_INT32_T, &status);
  std::vector<int> read_rey2_index(n_read_rey2, -1);
  for (int l = 0; l < n_read_rey2; ++l) {
    for (int i = 0; i < n_rey2nd; ++i) {
      if (rey2Var[l] == rey2ndVar[i]) {
        read_rey2_index[l] = i;
        counter_rey2nd[i] = counter_read[l];
        break;
      }
    }
  }
  // Favre 2nd order statistics
  nBlock = 0;
  MPI_File_read_at(fp_fav2, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_fav_2nd.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_fav2 = 0;
  MPI_File_read_at(fp_fav2, 4, &n_read_fav2, 1, MPI_INT32_T, &status);
  ngg_read = 0;
  MPI_File_read_at(fp_fav2, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_fav_2nd.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> fav2Var(n_read_fav2);
  offset_read[3] = 4 * 3;
  for (int l = 0; l < n_read_fav2; ++l) {
    fav2Var[l] = gxl::read_str_from_binary_MPI_ver(fp_fav2, offset_read[3]);
  }
  counter_read.resize(n_read_fav2);
  MPI_File_read_at(fp_fav2, offset_read[3], counter_read.data(), n_read_fav2, MPI_INT32_T, &status);
  std::vector<int> read_fav2_index(n_read_fav2, -1);
  for (int l = 0; l < n_read_fav2; ++l) {
    for (int i = 0; i < n_fav2nd; ++i) {
      if (fav2Var[l] == fav2ndVar[i]) {
        read_fav2_index[l] = i;
        counter_fav2nd[i] = counter_read[l];
        break;
      }
    }
  }

  int n_block_ahead = 0;
  for (int p = 0; p < parameter.get_int("myid"); ++p) {
    n_block_ahead += mesh.nblk[p];
  }
  for (int b = 0; b < n_block_ahead; ++b) {
    MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
    offset_read[0] += sz * n_read_rey1 + 4 * 3;
    offset_read[1] += sz * n_read_fav1 + 4 * 3;
    offset_read[2] += sz * n_read_rey2 + 4 * 3;
    offset_read[3] += sz * n_read_fav2 + 4 * 3;
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    int mx, my, mz;
    MPI_File_read_at(fp_rey1, offset_read[0], &mx, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    MPI_File_read_at(fp_rey1, offset_read[0], &my, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    MPI_File_read_at(fp_rey1, offset_read[0], &mz, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
      printf("Error: The mesh size in the previous statistical data is not consistent with the current mesh.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const auto sz = (long long) (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;
    MPI_Datatype ty;
    int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    for (int l = 0; l < n_read_rey1; ++l) {
      int i = read_rey1_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_rey1, offset_read[0], field[b].collect_reynolds_1st[i], 1, ty, &status);
      }
      offset_read[0] += sz;
    }
    cudaMemcpy(field[b].d_ptr->collect_reynolds_1st.data(), field[b].collect_reynolds_1st.data(), sz * n_reyAve,
               cudaMemcpyHostToDevice);
    for (int l = 0; l < n_read_fav1; ++l) {
      int i = read_fav1_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_fav1, offset_read[1], field[b].collect_favre_1st[i], 1, ty, &status);
      }
      offset_read[1] += sz;
    }
    cudaMemcpy(field[b].d_ptr->collect_favre_1st.data(), field[b].collect_favre_1st.data(), sz * n_favAve,
               cudaMemcpyHostToDevice);
    for (int l = 0; l < n_read_rey2; ++l) {
      int i = read_rey2_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_rey2, offset_read[2], field[b].collect_reynolds_2nd[i], 1, ty, &status);
      }
      offset_read[2] += sz;
    }
    cudaMemcpy(field[b].d_ptr->collect_reynolds_2nd.data(), field[b].collect_reynolds_2nd.data(), sz * n_rey2nd,
               cudaMemcpyHostToDevice);
    for (int l = 0; l < n_read_fav2; ++l) {
      int i = read_fav2_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_fav2, offset_read[3], field[b].collect_favre_2nd[i], 1, ty, &status);
      }
      offset_read[3] += sz;
    }
    cudaMemcpy(field[b].d_ptr->collect_favre_2nd.data(), field[b].collect_favre_2nd.data(), sz * n_fav2nd,
               cudaMemcpyHostToDevice);
  }
}

__global__ void collect_single_point_statistics(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= extent[0] + 1 || j >= extent[1] + 1 || k >= extent[2] + 1) return;

  const auto &bv = zone->bv;

  // The first order statistics of the flow field
  // Reynolds averaged variables
  auto &rey1st = zone->collect_reynolds_1st;
  auto rho = bv(i, j, k, 0), p = bv(i, j, k, 4);
  rey1st(i, j, k, 0) += rho; // rho
  rey1st(i, j, k, 1) += p; // p
  for (int l = 2; l < param->n_reyAve; ++l) {
    rey1st(i, j, k, l) += bv(i, j, k, param->reyAveVarIndex[l]);
  }
  // Favre averaged variables
  auto &fav1st = zone->collect_favre_1st;
  real u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3), T = bv(i, j, k, 5);
  fav1st(i, j, k, 0) += rho * u; // rho*u
  fav1st(i, j, k, 1) += rho * v; // rho*v
  fav1st(i, j, k, 2) += rho * w; // rho*w
  fav1st(i, j, k, 3) += rho * T; // rho*T
  const auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    fav1st(i, j, k, 4 + l) += rho * sv(i, j, k, l);
  }

  // The second order statistics of the flow field
  // Reynolds averaged variables
  auto &rey2nd = zone->collect_reynolds_2nd;
  rey2nd(i, j, k, 0) += rho * rho; // rho*rho
  rey2nd(i, j, k, 1) += p * p; // p*p
  if (param->rho_p_correlation)
    rey2nd(i, j, k, 2) += rho * p; // rho*p

  // Favre averaged variables
  auto &fav2nd = zone->collect_favre_2nd;
  fav2nd(i, j, k, 0) += rho * u * u; // rho*u*u
  fav2nd(i, j, k, 1) += rho * v * v; // rho*v*v
  fav2nd(i, j, k, 2) += rho * w * w; // rho*w*w
  fav2nd(i, j, k, 3) += rho * u * v; // rho*u*v
  fav2nd(i, j, k, 4) += rho * u * w; // rho*u*w
  fav2nd(i, j, k, 5) += rho * v * w; // rho*v*w
  fav2nd(i, j, k, 6) += rho * T * T; // rho*T*T
}
} // cfd