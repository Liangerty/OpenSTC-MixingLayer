#pragma once

#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"
#include <mpi.h>
#include "UDStat.h"
#include <filesystem>
#include "gxl_lib/MyString.h"
#include "ChemData.h"
#include "TurbMethod.hpp"

namespace cfd {

__global__ void collect_statistics(DZone *zone, DParameter *param);

class StatisticsCollector {
public:
  explicit
  StatisticsCollector(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field);

  template<MixtureModel mix_model, class turb>
  void initialize_statistics_collector(const Species &species);

  template<MixtureModel mix_model, class turb_method>
  void collect_data(DParameter *param);

  void export_statistical_data(DParameter *param, bool perform_spanwise_average);

private:
  // Basic info
  bool if_collect_statistics{false};
  int start_iter{0};
public:
  int counter{0};
private:
  int myid{0};
  // Data to be bundled
  const Parameter &parameter;
  const Mesh &mesh;
  std::vector<Field> &field;
  // Offset unit.
  // The first number in the file is the number of blocks, which would occupy 4 bytes.
  // The second is the number of variables, which also occupy 4 bytes.
  MPI_Offset offset_unit[3]{12, 12, 12};
  // User-defined statistical data name
  std::array<std::string, UserDefineStat::n_collect> ud_collect_name;
public:
  std::array<int, UserDefineStat::n_collect> counter_ud{0};
  int *counter_ud_device = nullptr;
private:
  // plot-related variables
  MPI_Offset offset_header{0};
  int n_plot{0};
  MPI_Offset *offset_minmax_var = nullptr;
  MPI_Offset *offset_var = nullptr;

  void compute_offset_for_export_data();

  void read_previous_statistical_data();

  void plot_statistical_data(DParameter *param, bool perform_spanwise_average);

  template<MixtureModel mix_model, class turb>
  void prepare_for_statistical_data_plot(const Species &species);

  template<MixtureModel mix_model, class turb>
  int32_t acquire_variable_names(std::vector<std::string> &var_name, const Species &species);
};

__global__ void compute_statistical_data(DZone *zone, DParameter *param, int counter, const int *counter_ud);

__global__ void
compute_statistical_data_spanwise_average(DZone *zone, DParameter *param, int counter, const int *counter_ud);

template<MixtureModel mix_model, class turb>
void StatisticsCollector::initialize_statistics_collector(const Species &species) {
  compute_offset_for_export_data();

  if (parameter.get_bool("if_continue_collect_statistics")) {
    read_previous_statistical_data();
  }

  prepare_for_statistical_data_plot<mix_model, turb>(species);
  cudaMalloc(&counter_ud_device, sizeof(int) * UserDefineStat::n_collect);
}

template<MixtureModel mix_model, class turb>
void StatisticsCollector::prepare_for_statistical_data_plot(const Species &species) {
  const std::filesystem::path out_dir("output");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header; thus we let process 0 to write it.

//  auto *offset_solution_time = new MPI_Offset[mesh.n_block_total];

  MPI_Offset offset{0};
  if (myid == 0) {
    // i. Magic number, Version number
    // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
    // difference is related to us. For common use, we use V112.
    constexpr auto magic_number{"#!TDV112"};
    gxl::write_str_without_null(magic_number, fp, offset);

    // ii. Integer value of 1
    constexpr int32_t byte_order{1};
    MPI_File_write_at(fp, offset, &byte_order, 1, MPI_INT32_T, &status);
    offset += 4;

    // iii. Title and variable names.
    // 1. FileType: 0=full, 1=grid, 2=solution.
    constexpr int32_t file_type{0};
    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
    offset += 4;
    // 2. Title
    gxl::write_str("Solution file", fp, offset);
    // 3. Number of variables in the datafile
    std::vector<std::string> var_name{"x", "y", "z", "<rho>", "<u>", "<v>", "<w>", "<p>", "<T>", "<u'u'>", "<v'v'>",
                                      "<w'w'>", "<u'v'>", "<u'w'>", "<v'w'>"};
    if (parameter.get_bool("perform_spanwise_average"))
      var_name = {"x", "y", "<rho>", "<u>", "<v>", "<w>", "<p>", "<T>", "<u'u'>", "<v'v'>",
                  "<w'w'>", "<u'v'>", "<u'w'>", "<v'w'>"};

    auto ud_names = UserDefineStat::namelistStat();
    for (auto &nn: ud_names) {
      var_name.emplace_back(nn);
      ++n_plot;
    }
    n_plot = acquire_variable_names<mix_model, turb>(var_name, species);
    MPI_File_write_at(fp, offset, &n_plot, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: var_name) {
      gxl::write_str(name.c_str(), fp, offset);
    }

    // iv. Zones
    for (int i = 0; i < mesh.n_block_total; ++i) {
      // 1. Zone marker. Value = 299.0, indicates a V112 header.
      constexpr float zone_marker{299.0f};
      MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
      offset += 4;
      // 2. Zone name.
      gxl::write_str(("zone " + std::to_string(i)).c_str(), fp, offset);
      // 3. Parent zone. No longer used
      constexpr int32_t parent_zone{-1};
      MPI_File_write_at(fp, offset, &parent_zone, 1, MPI_INT32_T, &status);
      offset += 4;
      // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
      constexpr int32_t strand_id{-2};
      MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
      offset += 4;
      // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
//      offset_solution_time[i] = offset;
      constexpr double solution_time{0};
      MPI_File_write_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
      offset += 8;
      // 6. Default Zone Color. Seldom used. Set to -1.
      constexpr int32_t zone_color{-1};
      MPI_File_write_at(fp, offset, &zone_color, 1, MPI_INT32_T, &status);
      offset += 4;
      // 7. ZoneType 0=ORDERED
      constexpr int32_t zone_type{0};
      MPI_File_write_at(fp, offset, &zone_type, 1, MPI_INT32_T, &status);
      offset += 4;
      // 8. Specify Var Location. 0 = All data is located at the nodes
      constexpr int32_t var_location{0};
      MPI_File_write_at(fp, offset, &var_location, 1, MPI_INT32_T, &status);
      offset += 4;
      // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
      // raw face neighbors are not defined for these zone types.
      constexpr int32_t raw_face_neighbor{0};
      MPI_File_write_at(fp, offset, &raw_face_neighbor, 1, MPI_INT32_T, &status);
      offset += 4;
      // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
      constexpr int32_t miscellaneous_face{0};
      MPI_File_write_at(fp, offset, &miscellaneous_face, 1, MPI_INT32_T, &status);
      offset += 4;
      // For ordered zone, specify IMax, JMax, KMax
      const auto mx{mesh.mx_blk[i]}, my{mesh.my_blk[i]};
      auto mz{mesh.mz_blk[i]};
      if (parameter.get_bool("perform_spanwise_average"))
        mz = 1;
      MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
      offset += 4;

      // 11. For all zone types (repeat for each Auxiliary data name/value pair)
      // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
      // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
      // No more data
      constexpr int32_t no_more_auxi_data{0};
      MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
      offset += 4;
    }

    // End of Header
    constexpr float EOHMARKER{357.0f};
    MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
    offset += 4;

    offset_header = offset;
  }
  MPI_Bcast(&offset_header, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_plot, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  MPI_Offset new_offset{0};
  int i_blk{0};
  for (int p = 0; p < myid; ++p) {
    const int n_blk = mesh.nblk[p];
    for (int b = 0; b < n_blk; ++b) {
      new_offset += 16 + 20 * n_plot;
      const int mx{mesh.mx_blk[i_blk]}, my{mesh.my_blk[i_blk]};
      int mz{mesh.mz_blk[i_blk]};
      if (parameter.get_bool("perform_spanwise_average"))
        mz = 1;
      const int64_t N = mx * my * mz;
      // We always write double precision out
      new_offset += n_plot * N * 8;
      ++i_blk;
    }
  }
  offset_header += new_offset;

  const auto n_block{mesh.n_block};
  offset_minmax_var = new MPI_Offset[n_block];
  offset_var = new MPI_Offset[n_block];

  offset = offset_header;
  for (int blk = 0; blk < n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_plot; ++l) {
      MPI_File_write_at(fp, offset, &data_format, 1, MPI_INT32_T, &status);
      offset += 4;
    }
    // 3. Has passive variables: 0 = no, 1 = yes.
    constexpr int32_t passive_var{0};
    MPI_File_write_at(fp, offset, &passive_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Has variable sharing 0 = no, 1 = yes.
    constexpr int32_t shared_var{0};
    MPI_File_write_at(fp, offset, &shared_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto &b{mesh[blk]};
    const auto mx{b.mx}, my{b.my};
    auto mz{b.mz};
    if (parameter.get_bool("perform_spanwise_average"))
      mz = 1;

    double min_val{b.x(0, 0, 0)}, max_val{b.x(0, 0, 0)};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          min_val = std::min(min_val, b.x(i, j, k));
          max_val = std::max(max_val, b.x(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.y(0, 0, 0);
    max_val = b.y(0, 0, 0);
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          min_val = std::min(min_val, b.y(i, j, k));
          max_val = std::max(max_val, b.y(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    if (!parameter.get_bool("perform_spanwise_average")) {
      min_val = b.z(0, 0, 0);
      max_val = b.z(0, 0, 0);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, b.z(i, j, k));
            max_val = std::max(max_val, b.z(i, j, k));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }

    // Then, the max/min values of variables should be printed.
    offset_minmax_var[blk] = offset;
    if (parameter.get_bool("perform_spanwise_average"))
      offset += 16 * (n_plot - 2);
    else
      offset += 16 * (n_plot - 3);

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx, my, mz};

    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, b.mz + 2 * b.ngg};
    int start_idx[3]{b.ngg, b.ngg, b.ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    if (!parameter.get_bool("perform_spanwise_average")) {
      MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
      offset += memsz;
    }

    // Then, the variables are outputted.
    offset_var[blk] = offset;
    if (parameter.get_bool("perform_spanwise_average"))
      offset += memsz * (n_plot - 2);
    else
      offset += memsz * (n_plot - 3);

    MPI_Type_free(&ty);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_File_close(&fp);
}

template<MixtureModel mix_model, class turb>
int32_t StatisticsCollector::acquire_variable_names(std::vector<std::string> &var_name, const Species &species) {
  auto nv = (int) (var_name.size()); // x,y,z + rho,u,v,w,p,T,Mach
  const auto nv_old = nv;
  if constexpr (mix_model != MixtureModel::Air) {
    nv += parameter.get_int("n_spec"); // Y_k
    var_name.resize(nv);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      var_name[ind + nv_old] = "<" + name + ">";
    }
  }
  if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SA) {
    nv += 1; // SA variable?
  } else if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
    nv += 2; // k, omega
    var_name.emplace_back("<tke>");
    var_name.emplace_back("<omega>");
  }
  if constexpr (mix_model == MixtureModel::FL || mix_model == MixtureModel::MixtureFraction) {
    nv += 2; // Z, Z_prime, chi
    var_name.emplace_back("<Z>");
    var_name.emplace_back("<Z''>");
  }
  return nv;
}

template<MixtureModel mix_model, class turb_method>
void StatisticsCollector::collect_data(DParameter *param) {
  ++counter;
  for (int l = 0; l < UserDefineStat::n_collect; ++l) {
    ++counter_ud[l];
  }

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    collect_statistics<<<bpg, tpb>>>(field[b].d_ptr, param);
  }
}

MPI_Offset write_ud_stat_max_min(MPI_Offset offset, const Field &field, MPI_File &fp, int mz);

MPI_Offset write_ud_stat_data(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, int64_t mem_sz);
} // cfd
