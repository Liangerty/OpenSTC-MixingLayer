//#pragma once
//
//#include "Parameter.h"
//#include "Mesh.h"
//#include "Field.h"
//#include "DParameter.cuh"
//#include <mpi.h>
//#include <filesystem>
//#include "gxl_lib/MyString.h"
//#include "ChemData.h"
//#include "TurbMethod.hpp"
//
//namespace cfd {
//
//__global__ void collect_statistics(DZone *zone, DParameter *param);
//
//class Stat {
//public:
//  explicit Stat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field);
//
//  template<MixtureModel mix_model, class turb>
//  void initialize_statistics_collector(const Species &species);
//
//  template<MixtureModel mix_model, class turb_method>
//  void collect_data(DParameter *param);
//
//  void export_statistical_data(DParameter *param);
//
//private:
//  // Basic info
//  bool if_collect_statistics{false};
//  integer myid{0};
//  // Data to be bundled
//  const Parameter &parameter;
//  const Mesh &mesh;
//  std::vector<Field> &field;
//  integer counter{0};
//
//  // About which data to be collected
//  bool collect_reynolds_average{false};
//
//  // Offset unit.
//  // The first number in the file is the number of blocks, which would occupy 4 bytes.
//  // The second is the number of variables, which also occupy 4 bytes.
//  MPI_Offset offset_unit[3]{12, 12, 12};
//
//  // Plot related
//  integer n_plot{0};
//  MPI_Offset offset_header{0};
//  MPI_Offset *offset_minmax_var = nullptr;
//  MPI_Offset *offset_var = nullptr;
//
//  // Available variables index
//  // This is the index of Favre rms variables, which only contains sv index(bv's are collected by default)
//  std::vector<integer> favre_rms_index_sv;
//  integer *favre_rms_index_sv_d = nullptr;
//
//private:
//  void compute_offset_for_export_data();
//
//  void read_previous_statistical_data();
//
//  template<MixtureModel mix_model, class turb>
//  void prepare_for_statistical_data_plot(const Species &species);
//
//  template<MixtureModel mix_model, class turb>
//  int32_t
//  acquire_variable_names(std::vector<std::string> &var_name, const Species &species);
//
//  template<MixtureModel mix_model, class turb>
//  int32_t
//  acquire_avail_variable_names(std::vector<std::string> &var_name, const Species &species);
//
//  void plot_statistical_data(DParameter *param);
//};
//
//template<MixtureModel mix_model, class turb>
//void Stat::initialize_statistics_collector(const Species &species) {
//  compute_offset_for_export_data();
//
//  if (parameter.get_bool("if_continue_collect_statistics")) {
//    read_previous_statistical_data();
//  }
//
//  prepare_for_statistical_data_plot<mix_model, turb>(species);
////  cudaMalloc(&counter_ud_device, sizeof(integer) * UserDefineStat::n_stat);
//}
//
//template<MixtureModel mix_model, class turb>
//void Stat::prepare_for_statistical_data_plot(const Species &species) {
//  const std::filesystem::path out_dir("output/stat");
//  MPI_File fp;
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                MPI_INFO_NULL, &fp);
//  MPI_Status status;
//
//  // I. Header section
//
//  // Each file should have only one header; thus we let process 0 to write it.
//
//  MPI_Offset offset{0};
//  std::vector<std::string> var_name{"x", "y", "z", "<<greek>r</greek>>", "<p>"};
////    n_plot = add_user_defined_statistical_data_name(var_name);
//  n_plot = acquire_variable_names<mix_model, turb>(var_name, species);
//  if (myid == 0) {
//    // i. Magic number, Version number
//    // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
//    // difference is related to us. For common use, we use V112.
//    constexpr auto magic_number{"#!TDV112"};
//    gxl::write_str_without_null(magic_number, fp, offset);
//
//    // ii. Integer value of 1
//    constexpr int32_t byte_order{1};
//    MPI_File_write_at(fp, offset, &byte_order, 1, MPI_INT32_T, &status);
//    offset += 4;
//
//    // iii. Title and variable names.
//    // 1. FileType: 0=full, 1=grid, 2=solution.
//    constexpr int32_t file_type{0};
//    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
//    offset += 4;
//    // 2. Title
//    gxl::write_str("Solution file", fp, offset);
//    // 3. Number of variables in the datafile
//    MPI_File_write_at(fp, offset, &n_plot, 1, MPI_INT32_T, &status);
//    offset += 4;
//    // 4. Variable names.
//    for (auto &name: var_name) {
//      gxl::write_str(name.c_str(), fp, offset);
//    }
//
//    // iv. Zones
//    for (int i = 0; i < mesh.n_block_total; ++i) {
//      // 1. Zone marker. Value = 299.0, indicates a V112 header.
//      constexpr float zone_marker{299.0f};
//      MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
//      offset += 4;
//      // 2. Zone name.
//      gxl::write_str(("zone " + std::to_string(i)).c_str(), fp, offset);
//      // 3. Parent zone. No longer used
//      constexpr int32_t parent_zone{-1};
//      MPI_File_write_at(fp, offset, &parent_zone, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
//      constexpr int32_t strand_id{-2};
//      MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
//      constexpr double solution_time{0};
//      MPI_File_write_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
//      offset += 8;
//      // 6. Default Zone Color. Seldom used. Set to -1.
//      constexpr int32_t zone_color{-1};
//      MPI_File_write_at(fp, offset, &zone_color, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 7. ZoneType 0=ORDERED
//      constexpr int32_t zone_type{0};
//      MPI_File_write_at(fp, offset, &zone_type, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 8. Specify Var Location. 0 = All data is located at the nodes
//      constexpr int32_t var_location{0};
//      MPI_File_write_at(fp, offset, &var_location, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
//      // raw face neighbors are not defined for these zone types.
//      constexpr int32_t raw_face_neighbor{0};
//      MPI_File_write_at(fp, offset, &raw_face_neighbor, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
//      constexpr int32_t miscellaneous_face{0};
//      MPI_File_write_at(fp, offset, &miscellaneous_face, 1, MPI_INT32_T, &status);
//      offset += 4;
//      // For ordered zone, specify IMax, JMax, KMax
//      const auto mx{mesh.mx_blk[i]}, my{mesh.my_blk[i]}, mz{mesh.mz_blk[i]};
//      MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
//      offset += 4;
//      MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
//      offset += 4;
//      MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
//      offset += 4;
//
//      // 11. For all zone types (repeat for each Auxiliary data name/value pair)
//      // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
//      // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
//      // No more data
//      constexpr int32_t no_more_auxi_data{0};
//      MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
//      offset += 4;
//    }
//
//    // End of Header
//    constexpr float EOHMARKER{357.0f};
//    MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
//    offset += 4;
//
//    offset_header = offset;
//  }
//  MPI_Bcast(&offset_header, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
//
//  MPI_Offset new_offset{0};
//  integer i_blk{0};
//  for (int p = 0; p < myid; ++p) {
//    const integer n_blk = mesh.nblk[p];
//    for (int b = 0; b < n_blk; ++b) {
//      new_offset += 16 + 20 * n_plot;
//      const integer mx{mesh.mx_blk[i_blk]}, my{mesh.my_blk[i_blk]}, mz{mesh.mz_blk[i_blk]};
//      const int64_t N = mx * my * mz;
//      // We always write double precision out
//      new_offset += n_plot * N * 8;
//      ++i_blk;
//    }
//  }
//  offset_header += new_offset;
//
//  const auto n_block{mesh.n_block};
//  offset_minmax_var = new MPI_Offset[n_block];
//  offset_var = new MPI_Offset[n_block];
//
//  offset = offset_header;
//  for (int blk = 0; blk < n_block; ++blk) {
//    // 1. Zone marker. Value = 299.0, indicates a V112 header.
//    constexpr float zone_marker{299.0f};
//    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
//    offset += 4;
//    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
//    constexpr int32_t data_format{2};
//    for (int l = 0; l < n_plot; ++l) {
//      MPI_File_write_at(fp, offset, &data_format, 1, MPI_INT32_T, &status);
//      offset += 4;
//    }
//    // 3. Has passive variables: 0 = no, 1 = yes.
//    constexpr int32_t passive_var{0};
//    MPI_File_write_at(fp, offset, &passive_var, 1, MPI_INT32_T, &status);
//    offset += 4;
//    // 4. Has variable sharing 0 = no, 1 = yes.
//    constexpr int32_t shared_var{0};
//    MPI_File_write_at(fp, offset, &shared_var, 1, MPI_INT32_T, &status);
//    offset += 4;
//    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
//    constexpr int32_t shared_connect{-1};
//    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
//    offset += 4;
//    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
//    // For each non-shared and non-passive variable (as specified above):
//    auto &b{mesh[blk]};
//    const auto mx{b.mx}, my{b.my}, mz{b.mz};
//
//    double min_val{b.x(0, 0, 0)}, max_val{b.x(0, 0, 0)};
//    for (int k = 0; k < mz; ++k) {
//      for (int j = 0; j < my; ++j) {
//        for (int i = 0; i < mx; ++i) {
//          min_val = std::min(min_val, b.x(i, j, k));
//          max_val = std::max(max_val, b.x(i, j, k));
//        }
//      }
//    }
//    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//    min_val = b.y(0, 0, 0);
//    max_val = b.y(0, 0, 0);
//    for (int k = 0; k < mz; ++k) {
//      for (int j = 0; j < my; ++j) {
//        for (int i = 0; i < mx; ++i) {
//          min_val = std::min(min_val, b.y(i, j, k));
//          max_val = std::max(max_val, b.y(i, j, k));
//        }
//      }
//    }
//    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//    min_val = b.z(0, 0, 0);
//    max_val = b.z(0, 0, 0);
//    for (int k = 0; k < mz; ++k) {
//      for (int j = 0; j < my; ++j) {
//        for (int i = 0; i < mx; ++i) {
//          min_val = std::min(min_val, b.z(i, j, k));
//          max_val = std::max(max_val, b.z(i, j, k));
//        }
//      }
//    }
//    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
//    offset += 8;
//
//    // Then, the max/min values of variables should be printed.
//    offset_minmax_var[blk] = offset;
//    offset += 16 * (n_plot - 3);
//
//    // 7. Zone Data.
//    MPI_Datatype ty;
//    integer lsize[3]{mx, my, mz};
//    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
//    integer memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
//    integer start_idx[3]{b.ngg, b.ngg, b.ngg};
//    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
//    MPI_Type_commit(&ty);
//    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
//    offset += memsz;
//    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
//    offset += memsz;
//    MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
//    offset += memsz;
//
//    // Then, the variables are outputted.
//    offset_var[blk] = offset;
//    offset += memsz * (n_plot - 3);
//
//    MPI_Type_free(&ty);
//  }
//
//  MPI_Barrier(MPI_COMM_WORLD);
//
//  MPI_File_close(&fp);
//}
//
//template<MixtureModel mix_model, class turb>
//int32_t
//Stat::acquire_avail_variable_names(std::vector<std::string> &var_name, const Species &species) {
//  auto nv = (integer) (var_name.size());
//  // The variables that are available to be collected.
//  const auto &spec_list = species.spec_list;
//
//  // First, about the Favre rms.
//  auto &list1 = parameter.get_string_array("favre_rms_stat");
//  for (auto &name: list1) {
//    auto found = false;
//    auto name1 = gxl::to_upper(name);
//    auto idx=0;
//    if constexpr (mix_model != MixtureModel::Air) {
//      if (spec_list.find(name1) != spec_list.end()) {
//        idx=spec_list.at(name1);
//        found = true;
//      }
//    }
//    if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
//      if (name1 == "TKE") {
//        idx=parameter.get_int("n_spec");
//        found = true;
//      } else if (name1 == "OMEGA") {
//        idx=parameter.get_int("n_spec") + 1;
//        found = true;
//      }
//    }
//    if (found) {
//      ++nv;
//      var_name.emplace_back(name + "<sup><math>\"\"</math></sup><sub>rms</sub>");
//      favre_rms_index_sv.push_back(idx);
//    } else {
//      if (parameter.get_int("myid") == 0)
//        printf("Warning: %s in the list of Favre rms statistics is not found.\n", name.c_str());
//    }
//  }
//  // Next, about the correlation between velocity fluctuation and any scalar phi fluctuation.
//
//  return nv;
//}
//
//template<MixtureModel mix_model, class turb>
//int32_t
//Stat::acquire_variable_names(std::vector<std::string> &var_name, const Species &species) {
//  auto nv = (integer) (var_name.size()); // x,y,z + <rho>,<p>
//  //  Favre average, which is the default option
//  nv += 4; // <u>_F, <v>_F, <w>_F, <T>_F
//  var_name.emplace_back("<u><sub>F</sub>");
//  var_name.emplace_back("<v><sub>F</sub>");
//  var_name.emplace_back("<w><sub>F</sub>");
//  var_name.emplace_back("<T><sub>F</sub>");
//  if constexpr (mix_model != MixtureModel::Air) {
//    const auto nv_old = nv;
//    nv += parameter.get_int("n_spec"); // Y_k
//    var_name.resize(nv);
//    auto &names = species.spec_list;
//    for (auto &[name, ind]: names) {
//      var_name[ind + nv_old] = "<" + name + "><sub>F</sub>";
//    }
//  }
//  if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SA) {
//    nv += 1; // SA variable?
//  } else if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
//    nv += 2; // k, omega
//    var_name.emplace_back("<tke><sub>F</sub>");
//    var_name.emplace_back("<omega><sub>F</sub>");
//  }
//  if constexpr (mix_model == MixtureModel::FL || mix_model == MixtureModel::MixtureFraction) {
//    nv += 2; // Z, Z_prime, chi
//    var_name.emplace_back("<Z><sub>F</sub>");
//    var_name.emplace_back("<Z''><sub>F</sub>");
//  }
//
//  // About rms
//  nv += 2; // rho_rms, p_rms
//  var_name.emplace_back("<greek>r</greek><sup><math>\"</math></sup><sub>rms</sub>");
//  var_name.emplace_back("p<sup><math>\"</math></sup><sub>rms</sub>");
//  ++nv;
//  var_name.emplace_back("T<sup><math>\"\"</math></sup><sub>rms</sub>");
//
//  // About Reynolds stress tensor
//  nv += 6; // uu, vv, ww, uv, uw, vw
//  var_name.emplace_back(R"(<u<sup><math>""</math></sup>u<sup><math>""</math></sup>><sub>F</sub>)");
//  var_name.emplace_back(R"(<v<sup><math>""</math></sup>v<sup><math>""</math></sup>><sub>F</sub>)");
//  var_name.emplace_back(R"(<w<sup><math>""</math></sup>w<sup><math>""</math></sup>><sub>F</sub>)");
//  var_name.emplace_back(R"(<u<sup><math>""</math></sup>v<sup><math>""</math></sup>><sub>F</sub>)");
//  var_name.emplace_back(R"(<u<sup><math>""</math></sup>w<sup><math>""</math></sup>><sub>F</sub>)");
//  var_name.emplace_back(R"(<v<sup><math>""</math></sup>w<sup><math>""</math></sup>><sub>F</sub>)");
//
//  // Reynolds average, not recommended.
//  if (parameter.get_bool("collect_reynolds_average")) {
//    nv += 4; // <u>, <v>, <w>, <T>
//    var_name.emplace_back("<u>");
//    var_name.emplace_back("<v>");
//    var_name.emplace_back("<w>");
//    var_name.emplace_back("<T>");
//    if constexpr (mix_model != MixtureModel::Air) {
//      const auto nv_old = nv;
//      nv += parameter.get_int("n_spec"); // Y_k
//      var_name.resize(nv);
//      auto &names = species.spec_list;
//      for (auto &[name, ind]: names) {
//        var_name[ind + nv_old] = "<" + name + ">";
//      }
//    }
//    if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SA) {
//      nv += 1; // SA variable?
//    } else if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
//      nv += 2; // k, omega
//      var_name.emplace_back("<tke>");
//      var_name.emplace_back("<omega>");
//    }
//    if constexpr (mix_model == MixtureModel::FL || mix_model == MixtureModel::MixtureFraction) {
//      nv += 2; // Z, Z_prime, chi
//      var_name.emplace_back("<Z>");
//      var_name.emplace_back("<Z<sup><math>\"\"</math></sup>>");
//    }
//
//    // About rms and Reynolds stress tensor
//    if (parameter.get_bool("collect_reynolds_rms_stress")) {
//      ++nv;
//      var_name.emplace_back("T<sup><math>\"</math></sup><sub>rms</sub>");
//
//      nv += 6; // uu, vv, ww, uv, uw, vw
//      var_name.emplace_back(R"(<u<sup><math>"</math></sup>u<sup><math>"</math></sup>>)");
//      var_name.emplace_back(R"(<v<sup><math>"</math></sup>v<sup><math>"</math></sup>>)");
//      var_name.emplace_back(R"(<w<sup><math>"</math></sup>w<sup><math>"</math></sup>>)");
//      var_name.emplace_back(R"(<u<sup><math>"</math></sup>v<sup><math>"</math></sup>>)");
//      var_name.emplace_back(R"(<u<sup><math>"</math></sup>w<sup><math>"</math></sup>>)");
//      var_name.emplace_back(R"(<v<sup><math>"</math></sup>w<sup><math>"</math></sup>>)");
//    }
//  }
//
//  // Next, there are many choices supplied by the code.
//  // First, about the second order statistics, such as rms and correlations.
//
//
//  return nv;
//}
//
//template<MixtureModel mix_model, class turb_method>
//void Stat::collect_data(DParameter *param) {
//  ++counter;
////  for (integer l = 0; l < UserDefineStat::n_stat; ++l) {
////    ++counter_ud[l];
////  }
//
//  dim3 tpb{8, 8, 4};
//  if (mesh.dimension == 2) {
//    tpb = {16, 16, 1};
//  }
//
//  for (integer b = 0; b < mesh.n_block; ++b) {
//    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
//    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
//    collect_statistics<<<bpg, tpb>>>(field[b].d_ptr, param);
//  }
//}
//
//} // cfd
