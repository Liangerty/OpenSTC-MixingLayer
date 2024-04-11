//#include "Stat.cuh"
//#include <filesystem>
//
//namespace cfd {
//Stat::Stat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field) :
//    if_collect_statistics{_parameter.get_bool("if_collect_statistics")}, myid{_parameter.get_int("myid")},
//    parameter(_parameter), mesh{_mesh}, field{_field},
//    collect_reynolds_average{_parameter.get_bool("collect_reynolds_average")} {
//  if (!if_collect_statistics) {
//    return;
//  }
//
//  const std::filesystem::path out_dir("output/stat");
//  if (!exists(out_dir))
//    create_directories(out_dir);
//
////  ud_stat_name = acquire_user_defined_statistical_data_name();
////  for (integer l = 0; l < UserDefineStat::n_stat; ++l) {
////    counter_ud[l] = 0;
////  }
//}
//
//void Stat::compute_offset_for_export_data() {
//  integer n_block = 0;
//  for (integer p = 0; p < parameter.get_int("myid"); ++p) {
//    n_block += mesh.nblk[p];
//  }
//  for (integer b = 0; b < n_block; ++b) {
//    MPI_Offset sz = mesh.mx_blk[b] * mesh.my_blk[b] * mesh.mz_blk[b] * 8;
//    offset_unit[0] += sz * (6 + parameter.get_int("n_scalar")) + 4 * 3;
//    offset_unit[1] += sz * 6 + 4 * 3;
//    offset_unit[2] += sz + 4 * 3;
//  }
//}
//
//void Stat::read_previous_statistical_data() {
//  const std::filesystem::path out_dir("output/stat");
//  MPI_File fp1, fp2;
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/1st-order_statistics.bin").c_str(),
//                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp1);
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/2nd-order_statistics.bin").c_str(),
//                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp2);
//  MPI_Status status;
//
//  MPI_File_read_at(fp1, 0, &counter, 1, MPI_INT32_T, &status);
//  MPI_Offset offset1 = offset_unit[0];
//  MPI_Offset offset2 = offset_unit[1];
//  for (integer b = 0; b < mesh.n_block; ++b) {
//    integer mx, my, mz;
//    MPI_File_read_at(fp1, offset1, &mx, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    MPI_File_read_at(fp1, offset1, &my, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    MPI_File_read_at(fp1, offset1, &mz, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
//      printf(
//          "The mesh size in the statistical data file 1st-order_statistics.bin is not consistent with the current mesh size.\n");
//      exit(1);
//    }
//    const auto size = (long long) (mx * my * mz * sizeof(real));
//    MPI_Datatype ty;
//    integer lsize[3]{mx, my, mz};
//    integer start_idx[3]{0, 0, 0};
//    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
//    MPI_Type_commit(&ty);
//    MPI_File_read_at(fp1, offset1, field[b].firstOrderMoment.data(), 6 + parameter.get_int("n_scalar"), ty, &status);
//    offset1 += size * (6 + parameter.get_int("n_scalar"));
//    cudaMemcpy(field[b].h_ptr->firstOrderMoment.data(), field[b].firstOrderMoment.data(),
//               size * (6 + parameter.get_int("n_scalar")), cudaMemcpyHostToDevice);
//
//    MPI_File_read_at(fp2, offset2, &mx, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    MPI_File_read_at(fp2, offset2, &my, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    MPI_File_read_at(fp2, offset2, &mz, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
//      printf(
//          "The mesh size in the statistical data file 2nd-order_velocity_statistics.bin is not consistent with the current mesh size.\n");
//      exit(1);
//    }
//    MPI_File_read_at(fp2, offset2, field[b].secondOrderMoment.data(), 6, ty, &status);
//    offset2 += size * 6;
//    cudaMemcpy(field[b].h_ptr->velocity2ndMoment.data(), field[b].secondOrderMoment.data(),
//               size * 6, cudaMemcpyHostToDevice);
//  }
//
//  // Next, let us care about the user-defined part. The data to be collected may vary from one simulation to another.
//  // So we need to check the consistency of the user-defined statistical data.
////  for (integer l = 0; l < UserDefineStat::n_stat; ++l) {
////    std::string file_name = "output/stat/" + ud_stat_name[l] + ".bin";
////    if (std::filesystem::exists(file_name)) {
////      MPI_File fp_ud;
////      MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_ud);
////      MPI_File_read_at(fp_ud, 0, &counter_ud[l], 1, MPI_INT32_T, &status);
////      MPI_Offset offset_ud = offset_unit[2];
////      for (integer b = 0; b < mesh.n_block; ++b) {
////        integer mx, my, mz;
////        MPI_File_read_at(fp_ud, offset_ud, &mx, 1, MPI_INT32_T, &status);
////        offset_ud += 4;
////        MPI_File_read_at(fp_ud, offset_ud, &my, 1, MPI_INT32_T, &status);
////        offset_ud += 4;
////        MPI_File_read_at(fp_ud, offset_ud, &mz, 1, MPI_INT32_T, &status);
////        offset_ud += 4;
////        if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
////          printf("The mesh size in the statistical data file %s is not consistent with the current mesh size.\n",
////                 file_name.c_str());
////          exit(1);
////        }
////        const auto size = (long long) (mx * my * mz * sizeof(real));
////        MPI_Datatype ty;
////        integer lsize[3]{mx, my, mz};
////        integer start_idx[3]{0, 0, 0};
////        MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
////        MPI_Type_commit(&ty);
////        MPI_File_read_at(fp_ud, offset_ud, field[b].userDefinedStatistics[l], 1, ty, &status);
////        offset_ud += size;
////        cudaMemcpy(field[b].h_ptr->userDefinedStatistics[l], field[b].userDefinedStatistics[l],
////                   size, cudaMemcpyHostToDevice);
////      }
////    }
////  }
//}
//
//void Stat::export_statistical_data(DParameter *param) {
//  const std::filesystem::path out_dir("output/stat");
//  MPI_File fp1, fp2;//, fp_ud[UserDefineStat::n_stat];
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/favre_1order_statistics.bin").c_str(),
//                MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                MPI_INFO_NULL, &fp1);
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/favre_2order_statistics.bin").c_str(),
//                MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                MPI_INFO_NULL, &fp2);
//  MPI_File fp_rey1, fp_rey2;
//  if (collect_reynolds_average){
//    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/reynolds_1order_statistics.bin").c_str(),
//                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                  MPI_INFO_NULL, &fp_rey1);
//    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/reynolds_2order_statistics.bin").c_str(),
//                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                  MPI_INFO_NULL, &fp_rey2);
//  }
////  for (integer l = 0; l < UserDefineStat::n_stat; ++l) {
////    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/" + ud_stat_name[l] + ".bin").c_str(),
////                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
////                  MPI_INFO_NULL, &fp_ud[l]);
////  }
//  MPI_Status status;
//
//  const integer n_scalar = parameter.get_int("n_scalar");
//
//  if (myid == 0) {
//    const integer n_block = mesh.n_block_total;
//    const integer n_var = 6 + n_scalar;
//    MPI_File_write_at(fp1, 0, &counter, 1, MPI_INT32_T, &status);
//    MPI_File_write_at(fp1, 4, &n_block, 1, MPI_INT32_T, &status);
//    MPI_File_write_at(fp1, 8, &n_var, 1, MPI_INT32_T, &status);
//    MPI_File_write_at(fp2, 0, &counter, 1, MPI_INT32_T, &status);
//    MPI_File_write_at(fp2, 4, &n_block, 1, MPI_INT32_T, &status);
//    constexpr integer nine = 9;
//    MPI_File_write_at(fp2, 8, &nine, 1, MPI_INT32_T, &status);
//
//    if (collect_reynolds_average) {
//      const int n_v = 4 + n_scalar;
//      MPI_File_write_at(fp_rey1, 0, &counter, 1, MPI_INT32_T, &status);
//      MPI_File_write_at(fp_rey1, 4, &n_block, 1, MPI_INT32_T, &status);
//      MPI_File_write_at(fp_rey1, 8, &n_v, 1, MPI_INT32_T, &status);
//      MPI_File_write_at(fp_rey2, 0, &counter, 1, MPI_INT32_T, &status);
//      MPI_File_write_at(fp_rey2, 4, &n_block, 1, MPI_INT32_T, &status);
//      constexpr integer seven = 7;
//      MPI_File_write_at(fp_rey2, 8, &seven, 1, MPI_INT32_T, &status);
//    }
////    for (auto &fp: fp_ud) {
////      MPI_File_write_at(fp, 0, &counter, 1, MPI_INT32_T, &status);
////      MPI_File_write_at(fp, 4, &n_block, 1, MPI_INT32_T, &status);
////      MPI_File_write_at(fp, 8, &UserDefineStat::n_stat, 1, MPI_INT32_T, &status);
////    }
//  }
//
//  MPI_Offset offset1 = offset_unit[0];
//  MPI_Offset offset2 = offset_unit[1];
////  MPI_Offset offset_ud[UserDefineStat::n_stat];
////  for (auto &l: offset_ud) {
////    l = offset_unit[2];
////  }
//  for (integer b = 0; b < mesh.n_block; ++b) {
//    auto &zone = field[b].h_ptr;
//    const integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
//    const auto size = (long long) (mx * my * mz * sizeof(real));
//
//    cudaMemcpy(field[b].firstOrderMoment.data(), zone->rhoP_sum.data(),
//               size * 2, cudaMemcpyDeviceToHost);
//    cudaMemcpy(field[b].firstOrderMoment[2], zone->favre_1_sum.data(),
//               size * (4 + n_scalar), cudaMemcpyDeviceToHost);
//    cudaMemcpy(field[b].secondOrderMoment.data(), zone->favre_2_sum.data(),
//               size * 9, cudaMemcpyDeviceToHost);
////    cudaMemcpy(field[b].userDefinedStatistics.data(), zone->userDefinedStatistics.data(),
////               size * UserDefineStat::n_stat, cudaMemcpyDeviceToHost);
//
//
//    MPI_Datatype ty;
//    integer lsize[3]{mx, my, mz};
//    integer start_idx[3]{0, 0, 0};
//    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
//    MPI_Type_commit(&ty);
//
//    // First, 1st order statistics
//    MPI_File_write_at(fp1, offset1, &mx, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    MPI_File_write_at(fp1, offset1, &my, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    MPI_File_write_at(fp1, offset1, &mz, 1, MPI_INT32_T, &status);
//    offset1 += 4;
//    MPI_File_write_at(fp1, offset1, field[b].firstOrderMoment.data(), 6 + n_scalar, ty, &status);
//    offset1 += size * (6 + n_scalar);
//
//    // Next, 2nd order velocity statistics
//    MPI_File_write_at(fp2, offset2, &mx, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    MPI_File_write_at(fp2, offset2, &my, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    MPI_File_write_at(fp2, offset2, &mz, 1, MPI_INT32_T, &status);
//    offset2 += 4;
//    MPI_File_write_at(fp2, offset2, field[b].secondOrderMoment.data(), 9, ty, &status);
//    offset2 += size * 9;
//
//    // Last, write out all user-defined statistical data.
////    for (integer l = 0; l < UserDefineStat::n_stat; ++l) {
////      MPI_File_write_at(fp_ud[l], offset_ud[l], &mx, 1, MPI_INT32_T, &status);
////      offset_ud[l] += 4;
////      MPI_File_write_at(fp_ud[l], offset_ud[l], &my, 1, MPI_INT32_T, &status);
////      offset_ud[l] += 4;
////      MPI_File_write_at(fp_ud[l], offset_ud[l], &mz, 1, MPI_INT32_T, &status);
////      offset_ud[l] += 4;
////      MPI_File_write_at(fp_ud[l], offset_ud[l], field[b].userDefinedStatistics[l], 1, ty, &status);
////      offset_ud[l] += size;
////    }
//
//    MPI_Type_free(&ty);
//  }
//
//  MPI_File_close(&fp1);
//  MPI_File_close(&fp2);
////  for (auto &fp: fp_ud) {
////    MPI_File_close(&fp);
////  }
//
//  plot_statistical_data(param);
//}
//
//__global__ void compute_statistical_data(DZone *zone, DParameter *param, integer counter, const integer *counter_ud) {
//  const integer extent[3]{zone->mx, zone->my, zone->mz};
//  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
//  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
//  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;
//
//  auto &mean = zone->mean_value;
//  auto &firstOrderStat = zone->favre_1_sum;
//
//  mean(i, j, k, 0) = zone->rhoP_sum(i, j, k, 0) / counter; // <rho>
//  mean(i, j, k, 1) = zone->rhoP_sum(i, j, k, 1) / counter; // <p>
//  const auto den = 1.0 / (mean(i, j, k, 0) * counter);
//  mean(i, j, k, 2) = firstOrderStat(i, j, k, 0) * den; // <u>
//  mean(i, j, k, 3) = firstOrderStat(i, j, k, 1) * den; // <v>
//  mean(i, j, k, 4) = firstOrderStat(i, j, k, 2) * den; // <w>
//  mean(i, j, k, 5) = firstOrderStat(i, j, k, 3) * den; // <T>
//  for (integer l = 0; l < param->n_scalar; ++l) {
//    mean(i, j, k, l + 6) = firstOrderStat(i, j, k, l + 4) * den;
//  }
//
//  auto &rey_tensor = zone->reynolds_stress_tensor;
//  auto &rms = zone->rms_value;
//  auto &favre_2_sum = zone->favre_2_sum;
//  rms(i, j, k, 0) = sqrt(favre_2_sum(i, j, k, 0) / counter - mean(i, j, k, 0) * mean(i, j, k, 0)); // rho'_rms
//  rms(i, j, k, 1) = sqrt(favre_2_sum(i, j, k, 1) / counter - mean(i, j, k, 1) * mean(i, j, k, 1)); // p'_rms
//  rms(i, j, k, 2) = sqrt(
//      favre_2_sum(i, j, k, 2) * den - mean(i, j, k, 5) * mean(i, j, k, 5)); // T''_rms
//  rey_tensor(i, j, k, 0) = favre_2_sum(i, j, k, 3) * den - mean(i, j, k, 2) * mean(i, j, k, 2);
//  rey_tensor(i, j, k, 1) = favre_2_sum(i, j, k, 4) * den - mean(i, j, k, 3) * mean(i, j, k, 3);
//  rey_tensor(i, j, k, 2) = favre_2_sum(i, j, k, 5) * den - mean(i, j, k, 4) * mean(i, j, k, 4);
//  rey_tensor(i, j, k, 3) = favre_2_sum(i, j, k, 6) * den - mean(i, j, k, 2) * mean(i, j, k, 3);
//  rey_tensor(i, j, k, 4) = favre_2_sum(i, j, k, 7) * den - mean(i, j, k, 2) * mean(i, j, k, 4);
//  rey_tensor(i, j, k, 5) = favre_2_sum(i, j, k, 8) * den - mean(i, j, k, 3) * mean(i, j, k, 4);
//
////  compute_user_defined_statistical_data(zone, param, counter_ud, i, j, k, counter);
//}
//
//void Stat::plot_statistical_data(DParameter *param) {
//  // First, compute the statistical data.
//  dim3 tpb{8, 8, 4};
//  if (mesh.dimension == 2) {
//    tpb = {16, 16, 1};
//  }
//
////  cudaMemcpy(counter_ud_device, counter_ud.data(), sizeof(integer) * UserDefineStat::n_stat, cudaMemcpyHostToDevice);
//  const integer n_scalar{parameter.get_int("n_scalar")};
//  for (integer b = 0; b < mesh.n_block; ++b) {
//    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
//    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
//    compute_statistical_data<<<bpg, tpb>>>(field[b].d_ptr, param, counter, counter_ud_device);
//    auto sz = mx * my * mz * sizeof(real);
//    cudaDeviceSynchronize();
//    cudaMemcpy(field[b].mean_value.data(), field[b].h_ptr->mean_value.data(), sz * (6 + n_scalar),
//               cudaMemcpyDeviceToHost);
//    cudaMemcpy(field[b].reynolds_stress_tensor_and_rms.data(), field[b].h_ptr->rms_value.data(), sz * 3,
//               cudaMemcpyDeviceToHost);
//    cudaMemcpy(field[b].reynolds_stress_tensor_and_rms[3], field[b].h_ptr->reynolds_stress_tensor.data(), sz * 6,
//               cudaMemcpyDeviceToHost);
////    cudaMemcpy(field[b].user_defined_statistical_data.data(), field[b].h_ptr->user_defined_statistical_data.data(),
////               sz * UserDefineStat::n_user_stat,
////               cudaMemcpyDeviceToHost);
//  }
//
//  // Next, transfer the data to CPU
//
//  // Next, output them.
//  const std::filesystem::path out_dir("output/stat");
//  MPI_File fp;
//  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
//                MPI_INFO_NULL, &fp);
//  MPI_Status status;
//
//  // II. Data Section
//  for (int blk = 0; blk < mesh.n_block; ++blk) {
//    // First, modify the new min/max values of the variables
//    MPI_Offset offset = offset_minmax_var[blk];
//
//    double min_val{0}, max_val{1};
//    auto &b{mesh[blk]};
//    auto &v{field[blk]};
//    const auto mx{b.mx}, my{b.my}, mz{b.mz};
//    for (int l = 0; l < 6 + n_scalar; ++l) {
//      min_val = v.mean_value(0, 0, 0, l);
//      max_val = v.mean_value(0, 0, 0, l);
//      for (int k = 0; k < mz; ++k) {
//        for (int j = 0; j < my; ++j) {
//          for (int i = 0; i < mx; ++i) {
//            min_val = std::min(min_val, v.mean_value(i, j, k, l));
//            max_val = std::max(max_val, v.mean_value(i, j, k, l));
//          }
//        }
//      }
//      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
//      offset += 8;
//      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
//      offset += 8;
//    }
//    for (int l = 0; l < 9; ++l) {
//      min_val = v.reynolds_stress_tensor_and_rms(0, 0, 0, l);
//      max_val = v.reynolds_stress_tensor_and_rms(0, 0, 0, l);
//      for (int k = 0; k < mz; ++k) {
//        for (int j = 0; j < my; ++j) {
//          for (int i = 0; i < mx; ++i) {
//            min_val = std::min(min_val, v.reynolds_stress_tensor_and_rms(i, j, k, l));
//            max_val = std::max(max_val, v.reynolds_stress_tensor_and_rms(i, j, k, l));
//          }
//        }
//      }
//      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
//      offset += 8;
//      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
//      offset += 8;
//    }
////    offset = write_ud_stat_max_min(offset, v, fp);
//
//    // 7. Zone Data.
//    MPI_Datatype ty;
//    integer lsize[3]{mx, my, mz};
//    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
//    integer start_idx[3]{0, 0, 0};
//    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
//    MPI_Type_commit(&ty);
//
//    offset = offset_var[blk];
//    for (int l = 0; l < 6 + n_scalar; ++l) {
//      auto var = v.mean_value[l];
//      MPI_File_write_at(fp, offset, var, 1, ty, &status);
//      offset += memsz;
//    }
//    for (int l = 0; l < 9; ++l) {
//      auto var = v.reynolds_stress_tensor_and_rms[l];
//      MPI_File_write_at(fp, offset, var, 1, ty, &status);
//      offset += memsz;
//    }
////    offset = write_ud_stat_data(offset, v, fp, ty, memsz);
//    MPI_Type_free(&ty);
//  }
//  MPI_File_close(&fp);
//}
//
//__global__ void collect_statistics(DZone *zone, DParameter *param) {
//  const integer extent[3]{zone->mx, zone->my, zone->mz};
//  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
//  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
//  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
//  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;
//
//  const auto &bv = zone->bv;
//
//  // The first order statistics of the flow field
//  zone->rhoP_sum(i, j, k, 0) += bv(i, j, k, 0); // rho
//  zone->rhoP_sum(i, j, k, 1) += bv(i, j, k, 4); // p
//
//  auto &favre_1_sum = zone->favre_1_sum;
//  favre_1_sum(i, j, k, 0) += bv(i, j, k, 0) * bv(i, j, k, 1); // rho*u
//  favre_1_sum(i, j, k, 1) += bv(i, j, k, 0) * bv(i, j, k, 2); // rho*v
//  favre_1_sum(i, j, k, 2) += bv(i, j, k, 0) * bv(i, j, k, 3); // rho*w
//  favre_1_sum(i, j, k, 3) += bv(i, j, k, 0) * bv(i, j, k, 5); // rho*T
//  const auto &sv = zone->sv;
//  for (integer l = 0; l < param->n_scalar; ++l) {
//    favre_1_sum(i, j, k, l + 4) += bv(i, j, k, 0) * sv(i, j, k, l);
//  }
//
//  // The second order statistics of the flow field, including rms and velocity correlation tensors
//  auto &favre_2_sum = zone->favre_2_sum;
//  favre_2_sum(i, j, k, 0) += bv(i, j, k, 0) * bv(i, j, k, 0); // rho*rho
//  favre_2_sum(i, j, k, 1) += bv(i, j, k, 4) * bv(i, j, k, 4); // p*p
//  favre_2_sum(i, j, k, 2) += bv(i, j, k, 0) * bv(i, j, k, 5) * bv(i, j, k, 5); // rho*T*T
//  favre_2_sum(i, j, k, 3) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 1); // rho*u*u
//  favre_2_sum(i, j, k, 4) += bv(i, j, k, 0) * bv(i, j, k, 2) * bv(i, j, k, 2); // rho*v*v
//  favre_2_sum(i, j, k, 5) += bv(i, j, k, 0) * bv(i, j, k, 3) * bv(i, j, k, 3); // rho*w*w
//  favre_2_sum(i, j, k, 6) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 2); // rho*u*v
//  favre_2_sum(i, j, k, 7) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 3); // rho*u*w
//  favre_2_sum(i, j, k, 8) += bv(i, j, k, 0) * bv(i, j, k, 2) * bv(i, j, k, 3); // rho*v*w
//
//  if (param->collect_reynolds_average) {
//    auto &ensemble_1_sum = zone->ensemble_1_sum;
//    ensemble_1_sum(i, j, k, 0) += bv(i, j, k, 1); // u
//    ensemble_1_sum(i, j, k, 1) += bv(i, j, k, 2); // v
//    ensemble_1_sum(i, j, k, 2) += bv(i, j, k, 3); // w
//    ensemble_1_sum(i, j, k, 3) += bv(i, j, k, 5); // T
//    for (integer l = 0; l < param->n_scalar; ++l) {
//      ensemble_1_sum(i, j, k, l + 4) += sv(i, j, k, l);
//    }
//
//    if (param->collect_reynolds_rms) {
//      auto &ensemble_2_sum = zone->ensemble_2_sum;
//      ensemble_2_sum(i, j, k, 0) += bv(i, j, k, 5) * bv(i, j, k, 5); // T*T
//      ensemble_2_sum(i, j, k, 1) += bv(i, j, k, 1) * bv(i, j, k, 1); // u*u
//      ensemble_2_sum(i, j, k, 2) += bv(i, j, k, 2) * bv(i, j, k, 2); // v*v
//      ensemble_2_sum(i, j, k, 3) += bv(i, j, k, 3) * bv(i, j, k, 3); // w*w
//      ensemble_2_sum(i, j, k, 4) += bv(i, j, k, 1) * bv(i, j, k, 2); // u*v
//      ensemble_2_sum(i, j, k, 5) += bv(i, j, k, 1) * bv(i, j, k, 3); // u*w
//      ensemble_2_sum(i, j, k, 6) += bv(i, j, k, 2) * bv(i, j, k, 3); // v*w
//    }
//  }
//
//  // Collect user-defined statistics
////  collect_user_defined_statistics(zone, param, i, j, k);
//}
//
//} // cfd