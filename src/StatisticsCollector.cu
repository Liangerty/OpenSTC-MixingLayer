#include "StatisticsCollector.cuh"

namespace cfd {
StatisticsCollector::StatisticsCollector(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field)
    : if_collect_statistics{_parameter.get_bool("if_collect_statistics")},
      start_iter{_parameter.get_int("start_collect_statistics_iter")}, myid{_parameter.get_int("myid")},
      mesh{_mesh}, field{_field}, parameter(_parameter) {
  if (!if_collect_statistics)
    return;

  const std::filesystem::path out_dir("output/stat");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }

  auto collect_name = UserDefineStat::namelistCollect();
  for (int l = 0; l < UserDefineStat::n_collect; ++l) {
    ud_collect_name[l] = collect_name[l];
  }
  for (int l = 0; l < UserDefineStat::n_collect; ++l) {
    counter_ud[l] = 0;
  }
}

void StatisticsCollector::export_statistical_data(DParameter *param, bool perform_spanwise_average) {
  const std::filesystem::path out_dir("output/stat");
  const auto nnn = std::max(1, UserDefineStat::n_collect);
  MPI_File fp1, fp2, fp_ud[nnn];
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/1st-order_statistics.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/2nd-order_velocity_statistics.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp2);
  for (int l = 0; l < UserDefineStat::n_collect; ++l) {
    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/" + ud_collect_name[l] + ".bin").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fp_ud[l]);
  }
  MPI_Status status;

  const int n_scalar = parameter.get_int("n_scalar");

  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    const int n_var = 6 + n_scalar;
    MPI_File_write_at(fp1, 0, &counter, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp1, 4, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp1, 8, &n_var, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp2, 0, &counter, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp2, 4, &n_block, 1, MPI_INT32_T, &status);
    constexpr int six = 6;
    MPI_File_write_at(fp2, 8, &six, 1, MPI_INT32_T, &status);
    if constexpr (UserDefineStat::n_collect > 0) {
      for (auto &fp: fp_ud) {
        MPI_File_write_at(fp, 0, &counter, 1, MPI_INT32_T, &status);
        MPI_File_write_at(fp, 4, &n_block, 1, MPI_INT32_T, &status);
        MPI_File_write_at(fp, 8, &UserDefineStat::n_collect, 1, MPI_INT32_T, &status);
      }
    }
  }

  MPI_Offset offset1 = offset_unit[0];
  MPI_Offset offset2 = offset_unit[1];
  MPI_Offset offset_ud[nnn];
  if constexpr (UserDefineStat::n_collect > 0) {
    for (auto &l: offset_ud) {
      l = offset_unit[2];
    }
  }
  for (int b = 0; b < mesh.n_block; ++b) {
    auto &zone = field[b].h_ptr;
    const int mx{mesh[b].mx + 2}, my{mesh[b].my + 2}, mz{mesh[b].mz + 2};
    const auto size = (long long) (mx * my * mz * sizeof(real));

    cudaMemcpy(field[b].firstOrderMoment.data(), zone->firstOrderMoment.data(),
               size * (6 + n_scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].secondOrderMoment.data(), zone->velocity2ndMoment.data(),
               size * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].userDefinedStatistics.data(), zone->userCollectForStat.data(),
               size * UserDefineStat::n_collect, cudaMemcpyDeviceToHost);


    MPI_Datatype ty;
    int lSize[3]{mx, my, mz};
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    // First, 1st order statistics
    MPI_File_write_at(fp1, offset1, &mx, 1, MPI_INT32_T, &status);
    offset1 += 4;
    MPI_File_write_at(fp1, offset1, &my, 1, MPI_INT32_T, &status);
    offset1 += 4;
    MPI_File_write_at(fp1, offset1, &mz, 1, MPI_INT32_T, &status);
    offset1 += 4;
    MPI_File_write_at(fp1, offset1, field[b].firstOrderMoment.data(), 6 + n_scalar, ty, &status);
    offset1 += size * (6 + n_scalar);

    // Next, 2nd order velocity statistics
    MPI_File_write_at(fp2, offset2, &mx, 1, MPI_INT32_T, &status);
    offset2 += 4;
    MPI_File_write_at(fp2, offset2, &my, 1, MPI_INT32_T, &status);
    offset2 += 4;
    MPI_File_write_at(fp2, offset2, &mz, 1, MPI_INT32_T, &status);
    offset2 += 4;
    MPI_File_write_at(fp2, offset2, field[b].secondOrderMoment.data(), 6, ty, &status);
    offset2 += size * 6;

    // Last, write out all user-defined statistical data.
    for (int l = 0; l < UserDefineStat::n_collect; ++l) {
      MPI_File_write_at(fp_ud[l], offset_ud[l], &mx, 1, MPI_INT32_T, &status);
      offset_ud[l] += 4;
      MPI_File_write_at(fp_ud[l], offset_ud[l], &my, 1, MPI_INT32_T, &status);
      offset_ud[l] += 4;
      MPI_File_write_at(fp_ud[l], offset_ud[l], &mz, 1, MPI_INT32_T, &status);
      offset_ud[l] += 4;
      MPI_File_write_at(fp_ud[l], offset_ud[l], field[b].userDefinedStatistics[l], 1, ty, &status);
      offset_ud[l] += size;
    }

    MPI_Type_free(&ty);
  }

  MPI_File_close(&fp1);
  MPI_File_close(&fp2);
  if constexpr (UserDefineStat::n_collect > 0) {
    for (auto &fp: fp_ud) {
      MPI_File_close(&fp);
    }
  }

  plot_statistical_data(param, perform_spanwise_average);
}

void StatisticsCollector::compute_offset_for_export_data() {
  int n_block = 0;
  for (int p = 0; p < parameter.get_int("myid"); ++p) {
    n_block += mesh.nblk[p];
  }
  for (int b = 0; b < n_block; ++b) {
    MPI_Offset sz = (mesh.mx_blk[b] + 2) * (mesh.my_blk[b] + 2) * (mesh.mz_blk[b] + 2) * 8;
    offset_unit[0] += sz * (6 + parameter.get_int("n_scalar")) + 4 * 3;
    offset_unit[1] += sz * 6 + 4 * 3;
    offset_unit[2] += sz + 4 * 3;
  }
}

void StatisticsCollector::read_previous_statistical_data() {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp1, fp2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/1st-order_statistics.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/2nd-order_velocity_statistics.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp2);
  MPI_Status status;

  MPI_File_read_at(fp1, 0, &counter, 1, MPI_INT32_T, &status);
  MPI_Offset offset1 = offset_unit[0];
  MPI_Offset offset2 = offset_unit[1];
  for (int b = 0; b < mesh.n_block; ++b) {
    int mx, my, mz;
    MPI_File_read_at(fp1, offset1, &mx, 1, MPI_INT32_T, &status);
    offset1 += 4;
    MPI_File_read_at(fp1, offset1, &my, 1, MPI_INT32_T, &status);
    offset1 += 4;
    MPI_File_read_at(fp1, offset1, &mz, 1, MPI_INT32_T, &status);
    offset1 += 4;
    if (mx != mesh[b].mx + 2 || my != mesh[b].my + 2 || mz != mesh[b].mz + 2) {
      printf(
          "The mesh size in the statistical data file 1st-order_statistics.bin is not consistent with the current mesh size.\n");
      exit(1);
    }
    const auto size = (long long) (mx * my * mz * sizeof(real));
    MPI_Datatype ty;
    int lsize[3]{mx, my, mz};
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_read_at(fp1, offset1, field[b].firstOrderMoment.data(), 6 + parameter.get_int("n_scalar"), ty, &status);
    offset1 += size * (6 + parameter.get_int("n_scalar"));
    cudaMemcpy(field[b].h_ptr->firstOrderMoment.data(), field[b].firstOrderMoment.data(),
               size * (6 + parameter.get_int("n_scalar")), cudaMemcpyHostToDevice);

    MPI_File_read_at(fp2, offset2, &mx, 1, MPI_INT32_T, &status);
    offset2 += 4;
    MPI_File_read_at(fp2, offset2, &my, 1, MPI_INT32_T, &status);
    offset2 += 4;
    MPI_File_read_at(fp2, offset2, &mz, 1, MPI_INT32_T, &status);
    offset2 += 4;
    if (mx != mesh[b].mx + 2 || my != mesh[b].my + 2 || mz != mesh[b].mz + 2) {
      printf(
          "The mesh size in the statistical data file 2nd-order_velocity_statistics.bin is not consistent with the current mesh size.\n");
      exit(1);
    }
    MPI_File_read_at(fp2, offset2, field[b].secondOrderMoment.data(), 6, ty, &status);
    offset2 += size * 6;
    cudaMemcpy(field[b].h_ptr->velocity2ndMoment.data(), field[b].secondOrderMoment.data(),
               size * 6, cudaMemcpyHostToDevice);
  }

  // Next, let us care about the user-defined part. The data to be collected may vary from one simulation to another.
  // So we need to check the consistency of the user-defined statistical data.
  for (int l = 0; l < UserDefineStat::n_collect; ++l) {
    std::string file_name = "output/stat/" + ud_collect_name[l] + ".bin";
    if (std::filesystem::exists(file_name)) {
      MPI_File fp_ud;
      MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_ud);
      MPI_File_read_at(fp_ud, 0, &counter_ud[l], 1, MPI_INT32_T, &status);
      MPI_Offset offset_ud = offset_unit[2];
      for (int b = 0; b < mesh.n_block; ++b) {
        int mx, my, mz;
        MPI_File_read_at(fp_ud, offset_ud, &mx, 1, MPI_INT32_T, &status);
        offset_ud += 4;
        MPI_File_read_at(fp_ud, offset_ud, &my, 1, MPI_INT32_T, &status);
        offset_ud += 4;
        MPI_File_read_at(fp_ud, offset_ud, &mz, 1, MPI_INT32_T, &status);
        offset_ud += 4;
        if (mx != mesh[b].mx + 2 || my != mesh[b].my + 2 || mz != mesh[b].mz + 2) {
          printf("The mesh size in the statistical data file %s is not consistent with the current mesh size.\n",
                 file_name.c_str());
          exit(1);
        }
        const auto size = (long long) (mx * my * mz * sizeof(real));
        MPI_Datatype ty;
        int lsize[3]{mx, my, mz};
        int start_idx[3]{0, 0, 0};
        MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
        MPI_Type_commit(&ty);
        MPI_File_read_at(fp_ud, offset_ud, field[b].userDefinedStatistics[l], 1, ty, &status);
        offset_ud += size;
        cudaMemcpy(field[b].h_ptr->userCollectForStat[l], field[b].userDefinedStatistics[l],
                   size, cudaMemcpyHostToDevice);
      }
    }
  }
}

void StatisticsCollector::plot_statistical_data(DParameter *param, bool perform_spanwise_average) {
  // First, compute the statistical data.
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }

  cudaMemcpy(counter_ud_device, counter_ud.data(), sizeof(int) * UserDefineStat::n_collect, cudaMemcpyHostToDevice);
  const int n_scalar{parameter.get_int("n_scalar")};
  for (int b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    compute_statistical_data<<<bpg, tpb>>>(field[b].d_ptr, param, counter, counter_ud_device);
    compute_user_defined_statistical_data<USER_DEFINE_STATISTICS><<<bpg, tpb>>>(field[b].d_ptr, param, counter,
                                                                                counter_ud_device);
    compute_UD_stat_data_2<SECOND_ORDER_UDSTAT><<<bpg, tpb>>>(field[b].d_ptr, param, counter,
                                                              counter_ud_device);
    if (perform_spanwise_average) {
      compute_statistical_data_spanwise_average<<<bpg, tpb>>>(field[b].d_ptr, param, counter, counter_ud_device);
      auto sz = mx * my * sizeof(real);
      cudaDeviceSynchronize();
      cudaMemcpy(field[b].mean_value.data(), field[b].h_ptr->mean_value_span_ave.data(), sz * (6 + n_scalar),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].reynolds_stress_tensor_and_rms.data(), field[b].h_ptr->reynolds_stress_tensor_span_ave.data(),
                 sz * 6, cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].user_defined_statistical_data.data(),
                 field[b].h_ptr->user_defined_statistical_data_span_ave.data(), sz * UserDefineStat::n_stat,
                 cudaMemcpyDeviceToHost);
    } else {
      auto sz = mx * my * mz * sizeof(real);
      cudaDeviceSynchronize();
      cudaMemcpy(field[b].mean_value.data(), field[b].h_ptr->mean_value.data(), sz * (6 + n_scalar),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].reynolds_stress_tensor_and_rms.data(), field[b].h_ptr->reynolds_stress_tensor.data(), sz * 6,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].user_defined_statistical_data.data(), field[b].h_ptr->user_defined_statistical_data.data(),
                 sz * UserDefineStat::n_stat, cudaMemcpyDeviceToHost);
    }
  }

  // Next, transfer the data to CPU

  // Next, output them.
  const std::filesystem::path out_dir("output");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // II. Data Section
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // First, modify the new min/max values of the variables
    MPI_Offset offset = offset_minmax_var[blk];

    double min_val{0}, max_val{1};
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my};
    auto mz{b.mz};
    if (perform_spanwise_average)
      mz = 1;
    for (int l = 0; l < 6; ++l) {
      min_val = v.mean_value(0, 0, 0, l);
      max_val = v.mean_value(0, 0, 0, l);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.mean_value(i, j, k, l));
            max_val = std::max(max_val, v.mean_value(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    for (int l = 0; l < 6; ++l) {
      min_val = v.reynolds_stress_tensor_and_rms(0, 0, 0, l);
      max_val = v.reynolds_stress_tensor_and_rms(0, 0, 0, l);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.reynolds_stress_tensor_and_rms(i, j, k, l));
            max_val = std::max(max_val, v.reynolds_stress_tensor_and_rms(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    offset = write_ud_stat_max_min(offset, v, fp, mz);
    for (int q = 0; q < n_scalar; ++q) {
      const int l = q + 6;
      min_val = v.mean_value(0, 0, 0, l);
      max_val = v.mean_value(0, 0, 0, l);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.mean_value(i, j, k, l));
            max_val = std::max(max_val, v.mean_value(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx, my, mz};
    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    offset = offset_var[blk];
    for (int l = 0; l < 6; ++l) {
      auto var = v.mean_value[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    for (int l = 0; l < 6; ++l) {
      auto var = v.reynolds_stress_tensor_and_rms[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    offset = write_ud_stat_data(offset, v, fp, ty, memsz);
    for (int q = 0; q < n_scalar; ++q) {
      auto var = v.mean_value[q + 6];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    MPI_Type_free(&ty);
  }
  MPI_File_close(&fp);
}

__global__ void collect_statistics(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= extent[0] + 1 || j >= extent[1] + 1 || k >= extent[2] + 1) return;

  const auto &bv = zone->bv;

  // The first order statistics of the flow field
  auto &firstOrderMoments = zone->firstOrderMoment;

  firstOrderMoments(i, j, k, 0) += bv(i, j, k, 0); // rho
  firstOrderMoments(i, j, k, 1) += bv(i, j, k, 0) * bv(i, j, k, 1); // rho*u
  firstOrderMoments(i, j, k, 2) += bv(i, j, k, 0) * bv(i, j, k, 2); // rho*v
  firstOrderMoments(i, j, k, 3) += bv(i, j, k, 0) * bv(i, j, k, 3); // rho*w
  firstOrderMoments(i, j, k, 4) += bv(i, j, k, 4); // p
  firstOrderMoments(i, j, k, 5) += bv(i, j, k, 0) * bv(i, j, k, 5); // rho*T

  const auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    firstOrderMoments(i, j, k, l + 6) += bv(i, j, k, 0) * sv(i, j, k, l);
  }

  // The second order statistics of the velocity field
  auto &velocity2ndOrder = zone->velocity2ndMoment;
  velocity2ndOrder(i, j, k, 0) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 1); // rho*u*u
  velocity2ndOrder(i, j, k, 1) += bv(i, j, k, 0) * bv(i, j, k, 2) * bv(i, j, k, 2); // rho*v*v
  velocity2ndOrder(i, j, k, 2) += bv(i, j, k, 0) * bv(i, j, k, 3) * bv(i, j, k, 3); // rho*w*w
  velocity2ndOrder(i, j, k, 3) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 2); // rho*u*v
  velocity2ndOrder(i, j, k, 4) += bv(i, j, k, 0) * bv(i, j, k, 1) * bv(i, j, k, 3); // rho*u*w
  velocity2ndOrder(i, j, k, 5) += bv(i, j, k, 0) * bv(i, j, k, 2) * bv(i, j, k, 3); // rho*v*w

  // Collect user-defined statistics
  collect_user_defined_statistics<USER_DEFINE_STATISTICS>(zone, param, i, j, k);
}

MPI_Offset write_ud_stat_max_min(MPI_Offset offset, const Field &field, MPI_File &fp, int mz) {
  MPI_Status status;
  const auto &b = field.block;
  for (int l = 0; l < UserDefineStat::n_stat; ++l) {
    real min_val = field.user_defined_statistical_data(0, 0, 0, l);
    real max_val = field.user_defined_statistical_data(0, 0, 0, l);
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < b.my; ++j) {
        for (int i = 0; i < b.mx; ++i) {
          min_val = std::min(min_val, field.user_defined_statistical_data(i, j, k, l));
          max_val = std::max(max_val, field.user_defined_statistical_data(i, j, k, l));
        }
      }
    }

    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
  }
  return offset;
}

MPI_Offset write_ud_stat_data(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, int64_t mem_sz) {
  MPI_Status status;
  for (int l = 0; l < UserDefineStat::n_stat; ++l) {
    MPI_File_write_at(fp, offset, field.user_defined_statistical_data[l], 1, ty, &status);
    offset += mem_sz;
  }

  return offset;
}

__global__ void compute_statistical_data(DZone *zone, DParameter *param, int counter, const int *counter_ud) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &mean = zone->mean_value;
  auto &firstOrderStat = zone->firstOrderMoment;

  mean(i, j, k, 0) = firstOrderStat(i, j, k, 0) / counter;
  const auto den = 1.0 / (mean(i, j, k, 0) * counter);
  mean(i, j, k, 1) = firstOrderStat(i, j, k, 1) * den;
  mean(i, j, k, 2) = firstOrderStat(i, j, k, 2) * den;
  mean(i, j, k, 3) = firstOrderStat(i, j, k, 3) * den;
  mean(i, j, k, 4) = firstOrderStat(i, j, k, 4) / counter;
  mean(i, j, k, 5) = firstOrderStat(i, j, k, 5) * den;
  for (int l = 0; l < param->n_scalar; ++l) {
    mean(i, j, k, l + 6) = firstOrderStat(i, j, k, l + 6) * den;
  }

  auto &rey_tensor = zone->reynolds_stress_tensor;
  auto &vel2ndMoment = zone->velocity2ndMoment;
  rey_tensor(i, j, k, 0) = vel2ndMoment(i, j, k, 0) * den - mean(i, j, k, 1) * mean(i, j, k, 1);
  rey_tensor(i, j, k, 1) = vel2ndMoment(i, j, k, 1) * den - mean(i, j, k, 2) * mean(i, j, k, 2);
  rey_tensor(i, j, k, 2) = vel2ndMoment(i, j, k, 2) * den - mean(i, j, k, 3) * mean(i, j, k, 3);
  rey_tensor(i, j, k, 3) = vel2ndMoment(i, j, k, 3) * den - mean(i, j, k, 1) * mean(i, j, k, 2);
  rey_tensor(i, j, k, 4) = vel2ndMoment(i, j, k, 4) * den - mean(i, j, k, 1) * mean(i, j, k, 3);
  rey_tensor(i, j, k, 5) = vel2ndMoment(i, j, k, 5) * den - mean(i, j, k, 2) * mean(i, j, k, 3);
}

__global__ void
compute_statistical_data_spanwise_average(DZone *zone, DParameter *param, int counter, const int *counter_ud) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  if (i >= extent[0] || j >= extent[1]) return;

  auto &mean = zone->mean_value;
  auto &mean_span_ave = zone->mean_value_span_ave;

  real bv_add[6];
  memset(bv_add, 0, sizeof(real) * 6);
  for (int k = 0; k < extent[2]; ++k) {
    bv_add[0] += mean(i, j, k, 0);
    bv_add[1] += mean(i, j, k, 1);
    bv_add[2] += mean(i, j, k, 2);
    bv_add[3] += mean(i, j, k, 3);
    bv_add[4] += mean(i, j, k, 4);
    bv_add[5] += mean(i, j, k, 5);
  }
  const real oneDivNz{1.0 / extent[2]};
  mean_span_ave(i, j, 0, 0) = bv_add[0] * oneDivNz;
  mean_span_ave(i, j, 0, 1) = bv_add[1] * oneDivNz;
  mean_span_ave(i, j, 0, 2) = bv_add[2] * oneDivNz;
  mean_span_ave(i, j, 0, 3) = bv_add[3] * oneDivNz;
  mean_span_ave(i, j, 0, 4) = bv_add[4] * oneDivNz;
  mean_span_ave(i, j, 0, 5) = bv_add[5] * oneDivNz;
  for (int l = 0; l < param->n_scalar; ++l) {
    real addUp{0};
    for (int k = 0; k < extent[2]; ++k) {
      addUp += mean(i, j, k, l + 6);
    }
    mean_span_ave(i, j, 0, l + 6) = addUp * oneDivNz;
  }

  auto &rey_tensor = zone->reynolds_stress_tensor;
  auto &vel2ndMoment = zone->velocity2ndMoment;
  real rey_tensor_add[6];
  memset(rey_tensor_add, 0, sizeof(real) * 6);
  int count_useful[3]{0, 0, 0};
  for (int k = 0; k < extent[2]; ++k) {
    auto temp = rey_tensor(i, j, k, 0);
    if (temp > 0) {
      rey_tensor_add[0] += temp;
      count_useful[0]++;
    }
    temp = rey_tensor(i, j, k, 1);
    if (temp > 0) {
      rey_tensor_add[1] += temp;
      count_useful[1]++;
    }
    temp = rey_tensor(i, j, k, 2);
    if (temp > 0) {
      rey_tensor_add[2] += temp;
      count_useful[2]++;
    }
    rey_tensor_add[3] += rey_tensor(i, j, k, 3);
    rey_tensor_add[4] += rey_tensor(i, j, k, 4);
    rey_tensor_add[5] += rey_tensor(i, j, k, 5);
  }
  rey_tensor(i, j, 0, 0) = count_useful[0] > 0 ? rey_tensor_add[0] / count_useful[0] : 0;
  rey_tensor(i, j, 0, 1) = count_useful[1] > 0 ? rey_tensor_add[1] / count_useful[1] : 0;
  rey_tensor(i, j, 0, 2) = count_useful[2] > 0 ? rey_tensor_add[2] / count_useful[2] : 0;
  rey_tensor(i, j, 0, 3) = rey_tensor_add[3] * oneDivNz;
  rey_tensor(i, j, 0, 4) = rey_tensor_add[4] * oneDivNz;
  rey_tensor(i, j, 0, 5) = rey_tensor_add[5] * oneDivNz;

  compute_user_defined_statistical_data_with_spanwise_average<USER_DEFINE_STATISTICS>(zone, param, counter_ud, i, j,
                                                                                      extent[2], counter);
}

} // cfd