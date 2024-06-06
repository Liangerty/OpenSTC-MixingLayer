#include "Monitor.cuh"
#include "gxl_lib/MyString.h"
#include <filesystem>
#include "Parallel.h"
#include "Field.h"
#include "Mesh.h"
#include "gxl_lib/MyAlgorithm.h"

namespace cfd {
Monitor::Monitor(const Parameter &parameter, const Species &species, const Mesh &mesh_) : if_monitor{
    parameter.get_int("if_monitor")}, output_file{parameter.get_int("output_file")}, n_block{
    parameter.get_int("n_block")}, n_point(n_block, 0), mesh(mesh_) {
  if (!if_monitor) {
    return;
  }

  h_ptr = new DeviceMonitorData;

  // Set up the labels to monitor
  auto var_name_found{setup_labels_to_monitor(parameter, species)};
  const auto myid{parameter.get_int("myid")};
  if (myid == 0) {
    printf("The following variables will be monitored:\n");
    for (const auto &name: var_name_found) {
      printf("%s\t", name.c_str());
    }
    printf("\n");
  }

  // Read the points to be monitored
  auto monitor_file_name{parameter.get_string("monitor_file")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!std::filesystem::exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  int counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    int pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    int i, j, k, b;
    line_stream >> b >> i >> j >> k;
    is_h.push_back(i);
    js_h.push_back(j);
    ks_h.push_back(k);
    bs_h.push_back(b);
    printf("Process %d monitors block %d, point (%d, %d, %d).\n", myid, b, i, j, k);
    ++n_point[b];
    ++counter;
  }
  // copy the indices to GPU
  cudaMalloc(&h_ptr->bs_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->is_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->js_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->ks_d, sizeof(int) * counter);
  cudaMemcpy(h_ptr->bs_d, bs_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->is_d, is_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->js_d, js_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->ks_d, ks_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  n_point_total = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point_total);
  disp.resize(parameter.get_int("n_block"), 0);
  for (int b = 1; b < n_block; ++b) {
    disp[b] = disp[b - 1] + n_point[b - 1];
  }
  cudaMalloc(&h_ptr->disp, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->disp, disp.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);
  cudaMalloc(&h_ptr->n_point, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->n_point, n_point.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point_total, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point_total);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point_total, nullptr);
  for (int l = 0; l < n_point_total; ++l) {
    std::string file_name{
        "/monitor_" + std::to_string(myid) + '_' + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' +
        std::to_string(js_h[l]) + '_' +
        std::to_string(ks_h[l]) + ".dat"};
    std::filesystem::path whole_name_path{out_dir.string() + file_name};
    if (!exists(whole_name_path)) {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
      fprintf(files[l], "variables=step,");
      for (const auto &name: var_name_found) {
        fprintf(files[l], "%s,", name.c_str());
      }
      fprintf(files[l], "time\n");
    } else {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
    }
  }

  // For slices.
  const int n_proc{parameter.get_int("n_proc")};
  auto xSlice = parameter.get_real_array("xSlice");
  // Here, we assume the points with the same i index have the same x coordinate.
  if (!xSlice.empty()) {
    const std::filesystem::path out_dir_slice("output/slice");
    if (!exists(out_dir_slice)) {
      create_directories(out_dir_slice);
    }
    const real gridScale{parameter.get_real("gridScale")};
    auto distance_smallest = new real[n_proc];
    for (auto xThis: xSlice) {
      int iThis{0}, bThis{0};
      real dist{1e+6};
      real xx = xThis * parameter.get_real("gridScale");
      for (int b = 0; b < n_block; ++b) {
        auto &x = mesh[b].x;
        for (int i = 0; i < mesh[b].mx; ++i) {
          if (abs(xx - x(i, 0, 0)) < dist) {
            dist = abs(xx - x(i, 0, 0));
            iThis = i;
            bThis = b;
          }
        }
      }
      // We have found the nearest block and i index to the slice in current process.
      // Next, we need to communicate among all processes to find the nearest.
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allgather(&dist, 1, MPI_DOUBLE, distance_smallest, 1, MPI_DOUBLE, MPI_COMM_WORLD);
      bool this_smallest{true};
      for (int p = 0; p < n_proc; ++p) {
        if (p == myid)
          continue;
        if (distance_smallest[p] < dist) {
          this_smallest = false;
          break;
        }
      }
      if (this_smallest) {
        iSlice.push_back(iThis);
        iSliceInBlock.push_back(bThis);
        ++n_iSlice;
        printf("Process %d monitors block %d, iSlice %d with x = %f.\n", myid, bThis, iThis, xx);
      }
    }
    delete[] distance_smallest;

    // If the slice is found, we need to monitor the slice.
    if (n_iSlice > 0) {
      auto core_range{parameter.get_real_range("coreRegion")};
      Range<real> core_region{core_range.xs * gridScale, core_range.xe * gridScale,
                              core_range.ys * gridScale, core_range.ye * gridScale,
                              core_range.zs * gridScale, core_range.ze * gridScale};
      int MY{1}, MZ{1};
      for (int s = 0; s < n_iSlice; ++s) {
        // We need to find the region excluding the buffer layers.
        int i = iSlice[s];
        const auto &b = mesh[iSliceInBlock[s]];
        MY = std::max(b.my, MY);
        MZ = std::max(b.mz, MZ);

        int js = 0, je = b.my - 1, ks = 0, ke = b.mz - 1;
        int jd = 1, kd = 1;
        if (b.y(i, 0, 0) > b.y(i, 1, 0))
          jd = -1;
        if (b.z(i, 0, 0) > b.z(i, 0, 1))
          kd = -1;
        for (int j = 0; j < b.my; ++j) {
          for (int k = 0; k < b.mz; ++k) {
            if (b.y(i, j, k) < core_region.ys) {
              js = j + jd;
            }
            if (b.y(i, j, k) > core_region.ye) {
              je = j - jd;
            }
            if (b.z(i, j, k) < core_region.zs) {
              ks = k + kd;
            }
            if (b.z(i, j, k) > core_region.ze) {
              ke = k - kd;
            }
          }
        }
        if (mesh.dimension == 2) {
          ks = 0;
          ke = 0;
        }
        iSlice_js.push_back(js);
        iSlice_je.push_back(je);
        iSlice_ks.push_back(ks);
        iSlice_ke.push_back(ke);
        // Here, we should first output the slices' coordinates.
        MPI_File fp;
        char file_name[1024];
        sprintf(file_name, "%s/xSlice_%f_coordinates.bin", out_dir_slice.string().c_str(),
                b.x(iSlice[s], 0, 0) / parameter.get_real("gridScale"));
        MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
        MPI_Status status;
        MPI_Offset offset{0};
        const auto ny{je - js + 1}, nz{ke - ks + 1};
        int nnv = parameter.get_int("n_var") + 1;
        MPI_File_write_at(fp, offset, &nnv, 1, MPI_INT, &status);
        offset += 4;
        MPI_File_write_at(fp, offset, &ny, 1, MPI_INT, &status);
        offset += 4;
        MPI_File_write_at(fp, offset, &nz, 1, MPI_INT, &status);
        offset += 4;

        MPI_Datatype ty;
        int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
        int small_size[3]{1, ny, nz};
        const auto mem_sz = ny * nz * 8;
        int start_idx[3]{b.ngg + iSlice[s], b.ngg + iSlice_js[s], b.ngg + iSlice_ks[s]};
        MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
        MPI_Type_commit(&ty);
        MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
        offset += mem_sz;
        MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
        // offset += mem_sz;
        MPI_File_close(&fp);
      }
    }
  }
//  auto zSlice = parameter.get_real_array("zSlice");
  // Here, we assume the points with the same k index have the same z coordinate.
  // Besides, we also assume that the blocking is only in x direction.
  // Thus, the zSlice would be contained in multiple processes.
//  if ((!zSlice.empty()) && mesh.dimension == 3) {
//    const std::filesystem::path out_dir_slice("output/slice");
//    if (!exists(out_dir_slice)) {
//      create_directories(out_dir_slice);
//    }
//    const real gridScale{parameter.get_real("gridScale")};
//    auto distance_smallest = new real[n_proc];
//    for (auto zThis: zSlice) {
//      int kThis{0}, bThis{0};
//      real dist{1e+6};
//      real zz = zThis * parameter.get_real("gridScale");
//      for (int b = 0; b < n_block; ++b) {
//        auto &z = mesh[b].z;
//        for (int k = 0; k < mesh[b].mz; ++k) {
//          if (abs(zz - z(0, 0, k)) < dist) {
//            dist = abs(zz - z(0, 0, k));
//            kThis = k;
//            bThis = b;
//          }
//        }
//      }
//      // We have found the nearest block and i index to the slice in current process.
//      // Next, we need to communicate among all processes to find the nearest.
//      MPI_Barrier(MPI_COMM_WORLD);
//      MPI_Allgather(&dist, 1, MPI_DOUBLE, distance_smallest, 1, MPI_DOUBLE, MPI_COMM_WORLD);
//      bool this_smallest{true};
//      for (int p = 0; p < n_proc; ++p) {
//        if (p == myid)
//          continue;
//        if (distance_smallest[p] < dist) {
//          this_smallest = false;
//          break;
//        }
//      }
//      if (this_smallest) {
//        kSlice.push_back(kThis);
//        kSliceInBlock.push_back(bThis);
//        ++n_kSlice;
//        printf("Process %d monitors block %d, kSlice %d with x = %f.\n", myid, bThis, kThis, zz);
//      }
//    }
//    delete[] distance_smallest;
//
//    // If the slice is found, we need to monitor the slice.
//    if (n_iSlice > 0) {
//      auto core_range{parameter.get_real_range("coreRegion")};
//      Range<real> core_region{core_range.xs * gridScale, core_range.xe * gridScale,
//                              core_range.ys * gridScale, core_range.ye * gridScale,
//                              core_range.zs * gridScale, core_range.ze * gridScale};
//      int MX{1}, MY{1};
//      for (int s = 0; s < n_iSlice; ++s) {
//        // We need to find the region excluding the buffer layers.
//        int k = kSlice[s];
//        const auto &b = mesh[iSliceInBlock[s]];
//        MX = std::max(b.mx, MX);
//        MY = std::max(b.mz, MY);
//
//        int is = 0, ie = b.mx - 1, js = 0, je = b.my - 1;
//        // Here, we assume the x increases with the i index.
//        for (int i = 0; i < b.mx; ++i) {
//          if (b.x(i, 0, k) > core_region.xs) {
//            is = i;
//            break;
//          }
//        }
//        for (int i = 0; i < b.mx; ++i) {
//          if (b.x(i, 0, k) > core_region.xe) {
//            ie = i - 1;
//            break;
//          }
//        }
//        // We also assume the y increases with the j index.
//        for (int j = 0; j < b.my; ++j) {
//          if (b.y(0, j, k) > core_region.ys) {
//            js = j;
//            break;
//          }
//        }
//        for (int j = 0; j < b.my; ++j) {
//          if (b.y(0, j, k) > core_region.ye) {
//            je = j - 1;
//            break;
//          }
//        }
//        kSlice_is.push_back(is);
//        kSlice_ie.push_back(ie);
//        kSlice_js.push_back(js);
//        kSlice_je.push_back(je);
//        // Here, we should first output the slices' coordinates.
//        MPI_File fp;
//        char file_name[1024];
//        sprintf(file_name, "%s/zSlice_%f_coordinates.bin", out_dir_slice.string().c_str(),
//                b.z(0, 0, kSlice[s]) / parameter.get_real("gridScale"));
//        MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
//        MPI_Status status;
//        MPI_Offset offset{0};
//        const auto nx{ie - is + 1}, ny{je - js + 1};
//        int nnv = parameter.get_int("n_var") + 1;
//        MPI_File_write_at(fp, offset, &nnv, 1, MPI_INT, &status);
//        offset += 4;
//        MPI_File_write_at(fp, offset, &nx, 1, MPI_INT, &status);
//        offset += 4;
//        MPI_File_write_at(fp, offset, &ny, 1, MPI_INT, &status);
//        offset += 4;
//
//        MPI_Datatype ty;
//        int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
//        int small_size[3]{nx, ny, 1};
//        const auto mem_sz = nx * ny * 8;
//        int start_idx[3]{b.ngg + kSlice_is[s], b.ngg + kSlice_js[s], b.ngg + kSlice[s]};
//        MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
//        MPI_Type_commit(&ty);
//        MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
//        offset += mem_sz;
//        MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
//        // offset += mem_sz;
//        MPI_File_close(&fp);
//      }
//    }
//  }
}

std::vector<std::string> Monitor::setup_labels_to_monitor(const Parameter &parameter, const Species &species) {
  auto n_spec{species.n_spec};
  auto &spec_list{species.spec_list};

  auto var_name{parameter.get_string_array("monitor_var")};

  std::vector<int> bv_idx, sv_idx;
  auto n_found{0};
  std::vector<std::string> var_name_found;
  for (auto name: var_name) {
    name = gxl::to_upper(name);
    if (name == "DENSITY" || name == "RHO") {
      bv_idx.push_back(0);
      var_name_found.emplace_back("Density");
      ++n_found;
    } else if (name == "U") {
      bv_idx.push_back(1);
      var_name_found.emplace_back("U");
      ++n_found;
    } else if (name == "V") {
      bv_idx.push_back(2);
      var_name_found.emplace_back("V");
      ++n_found;
    } else if (name == "W") {
      bv_idx.push_back(3);
      var_name_found.emplace_back("W");
      ++n_found;
    } else if (name == "PRESSURE" || name == "P") {
      bv_idx.push_back(4);
      var_name_found.emplace_back("Pressure");
      ++n_found;
    } else if (name == "TEMPERATURE" || name == "T") {
      bv_idx.push_back(5);
      var_name_found.emplace_back("Temperature");
      ++n_found;
    } else if (n_spec > 0) {
      auto it = spec_list.find(name);
      if (it != spec_list.end()) {
        sv_idx.push_back(it->second);
        var_name_found.emplace_back(name);
        ++n_found;
      }
    } else if (name == "TKE") {
      sv_idx.push_back(n_spec);
      var_name_found.emplace_back("TKE");
      ++n_found;
    } else if (name == "OMEGA") {
      sv_idx.push_back(n_spec + 1);
      var_name_found.emplace_back("Omega");
      ++n_found;
    } else if (name == "MIXTUREFRACTION" || name == "Z") {
      sv_idx.push_back(n_spec + 2);
      var_name_found.emplace_back("Mixture fraction");
      ++n_found;
    } else if (name == "MIXTUREFRACTIONVARIANCE") {
      sv_idx.push_back(n_spec + 3);
      var_name_found.emplace_back("Mixture fraction variance");
      ++n_found;
    } else {
      if (parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  n_bv = (int) (bv_idx.size());
  n_sv = (int) (sv_idx.size());
  // The +1 is for physical time
  n_var = n_bv + n_sv + 1;
  h_ptr->n_bv = n_bv;
  h_ptr->n_sv = n_sv;
  h_ptr->n_var = n_var;
  cudaMalloc(&h_ptr->bv_label, sizeof(int) * n_bv);
  cudaMalloc(&h_ptr->sv_label, sizeof(int) * n_sv);
  cudaMemcpy(h_ptr->bv_label, bv_idx.data(), sizeof(int) * n_bv, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->sv_label, sv_idx.data(), sizeof(int) * n_sv, cudaMemcpyHostToDevice);

  return var_name_found;
}

Monitor::~Monitor() {
  for (auto fp: files) {
    fclose(fp);
  }
}

void Monitor::monitor_point(int step, real physical_time, std::vector<cfd::Field> &field) {
  if (counter_step == 0)
    step_start = step;

  for (int b = 0; b < n_block; ++b) {
    if (n_point[b] > 0) {
      const auto tpb{128};
      const auto bpg{(n_point[b] - 1) / tpb + 1};
      record_monitor_data<<<bpg, tpb>>>(field[b].d_ptr, d_ptr, b, counter_step % output_file, physical_time);
    }
  }
  ++counter_step;
}

void Monitor::output_data() {
  cudaMemcpy(mon_var_h.data(), h_ptr->data.data(), sizeof(real) * n_var * output_file * n_point_total,
             cudaMemcpyDeviceToHost);

  for (int p = 0; p < n_point_total; ++p) {
    for (int l = 0; l < counter_step; ++l) {
      fprintf(files[p], "%d\t", step_start + l);
      for (int k = 0; k < n_var; ++k) {
        fprintf(files[p], "%e\t", mon_var_h(k, l, p));
      }
      fprintf(files[p], "\n");
    }
  }
  counter_step = 0;
}

void Monitor::output_slices(const Parameter &parameter, std::vector<cfd::Field> &field, int step, real t) {
  if (n_iSlice <= 0) {
    return;
  }
  std::vector<int> blk_read{};
  const std::filesystem::path out_dir("output/slice");
  for (int s = 0; s < n_iSlice; ++s) {
    auto blk = iSliceInBlock[s];
    const auto &b = mesh[iSliceInBlock[s]];
    const int64_t size = (b.mx + 2 * b.ngg) * (b.my + 2 * b.ngg) * (b.mz + 2 * b.ngg);
    auto &f = field[blk];
    if (!gxl::exists(blk_read, blk)) {
      cudaMemcpy(f.bv.data(), f.h_ptr->bv.data(), sizeof(real) * size * 6, cudaMemcpyDeviceToHost);
      cudaMemcpy(f.sv.data(), f.h_ptr->sv.data(), sizeof(real) * size * parameter.get_int("n_scalar"),
                 cudaMemcpyDeviceToHost);
      blk_read.push_back(blk);
    }

    MPI_Datatype ty;
    int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    auto ny{iSlice_je[s] - iSlice_js[s] + 1}, nz{iSlice_ke[s] - iSlice_ks[s] + 1};
    int small_size[3]{1, ny, nz};
    const auto mem_sz = ny * nz * 8;
    int start_idx[3]{b.ngg + iSlice[s], b.ngg + iSlice_js[s], b.ngg + iSlice_ks[s]};
    MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    MPI_File fp;
    char file_name[1024];
    sprintf(file_name, "%s/xSlice_%f_%d_t=%e.bin", out_dir.string().c_str(),
            b.x(iSlice[s], 0, 0) / parameter.get_real("gridScale"), slice_counter, t);
    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    MPI_Status status;
    MPI_Offset offset{0};
    MPI_File_write_at(fp, offset, &t, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &parameter.get_int("n_var") + 1, 1, MPI_INT, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &ny, 1, MPI_INT, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &nz, 1, MPI_INT, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, f.bv[0], 1, ty, &status);
    offset += mem_sz;
    MPI_File_write_at(fp, offset, f.bv[1], 1, ty, &status);
    offset += mem_sz;
    MPI_File_write_at(fp, offset, f.bv[2], 1, ty, &status);
    offset += mem_sz;
    MPI_File_write_at(fp, offset, f.bv[3], 1, ty, &status);
    offset += mem_sz;
    MPI_File_write_at(fp, offset, f.bv[4], 1, ty, &status);
    offset += mem_sz;
    MPI_File_write_at(fp, offset, f.bv[5], 1, ty, &status);
    offset += mem_sz;
    for (int l = 0; l < parameter.get_int("n_scalar"); ++l) {
      MPI_File_write_at(fp, offset, f.sv[l], 1, ty, &status);
      offset += mem_sz;
    }
    MPI_File_close(&fp);
  }

//  if (n_kSlice <= 0) {
//    return;
//  }
  ++slice_counter;
}

__global__ void
record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos, real physical_time) {
  auto idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  if (idx >= monitor_info->n_point[blk_id])
    return;
  auto idx_tot = monitor_info->disp[blk_id] + idx;
  auto i = monitor_info->is_d[idx_tot];
  auto j = monitor_info->js_d[idx_tot];
  auto k = monitor_info->ks_d[idx_tot];

  auto &data = monitor_info->data;
  const auto bv_label = monitor_info->bv_label;
  const auto sv_label = monitor_info->sv_label;
  const auto n_bv{monitor_info->n_bv};
  int var_counter{0};
  for (int l = 0; l < n_bv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->bv(i, j, k, bv_label[l]);
    ++var_counter;
  }
  for (int l = 0; l < monitor_info->n_sv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->sv(i, j, k, sv_label[l]);
    ++var_counter;
  }
  data(var_counter, counter_pos, idx_tot) = physical_time;
}
} // cfd