#pragma once

#include "Parameter.h"
#include "ChemData.h"
#include "gxl_lib/Array.cuh"

namespace cfd {

struct Field;
class Mesh;

struct DeviceMonitorData {
  int n_bv{0}, n_sv{0}, n_var{0};
  int *bv_label{nullptr};
  int *sv_label{nullptr};
  int *bs_d{nullptr}, *is_d{nullptr}, *js_d{nullptr}, *ks_d{nullptr};
  int *disp{nullptr}, *n_point{nullptr};
  ggxl::Array3D<real> data;
};

class Monitor {
public:
  explicit Monitor(const Parameter &parameter, const Species &species, const Mesh& mesh_);

  void monitor_point(int step, real physical_time, std::vector<cfd::Field> &field);

  void output_data();

  void output_slices(const Parameter &parameter, std::vector<cfd::Field> &field, int step, real t);

  ~Monitor();

private:
  int if_monitor{0};
  int output_file{0};
  int step_start{0};
  int counter_step{0};
  int n_block{0};
  int n_bv{0}, n_sv{0}, n_var{0};
  std::vector<int> bs_h, is_h, js_h, ks_h;
  int n_point_total{0};
  std::vector<int> n_point;
  std::vector<int> disp;
  ggxl::Array3DHost<real> mon_var_h;
  DeviceMonitorData *h_ptr, *d_ptr{nullptr};
  std::vector<FILE *> files;

  const Mesh& mesh;
  int slice_counter{0};
  int n_iSlice{0};
  std::vector<int> iSlice, iSliceInBlock;
  std::vector<int> iSlice_js, iSlice_je, iSlice_ks, iSlice_ke;
//  int n_kSlice{0};
//  std::vector<int> kSlice, kSliceInBlock;
//  std::vector<int> kSlice_is, kSlice_ie, kSlice_js, kSlice_je;

private:
  // Utility functions
  std::vector<std::string> setup_labels_to_monitor(const Parameter &parameter, const Species &species);
};

struct DZone;
__global__ void record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos,
                                    real physical_time);

} // cfd
