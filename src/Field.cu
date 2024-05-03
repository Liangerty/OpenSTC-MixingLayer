#include "Field.h"
#include "BoundCond.h"
#include "UDIO.h"
#include "UDStat.h"
#include <fstream>
#include "Parallel.h"
#include "gxl_lib/MyString.h"
#include "Transport.cuh"
#include "DParameter.cuh"

cfd::Field::Field(Parameter &parameter, const Block &block_in) : block(block_in) {
  const integer mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  // Let us re-compute the number of variables to be solved here.
  n_var = 5;
  // The variable "n_scalar_transported" is the number of scalar variables to be transported.
  integer n_scalar_transported{0};
  integer n_other_var{1}; // Default, mach number
  // The variable "n_scalar" is the number of scalar variables in total, including those not transported.
  // This is different from the variable "n_scalar_transported" only when the flamelet model is used.
  integer n_scalar{0};
  // turbulent variable in sv array is always located after mass fractions, thus its label is always n_spec;
  // however, for conservative array, it may depend on whether the employed method is flamelet or finite rate.
  // This label, "i_turb_cv", is introduced to label the index of the first turbulent variable in the conservative variable array.
  integer i_turb_cv{5}, i_fl_cv{0};

  if (parameter.get_int("species") == 1) {
    n_scalar += parameter.get_int("n_spec");
    if (parameter.get_int("reaction") != 2) {
      // Mixture / Finite rate chemistry
      n_scalar_transported += parameter.get_int("n_spec");
      i_turb_cv += parameter.get_int("n_spec");
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
      i_fl_cv = 5 + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  } else if (parameter.get_int("species") == 2) {
    n_scalar += parameter.get_int("n_spec");
    if (parameter.get_int("reaction") != 2) {
      // Mixture with mixture fraction and variance solved.
      n_scalar_transported += parameter.get_int("n_spec") + 2;
      n_scalar += 2;
      i_turb_cv += parameter.get_int("n_spec");
      i_fl_cv = i_turb_cv + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
      i_fl_cv = 5 + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  }
  if (parameter.get_bool("turbulence")) {
    // turbulence simulation
    if (auto turb_method = parameter.get_int("turbulence_method");turb_method == 1) {
      // RANS
      n_scalar_transported += parameter.get_int("n_turb");
      n_scalar += parameter.get_int("n_turb");
      ++n_other_var; // mut
    } else if (turb_method == 2) {
      // DES type
      n_scalar_transported += parameter.get_int("n_turb");
      n_scalar += parameter.get_int("n_turb");
      ++n_other_var; // mut
    }
  }
  n_var += n_scalar_transported;
  parameter.update_parameter("n_var", n_var);
  parameter.update_parameter("n_scalar", n_scalar);
  parameter.update_parameter("n_scalar_transported", n_scalar_transported);
  parameter.update_parameter("i_turb_cv", i_turb_cv);
  parameter.update_parameter("i_fl", parameter.get_int("n_turb") + parameter.get_int("n_spec"));
  parameter.update_parameter("i_fl_cv", i_fl_cv);

  // Acquire memory for variable arrays
  bv.resize(mx, my, mz, 6, ngg);
  sv.resize(mx, my, mz, n_scalar, ngg);
  ov.resize(mx, my, mz, n_other_var, ngg);
  udv.resize(mx, my, mz, UserDefineIO::n_dynamic_auxiliary, ngg);

  if (parameter.get_bool("if_collect_statistics")) {
    // If we need to collect the statistics, we need to allocate memory for the data.
    firstOrderMoment.resize(mx, my, mz, 6 + n_scalar, 0);
    secondOrderMoment.resize(mx, my, mz, 9, 0);
    userDefinedStatistics.resize(mx, my, mz, UserDefineStat::n_collect, 0);

    if (parameter.get_bool("perform_spanwise_average")) {
      mean_value.resize(mx, my, 1, 6 + n_scalar, 0);
      reynolds_stress_tensor_and_rms.resize(mx, my, 1, 9, 0);
      user_defined_statistical_data.resize(mx, my, 1, UserDefineStat::n_stat, 0);
    } else {
      mean_value.resize(mx, my, mz, 6 + n_scalar, 0);
      reynolds_stress_tensor_and_rms.resize(mx, my, mz, 9, 0);
      user_defined_statistical_data.resize(mx, my, mz, UserDefineStat::n_stat, 0);
    }
  }
}

std::vector<integer>
identify_variable_labels(const cfd::Parameter &parameter, std::vector<std::string> &var_name,
                         const cfd::Species &species,
                         bool &has_pressure, bool &has_temperature, bool &has_tke) {
  std::vector<integer> labels;
  const integer n_spec = species.n_spec;
  const integer n_turb = parameter.get_int("n_turb");
  for (auto &name: var_name) {
    integer l = 999;
    // The first three names are x, y and z, they are assigned value 0 and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "X") {
      l = 0;
    } else if (n == "Y") {
      l = 1;
    } else if (n == "Z") {
      l = 2;
    } else if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0 + 3;
    } else if (n == "U") {
      l = 1 + 3;
    } else if (n == "V") {
      l = 2 + 3;
    } else if (n == "W") {
      l = 3 + 3;
    } else if (n == "P" || n == "PRESSURE") {
      l = 4 + 3;
      has_pressure = true;
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5 + 3;
      has_temperature = true;
    } else {
      if (n_spec > 0) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (auto [spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label + 3;
            break;
          }
        }
        if (n == "MIXTUREFRACTION") {
          // Mixture fraction
          l = 6 + n_spec + n_turb + 3;
        } else if (n == "MIXTUREFRACTIONVARIANCE") {
          // Mixture fraction variance
          l = 6 + n_spec + n_turb + 1 + 3;
        }
      }
      if (n_turb > 0) {
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K" || n == "TKE") { // turbulent kinetic energy
          if (n_turb == 2) {
            l = 6 + n_spec + 3;
            has_tke = true;
          }
        } else if (n == "OMEGA") { // specific dissipation rate
          if (n_turb == 2) {
            l = 6 + n_spec + 1 + 3;
          }
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          if (n_turb == 1) {
            l = 6 + n_spec + 3;
          }
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

std::array<int, 3> read_dat_profile_for_init(gxl::VectorField3D<real> &profile, const std::string &file,
                                             const cfd::Parameter &parameter, const cfd::Species &species,
                                             integer profile_idx) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    cfd::MpiParallel::exit();
  }
//  const integer direction = boundary.face;

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
  bool has_pressure{false}, has_temperature{false}, has_tke{false};
  auto label_order = identify_variable_labels(parameter, var_name, species, has_pressure, has_temperature, has_tke);
  if ((!has_temperature) && (!has_pressure)) {
    printf("The temperature or pressure is not given in the profile, please provide at least one of them!\n");
    cfd::MpiParallel::exit();
  }
  real turb_viscosity_ratio{0}, turb_intensity{0};
  if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
    auto &info = parameter.get_struct(parameter.get_string_array("profile_related_bc_names")[profile_idx]);
    if (info.find("turb_viscosity_ratio") == info.end() || info.find("turbulence_intensity") == info.end()) {
      printf(
          "The turbulence intensity or turbulent viscosity ratio is not given in the profile, please provide both of them!\n");
      cfd::MpiParallel::exit();
    }
    turb_viscosity_ratio = std::get<real>(info.at("turb_viscosity_ratio"));
    turb_intensity = std::get<real>(info.at("turbulence_intensity"));
  }
  integer mx, my, mz;
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

  std::array<int, 3> extent{mx, my, mz};
  // Then we read the variables.
  auto nv_read = (integer) var_name.size();
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

  const integer n_var = parameter.get_int("n_var");
  profile.resize(extent[0], extent[1], extent[2], n_var + 4, 0);
  for (int k = 0; k < extent[2]; ++k) {
    for (int j = 0; j < extent[1]; ++j) {
      for (int i = 0; i < extent[0]; ++i) {

        for (int l = 0; l < nv_read; ++l) {
          if (label_order[l] < n_var + 1 + 3) {
            profile(i, j, k, label_order[l]) = profile_read(i, j, k, l);
          }
        }

        // If T or p is not given, compute it.
        if (!has_temperature) {
          real mw{cfd::mw_air};
          if (species.n_spec > 0) {
            mw = 0;
            for (int l = 0; l < species.n_spec; ++l) mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
            mw = 1 / mw;
          }
          profile(i, j, k, 5 + 3) = profile(i, j, k, 4 + 3) * mw / (cfd::R_u * profile(i, j, k, 0 + 3));
        }
        if (!has_pressure) {
          real mw{cfd::mw_air};
          if (species.n_spec > 0) {
            mw = 0;
            for (int l = 0; l < species.n_spec; ++l) mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
            mw = 1 / mw;
          }
          profile(i, j, k, 4 + 3) = profile(i, j, k, 5 + 3) * cfd::R_u * profile(i, j, k, 0 + 3) / mw;
        }
        if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !(has_tke)) {
          // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
          real mu{};
          if (species.n_spec > 0) {
            real mw = 0;
            std::vector<real> Y;
            for (int l = 0; l < species.n_spec; ++l) {
              mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
              Y.push_back(profile(i, j, k, 6 + l + 3));
            }
            mw = 1 / mw;
            mu = cfd::compute_viscosity(profile(i, j, k, 5 + 3), mw, Y.data(), species);
          } else {
            mu = cfd::Sutherland(profile(i, j, k, 5 + 3));
          }
          real mut = mu * turb_viscosity_ratio;
          const real vel2 = profile(i, j, k, 1 + 3) * profile(i, j, k, 1 + 3) +
                            profile(i, j, k, 2 + 3) * profile(i, j, k, 2 + 3) +
                            profile(i, j, k, 3 + 3) * profile(i, j, k, 3 + 3);
          profile(i, j, k, 6 + species.n_spec + 3) = 1.5 * vel2 * turb_intensity * turb_intensity;
          profile(i, j, k, 6 + species.n_spec + 1 + 3) =
              profile(i, j, k, 0 + 3) * profile(i, j, k, 6 + species.n_spec + 3) / mut;
        }

      }
    }
  }
  return extent;
}

std::array<int, 3> read_profile_to_init(gxl::VectorField3D<real> &profile, integer profile_idx,
                                        const cfd::Parameter &parameter, const cfd::Species &species) {
  const std::string &file = parameter.get_string_array("profile_file_names")[profile_idx];
  auto dot = file.find_last_of('.');
  auto suffix = file.substr(dot + 1, file.size());
  if (suffix == "dat") {
    auto extent = read_dat_profile_for_init(profile, file, parameter, species, profile_idx);
    return extent;
  } else if (suffix == "plt") {
//    read_plt_profile();
  }
}

__global__ void initialize_bv_with_inflow(real *var_info, int n_inflow, cfd::DZone *zone, const real *coordinate_ranges,
                                          int n_scalar, const int *if_profile,
                                          ggxl::VectorField3D<real> *profiles, int *extents) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  int i = (int) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  int j = (int) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  int k = (int) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  auto &x = zone->x, &y = zone->y, &z = zone->z;
  auto &bv = zone->bv, &sv = zone->sv;
  real *rho = var_info, *u = rho + n_inflow, *v = u + n_inflow, *w = v + n_inflow, *p = w + n_inflow, *T = p + n_inflow;
  real *scalar_inflow = T + n_inflow;
  int i_init{0};

  if (n_inflow > 1) {
    for (int l = 0; l < n_inflow - 1; ++l) {
      if (x(i, j, k) >= coordinate_ranges[l * 6] && x(i, j, k) <= coordinate_ranges[l * 6 + 1]
          && y(i, j, k) >= coordinate_ranges[l * 6 + 2] && y(i, j, k) <= coordinate_ranges[l * 6 + 3]
          && z(i, j, k) >= coordinate_ranges[l * 6 + 4] && z(i, j, k) <= coordinate_ranges[l * 6 + 5]) {
        i_init = l + 1;
        break;
      }
    }
  }
  if (if_profile[i_init]) {
    // Assign the profile by 0th order interpolation
    auto &profile = profiles[i_init];
    const auto xx = x(i, j, k), yy = y(i, j, k), zz = z(i, j, k);
    const auto extent = &extents[i_init * 3];
    real d_mix = 1e+6;
    int i0, j0, k0;
    for (int kk = 0; kk < extent[2]; ++kk) {
      for (int jj = 0; jj < extent[1]; ++jj) {
        for (int ii = 0; ii < extent[0]; ++ii) {
          real d = sqrt((xx - profile(ii, jj, kk, 0)) * (xx - profile(ii, jj, kk, 0)) +
                        (yy - profile(ii, jj, kk, 1)) * (yy - profile(ii, jj, kk, 1)) +
                        (zz - profile(ii, jj, kk, 2)) * (zz - profile(ii, jj, kk, 2)));
          if (d < d_mix) {
            d_mix = d;
            i0 = ii;
            j0 = jj;
            k0 = kk;
          }
        }
      }
    }

    // 0th order interpolation
    bv(i, j, k, 0) = profile(i0, j0, k0, 3);
    bv(i, j, k, 1) = profile(i0, j0, k0, 4);
    bv(i, j, k, 2) = profile(i0, j0, k0, 5);
    bv(i, j, k, 3) = profile(i0, j0, k0, 6);
    bv(i, j, k, 4) = profile(i0, j0, k0, 7);
    bv(i, j, k, 5) = profile(i0, j0, k0, 8);
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = profile(i0, j0, k0, 9 + l);
    }
  } else {
    bv(i, j, k, 0) = rho[i_init];
    bv(i, j, k, 1) = u[i_init];
    bv(i, j, k, 2) = v[i_init];
    bv(i, j, k, 3) = w[i_init];
    bv(i, j, k, 4) = p[i_init];
    bv(i, j, k, 5) = T[i_init];
    for (integer l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = scalar_inflow[l * n_inflow + i_init];
    }
  }
}

void cfd::Field::initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow> &inflows,
                                            const std::vector<real> &xs, const std::vector<real> &xe,
                                            const std::vector<real> &ys, const std::vector<real> &ye,
                                            const std::vector<real> &zs, const std::vector<real> &ze,
                                            const cfd::Species &species) const {
  const auto n = inflows.size();
  const integer n_scalar = parameter.get_int("n_scalar");
  std::vector<real> var_info((6 + n_scalar) * n, 0);
  real *rho = var_info.data(), *u = rho + n, *v = u + n, *w = v + n, *p = w + n, *T = p + n;
  real *scalar_inflow = T + n;

  std::vector<integer> if_profile(n, 0);
  std::vector<integer> profile_id(n, 0);
  std::vector<gxl::VectorField3D<real>> profiles(n);
  std::vector<int> extent;
  std::vector<std::array<int, 3>> extents(n);

  for (integer i = 0; i < (integer) inflows.size(); ++i) {
    std::tie(rho[i], u[i], v[i], w[i], p[i], T[i]) = inflows[i].var_info();
    if_profile[i] = inflows[i].inflow_type == 1 ? 1 : 0;
    if (if_profile[i]) {
      profile_id[i] = inflows[i].profile_idx;
      extents[i] = read_profile_to_init(profiles[i], i, parameter, species);
      extent.push_back(extents[i][0]);
      extent.push_back(extents[i][1]);
      extent.push_back(extents[i][2]);
    }
  }
  for (size_t i = 0; i < inflows.size(); ++i) {
    auto sv_this = inflows[i].sv;
    for (int l = 0; l < n_scalar; ++l) {
      scalar_inflow[n * l + i] = sv_this[l];
    }
  }
  real *var_info_device;
  cudaMalloc(&var_info_device, sizeof(real) * var_info.size());
  cudaMemcpy(var_info_device, var_info.data(), sizeof(real) * var_info.size(), cudaMemcpyHostToDevice);
  real *coordinate_ranges_device;
  cudaMalloc(&coordinate_ranges_device, sizeof(real) * 6 * (n - 1));
  std::vector<real> coordinate_ranges;
  for (int l = 0; l < n - 1; ++l) {
    coordinate_ranges.emplace_back(xs[l]);
    coordinate_ranges.emplace_back(xe[l]);
    coordinate_ranges.emplace_back(ys[l]);
    coordinate_ranges.emplace_back(ye[l]);
    coordinate_ranges.emplace_back(zs[l]);
    coordinate_ranges.emplace_back(ze[l]);
  }
  cudaMemcpy(coordinate_ranges_device, coordinate_ranges.data(), sizeof(real) * coordinate_ranges.size(),
             cudaMemcpyHostToDevice);
  std::vector<ggxl::VectorField3D<real>> profile_dev_temp(n);
  for (int l = 0; l < n; ++l) {
    profile_dev_temp[l].allocate_memory(extents[l][0], extents[l][1], extents[l][2], 9 + n_scalar, 0);
    cudaMemcpy(profile_dev_temp[l].data(), profiles[l].data(),
               sizeof(real) * profile_dev_temp[l].size() * (9 + n_scalar), cudaMemcpyHostToDevice);
  }
  ggxl::VectorField3D<real> *profiles_device;
  cudaMalloc(&profiles_device, sizeof(ggxl::VectorField3D<real>) * n);
  cudaMemcpy(profiles_device, profile_dev_temp.data(), sizeof(ggxl::VectorField3D<real>) * n, cudaMemcpyHostToDevice);
  int *extent_device;
  cudaMalloc(&extent_device, sizeof(int) * 3 * n);
  if (!extent.empty())
    cudaMemcpy(extent_device, extent.data(), sizeof(int) * 3 * n, cudaMemcpyHostToDevice);
  int *if_profile_device;
  cudaMalloc(&if_profile_device, sizeof(int) * n);
  cudaMemcpy(if_profile_device, if_profile.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
  dim3 tpb{8, 8, 4};
  if (block.mz == 1) {
    tpb = {16, 16, 1};
  }
  const int ngg{block.ngg};
  dim3 bpg = {(block.mx + 2 * ngg - 1) / tpb.x + 1, (block.my + 2 * ngg - 1) / tpb.y + 1,
              (block.mz + 2 * ngg - 1) / tpb.z + 1};
  initialize_bv_with_inflow<<<bpg, tpb>>>(var_info_device, n, d_ptr, coordinate_ranges_device,
                                          n_scalar, if_profile_device,
                                          profiles_device, extent_device);
  cudaMemcpy(bv.data(), h_ptr->bv.data(), sizeof(real) * h_ptr->bv.size() * 6, cudaMemcpyDeviceToHost);
  cudaMemcpy(sv.data(), h_ptr->sv.data(), sizeof(real) * h_ptr->sv.size() * n_scalar, cudaMemcpyDeviceToHost);
}

void cfd::Field::setup_device_memory(const Parameter &parameter) {
  h_ptr = new DZone;
  const auto mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  h_ptr->mx = mx, h_ptr->my = my, h_ptr->mz = mz, h_ptr->ngg = ngg;

  h_ptr->x.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->x.data(), block.x.data(), sizeof(real) * h_ptr->x.size(), cudaMemcpyHostToDevice);
  h_ptr->y.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->y.data(), block.y.data(), sizeof(real) * h_ptr->y.size(), cudaMemcpyHostToDevice);
  h_ptr->z.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->z.data(), block.z.data(), sizeof(real) * h_ptr->z.size(), cudaMemcpyHostToDevice);
  if (parameter.get_bool("turbulence") && parameter.get_int("turbulence_method") == 2) {
    h_ptr->des_delta.allocate_memory(mx, my, mz, ngg);
    cudaMemcpy(h_ptr->des_delta.data(), block.des_scale.data(), sizeof(real) * h_ptr->des_delta.size(),
               cudaMemcpyHostToDevice);
  }

  auto n_bound{block.boundary.size()};
  auto n_inner{block.inner_face.size()};
  auto n_par{block.parallel_face.size()};
  auto mem_sz = sizeof(Boundary) * n_bound;
  cudaMalloc(&h_ptr->boundary, mem_sz);
  cudaMemcpy(h_ptr->boundary, block.boundary.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(InnerFace) * n_inner;
  cudaMalloc(&h_ptr->innerface, mem_sz);
  cudaMemcpy(h_ptr->innerface, block.inner_face.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(ParallelFace) * n_par;
  cudaMalloc(&h_ptr->parface, mem_sz);
  cudaMemcpy(h_ptr->parface, block.parallel_face.data(), mem_sz, cudaMemcpyHostToDevice);

  h_ptr->jac.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->jac.data(), block.jacobian.data(), sizeof(real) * h_ptr->jac.size(), cudaMemcpyHostToDevice);
  h_ptr->metric.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->metric.data(), block.metric.data(), sizeof(gxl::Matrix<real, 3, 3, 1>) * h_ptr->metric.size(),
             cudaMemcpyHostToDevice);

  h_ptr->cv.allocate_memory(mx, my, mz, n_var, ngg);
  h_ptr->bv.allocate_memory(mx, my, mz, 6, ngg);
  h_ptr->bv_last.allocate_memory(mx, my, mz, 4, 0);
  h_ptr->vel.allocate_memory(mx, my, mz, ngg);
  h_ptr->acoustic_speed.allocate_memory(mx, my, mz, ngg);
  h_ptr->mach.allocate_memory(mx, my, mz, ngg);
  h_ptr->mul.allocate_memory(mx, my, mz, ngg);
  h_ptr->thermal_conductivity.allocate_memory(mx, my, mz, ngg);

  const auto n_spec{parameter.get_int("n_spec")};
  const auto n_scalar = parameter.get_int("n_scalar");
  h_ptr->sv.allocate_memory(mx, my, mz, n_scalar, ngg);
  h_ptr->rho_D.allocate_memory(mx, my, mz, n_spec, ngg);
  if (n_spec > 0) {
    h_ptr->gamma.allocate_memory(mx, my, mz, ngg);
    h_ptr->cp.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("reaction") == 1) {
      // Finite rate chemistry
      if (const integer chemSrcMethod = parameter.get_int("chemSrcMethod");chemSrcMethod == 1) {
        // EPI
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec * n_spec, 0);
      } else if (chemSrcMethod == 2) {
        // DA
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec, 0);
      }
    } else if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
      // Flamelet model
      h_ptr->scalar_diss_rate.allocate_memory(mx, my, mz, ngg);
      // Maybe we can also implicitly treat the source term here.
    }
  }
  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS method or DES method
    h_ptr->mut.allocate_memory(mx, my, mz, ngg);
    h_ptr->turb_therm_cond.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      h_ptr->wall_distance.allocate_memory(mx, my, mz, ngg);
      if (parameter.get_int("turb_implicit") == 1) {
        h_ptr->turb_src_jac.allocate_memory(mx, my, mz, 2, 0);
      }
    }
  }
  if (parameter.get_int("if_compute_wall_distance") == 1) {
    h_ptr->wall_distance.allocate_memory(mx, my, mz, ngg);
  }

  h_ptr->dq.allocate_memory(mx, my, mz, n_var, 0);
  h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 0);
  h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 0);
  if (parameter.get_int("implicit_method") == 1) { // DPLUR
    // If DPLUR type, when computing the products of convective jacobian and dq, we need 1 layer of ghost grids whose dq=0.
    // Except those inner or parallel communication faces, they need to get the dq from neighbor blocks.
    h_ptr->dq.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->dq0.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->dqk.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 1);
    h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 1);
  }
//  if (parameter.get_bool("steady")) { // steady simulation
  h_ptr->dt_local.allocate_memory(mx, my, mz, 0);
  h_ptr->unphysical.allocate_memory(mx, my, mz, 0);
//  }
  if (parameter.get_int("inviscid_scheme") == 2) {
    // Roe scheme
    h_ptr->entropy_fix_delta.allocate_memory(mx, my, mz, 1);
  }

  if (!parameter.get_bool("steady")) {
    // unsteady simulation
    if (parameter.get_int("temporal_scheme") == 2) {
      // dual time stepping
      h_ptr->qn1.allocate_memory(mx, my, mz, n_var, ngg);
      h_ptr->qn_star.allocate_memory(mx, my, mz, n_var, ngg);
      h_ptr->in_last_step.allocate_memory(mx, my, mz, 4, 0);
    }
    if (parameter.get_int("temporal_scheme") == 3) {
      // rk scheme
      h_ptr->qn.allocate_memory(mx, my, mz, n_var, ngg);
    }
  }

  if (parameter.get_bool("if_collect_statistics")) {
    // If we need to collect the statistics, we need to allocate memory for the data.
    if (parameter.get_bool("perform_spanwise_average")) {
      h_ptr->mean_value.allocate_memory(mx, my, 1, 6 + n_scalar, 0);
      h_ptr->reynolds_stress_tensor.allocate_memory(mx, my, 1, 6, 0);
      h_ptr->user_defined_statistical_data.allocate_memory(mx, my, 1, UserDefineStat::n_stat, 0);
    } else {
      h_ptr->mean_value.allocate_memory(mx, my, mz, 6 + n_scalar, 0);
      h_ptr->reynolds_stress_tensor.allocate_memory(mx, my, mz, 6, 0);
      h_ptr->user_defined_statistical_data.allocate_memory(mx, my, mz, UserDefineStat::n_stat, 0);
    }

    h_ptr->firstOrderMoment.allocate_memory(mx, my, mz, 6 + n_scalar, 0);
    h_ptr->velocity2ndMoment.allocate_memory(mx, my, mz, 6, 0);
    h_ptr->userCollectForStat.allocate_memory(mx, my, mz, UserDefineStat::n_collect, 0);
  }

  // Assign memory for auxiliary variables
  h_ptr->udv.allocate_memory(mx, my, mz, UserDefineIO::n_dynamic_auxiliary, ngg);

  cudaMalloc(&d_ptr, sizeof(DZone));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DZone), cudaMemcpyHostToDevice);
}

void cfd::Field::copy_data_from_device(const Parameter &parameter) {
  const auto size = (block.mx + 2 * block.ngg) * (block.my + 2 * block.ngg) * (block.mz + 2 * block.ngg);

  cudaMemcpy(bv.data(), h_ptr->bv.data(), 6 * size * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ov.data(), h_ptr->mach.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    cudaMemcpy(ov[1], h_ptr->mut.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(sv.data(), h_ptr->sv.data(), parameter.get_int("n_scalar") * size * sizeof(real), cudaMemcpyDeviceToHost);
  if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
    cudaMemcpy(ov[2], h_ptr->scalar_diss_rate.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  }
  copy_auxiliary_data_from_device(*this, size);
}
