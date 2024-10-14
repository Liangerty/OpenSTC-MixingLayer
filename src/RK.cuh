#pragma once

#include "Driver.cuh"
#include "FieldOperation.cuh"
#include "Monitor.cuh"
#include "StatisticsCollector.cuh"
#include "SpongeLayer.cuh"

namespace cfd {

namespace SSPRK3 {
__device__ constexpr real a[3]{1.0, 0.75, 1.0 / 3.0};
__device__ constexpr real b[3]{0.0, 0.25, 2.0 / 3.0};
__device__ constexpr real c[3]{1.0, 0.25, 2.0 / 3.0};
}

template<MixtureModel mix_model, class turb_method>
__global__ void update_cv_and_bv_rk(cfd::DZone *zone, DParameter *param, real dt, int rk);

template<MixtureModel mix_model, class turb>
void RK3(Driver<mix_model, turb> &driver) {
  if (driver.myid == 0) {
    printf("\n****************************Time advancement starts*****************************\n");
  }

  dim3 tpb{8, 8, 4};
  auto &mesh{driver.mesh};
  const int n_block{mesh.n_block};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (int b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  std::vector<cfd::Field> &field{driver.field};
  const int ng_1 = 2 * mesh[0].ngg - 1;
  DParameter *param{driver.param};
  for (auto b = 0; b < n_block; ++b) {
    // Store the initial value of the flow field
    store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
    // Compute the conservative variables from basic variables
    // In unsteady simulations, because of the upwind high-order method to be used;
    // we need the conservative variables.
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model, turb><<<BPG, tpb>>>(field[b].d_ptr, param);
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }

  auto &parameter{driver.parameter};
  IOManager<mix_model, turb> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model, turb> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  int output_time_series = parameter.get_int("output_time_series");

  Monitor monitor(parameter, driver.spec, mesh);
  const int if_monitor{parameter.get_int("if_monitor")};

  int step{parameter.get_int("step")};
  int total_step{parameter.get_int("total_step") + step};
  const int output_screen = parameter.get_int("output_screen");
  const int n_var{parameter.get_int("n_var")};
  real total_simulation_time{parameter.get_real("total_simulation_time")};
  const int output_file = parameter.get_int("output_file");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  real dt{1e+6};
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};
  if (fixed_time_step) {
    dt = parameter.get_real("dt");
  }

  const bool if_collect_statistics{parameter.get_bool("if_collect_statistics")};
  const int collect_statistics_iter_start{parameter.get_int("start_collect_statistics_iter")};
  auto &statistics_collector{driver.stat_collector};

  while (!finished) {
    ++step;

    update_physical_time<<<1, 1>>>(param, physical_time);

    // Start a single iteration
    // First, store the value of last step if we need to compute residual
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    // For every iteration, we need to save the conservative variables of the last step.
    for (auto b = 0; b < n_block; ++b) {
      cudaMemcpy(field[b].h_ptr->qn.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
                 cudaMemcpyDeviceToDevice);
    }

    if (parameter.get_bool("sponge_layer")) {
      for (int b = 0; b < n_block; ++b) {
        update_sponge_layer_value<<<bpg[b], tpb>>>(field[b].d_ptr, param);
      }
      update_sponge_iter(param, parameter);
    }

    // Rk inner iteration
    for (int rk = 0; rk < 3; ++rk) {
      for (auto b = 0; b < n_block; ++b) {
        // Set dq to 0
        cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));

        // First, compute the source term, because properties such as mut are updated here, which is used by computing dt.
        compute_source<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      }

      // For unsteady simulations, the time step should be consistent in all grid points
      // Because if Positive-preserving WENO is used, the dt will be needed.
      // Therefore, we compute the time step ahead of inviscid flux.
      // Besides, when simulating turbulent flows, mut is needed in computing the time step,
      // which makes the computation of source term ahead of this.
      if (!fixed_time_step && rk == 0) {
        // compute the local time step
        for (auto b = 0; b < n_block; ++b)
          local_time_step<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
        // After all processes and all blocks computing dt_local, we compute the global time step.
        dt = global_time_step(mesh, parameter, field);

        update_dt_global<<<1, 1>>>(param, dt);
      }

      for (auto b = 0; b < n_block; ++b) {
        // Second, for each block, compute the residual dq
        compute_inviscid_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var, parameter);
        compute_viscous_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var);

        // Explicit temporal schemes should not use any implicit treatment.

        // update basic variables
        update_cv_and_bv_rk<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt, rk);

        // limit unphysical values computed by the program
        if (parameter.get_bool("limit_flow"))
          limit_flow<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);

        // Apply boundary conditions
        // Attention: "driver" is a template class, when a template class calls a member function of another template,
        // the compiler will not treat the called function as a template function,
        // so we need to explicitly specify the "template" keyword here.
        // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
        driver.bound_cond.template apply_boundary_conditions<mix_model, turb>(mesh[b], field[b], param);
      }
      // Third, transfer data between and within processes
      data_communication<mix_model, turb>(mesh, field, parameter, step, param);

      if (mesh.dimension == 2) {
        for (auto b = 0; b < n_block; ++b) {
          const auto mx{mesh[b].mx}, my{mesh[b].my};
          dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
          eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
        }
      }

      // update physical properties such as Mach number, transport coefficients et, al.
      for (auto b = 0; b < n_block; ++b) {
        int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
        update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
      }
    }

    // Finally, test if the simulation reaches convergence state
    physical_time += dt;
    parameter.update_parameter("solution_time", physical_time);
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
      if (driver.myid == 0) {
        unsteady_screen_output(step, err_max, driver.time, driver.res, dt, physical_time);
      }
    }
    cudaDeviceSynchronize();
    if (physical_time > total_simulation_time || step == total_step) {
      if (physical_time > total_simulation_time)
        printf("The simulated physical time is %e, which is larger than required physical time %e.\n", physical_time,
               total_simulation_time);
      else if (step == total_step)
        printf("The step %d reaches specified total step.\n", step);
      finished = true;
    }
    if (if_monitor) {
      monitor.monitor_point(step, physical_time, field);
      if (step % parameter.get_int("slice_frequency") == 0) {
        monitor.output_slices(parameter, field, step, physical_time);
      }
    }
    if (if_collect_statistics && step > collect_statistics_iter_start) {
//      statistics_collector.template collect_data<mix_model, turb>(param);
      statistics_collector.collect_data(param);
    }
    if (step % output_file == 0 || finished) {
      ioManager.print_field(step, parameter, physical_time);
      if (if_collect_statistics && step > collect_statistics_iter_start)
        statistics_collector.export_statistical_data(param, parameter.get_bool("perform_spanwise_average"));
      post_process(driver);
      if (if_monitor)
        monitor.output_data();
      if (parameter.get_bool("sponge_layer")) {
        output_sponge_layer(parameter, field, mesh, driver.spec);
      }
    }
    if (output_time_series > 0 && step % output_time_series == 0) {
      timeSeriesIOManager.print_field(step, parameter, physical_time);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model, class turb_method>
__global__ void update_cv_and_bv_rk(cfd::DZone *zone, DParameter *param, real dt, int rk) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (int) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (int) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (int) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;
  auto &qn = zone->qn;

  real dt_div_jac = dt / zone->jac(i, j, k);
  for (int l = 0; l < param->n_var; ++l) {
    cv(i, j, k, l) = SSPRK3::a[rk] * qn(i, j, k, l) + SSPRK3::b[rk] * cv(i, j, k, l) +
                     SSPRK3::c[rk] * dt_div_jac * zone->dq(i, j, k, l);
  }
  if (extent[2] == 1) {
    cv(i, j, k, 3) = 0;
  }

  auto &bv = zone->bv;

  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;
  auto V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3); //V^2

  auto &sv = zone->sv;
  if constexpr (mix_model != MixtureModel::FL) {
    // For multiple species or RANS methods, there will be scalars to be computed
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
    }
  } else {
    // Flamelet model
    for (int l = 0; l < param->n_scalar_transported; ++l) {
      sv(i, j, k, l + param->n_spec) = cv(i, j, k, 5 + l) * density_inv;
    }
    real yk_ave[MAX_SPEC_NUMBER];
    memset(yk_ave, 0, sizeof(real) * param->n_spec);
    compute_massFraction_from_MixtureFraction(zone, i, j, k, param, yk_ave);
    for (int l = 0; l < param->n_spec; ++l) {
      sv(i, j, k, l) = yk_ave[l];
    }
  }
  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
  } else {
    // Air
    bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * V2);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }
}
}