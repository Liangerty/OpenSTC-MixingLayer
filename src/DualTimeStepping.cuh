#pragma once

#include "Define.h"
#include "Driver.cuh"
#include <cstdio>
#include "TimeAdvanceFunc.cuh"
#include "IOManager.h"
#include "Monitor.cuh"
#include "SourceTerm.cuh"
#include "SchemeSelector.cuh"
#include "ImplicitTreatmentHPP.cuh"
#include "DataCommunication.cuh"
#include "Residual.cuh"
#include "PostProcess.h"

namespace cfd {
__global__ void compute_qn_star(cfd::DZone *zone, integer n_var, real dt_global);

template<MixtureModel mixture_model, class turb_method>
void dual_time_stepping_implicit_treat(const Block &block, DParameter *param, DZone *d_ptr, DZone *h_ptr,
                                       Parameter &parameter, DBoundCond &bound_cond, real diag_factor);

__global__ void compute_modified_rhs(cfd::DZone *zone, integer n_var, real dt_global);

bool inner_converged(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, integer iter,
                     std::array<real, 4> &res_scale, integer myid, integer step, integer &inner_iter);

__global__ void compute_square_of_dbv_wrt_last_inner_iter(cfd::DZone *zone);

__global__ void store_last_iter(cfd::DZone *zone);

template<MixtureModel mix_model, class turb>
void dual_time_stepping(Driver<mix_model, turb> &driver) {
  auto &parameter{driver.parameter};
  const real dt = parameter.get_real("dt");
  const real diag_factor = 1.5 / dt;
  if (driver.myid == 0) {
    printf("Unsteady flow simulation with 2rd order dual-time stepping for time advancing.\n");
    printf("The physical time step is %e.\n", dt);
  }

  dim3 tpb{8, 8, 4};
  auto &mesh{driver.mesh};
  const integer n_block{mesh.n_block};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  std::vector<cfd::Field> &field{driver.field};
  const integer n_var{parameter.get_int("n_var")};
  const integer ng_1 = 2 * mesh[0].ngg - 1;
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

    // Initialize qn1. This should be a condition if we read from previous or initialize with the current cv.
    // We currently initialize with the cv.
    cudaMemcpy(field[b].h_ptr->qn1.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
               cudaMemcpyDeviceToDevice);
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }

  IOManager<mix_model, turb> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model, turb> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);

  Monitor monitor(parameter, driver.spec);
  const integer if_monitor{parameter.get_int("if_monitor")};

  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  const integer output_screen = parameter.get_int("output_screen");
  real total_simulation_time{parameter.get_real("total_simulation_time")};
  const integer output_file = parameter.get_int("output_file");
  integer inner_iteration = parameter.get_int("inner_iteration");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  const bool if_collect_statistics{parameter.get_bool("if_collect_statistics")};
  const int collect_statistics_iter_start{parameter.get_int("start_collect_statistics_iter")};
  auto &stat_collector{driver.stat_collector};

  bool monitor_inner_iteration;
  std::array<real, 4> res_scale_inner{1, 1, 1, 1};
  const integer iteration_adjust_step{parameter.get_int("iteration_adjust_step")};

  while (!finished) {
    ++step;

    // Start a single iteration
    // First, store the value of last step if we need to compute residual
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    // For every time step, we need to compute the qn_star and replace the qn1 with the current cv.
    for (auto b = 0; b < n_block; ++b) {
      compute_qn_star<<<bpg[b], tpb>>>(field[b].d_ptr, n_var, dt);
      cudaMemcpy(field[b].h_ptr->qn1.data(), field[b].h_ptr->cv.data(),
                 field[b].h_ptr->cv.size() * sizeof(real) * n_var,
                 cudaMemcpyDeviceToDevice);
    }

    if (step % iteration_adjust_step == 0) {
      monitor_inner_iteration = true;
    } else {
      monitor_inner_iteration = false;
    }

    // dual-time stepping inner iteration
    for (integer iter = 1; iter <= inner_iteration; ++iter) {
      for (auto b = 0; b < n_block; ++b) {
        if (monitor_inner_iteration) {
          store_last_iter<<<bpg[b], tpb>>>(field[b].d_ptr);
        }

        // Set dq to 0
        cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));

        // Second, for each block, compute the residual dq
        // First, compute the source term, because properties such as mut are updated here.
        compute_source<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
        compute_inviscid_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var, parameter);
        compute_viscous_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var);

        // compute the local time step
        local_time_step<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);

        // Implicit treat
        dual_time_stepping_implicit_treat<mix_model, turb>(mesh[b], param, field[b].d_ptr, field[b].h_ptr, parameter,
                                                           driver.bound_cond, diag_factor);

        // update basic and conservative variables
        update_cv_and_bv<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);

        limit_flow<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

        // Apply boundary conditions
        // Attention: "driver" is a template class, when a template class calls a member function of another template,
        // the compiler will not treat the called function as a template function,
        // so we need to explicitly specify the "template" keyword here.
        // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
        driver.bound_cond.template apply_boundary_conditions<mix_model, turb, true>(mesh[b], field[b], param);
      }
      // Third, transfer data between and within processes
      data_communication<mix_model, turb, true>(mesh, field, parameter, step, param);

      if (mesh.dimension == 2) {
        for (auto b = 0; b < n_block; ++b) {
          const auto mx{mesh[b].mx}, my{mesh[b].my};
          dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
          eliminate_k_gradient<true><<<BPG, tpb>>>(field[b].d_ptr, param);
        }
      }

      // update physical properties such as Mach number, transport coefficients et, al.
      for (auto b = 0; b < n_block; ++b) {
        integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
        dim3 BPG{(mx + 1) / tpb.x + 1, (my + 1) / tpb.y + 1, (mz + 1) / tpb.z + 1};
        update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
      }

      if (monitor_inner_iteration)
        if (inner_converged(mesh, field, parameter, iter, res_scale_inner, driver.myid, step, inner_iteration))
          break;
    }

    // Finally, test if the simulation reaches convergence state
    physical_time += dt;
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
      if (driver.myid == 0) {
        unsteady_screen_output(step, err_max, driver.time, driver.res, dt, physical_time);
      }
    }
    cudaDeviceSynchronize();
    if (physical_time > total_simulation_time || step == total_step) {
      finished = true;
    }
    if (if_monitor) {
      monitor.monitor_point(step, physical_time, field);
    }
    if (if_collect_statistics && step > collect_statistics_iter_start) {
      stat_collector.template collect_data<mix_model, turb>(param);
    }
    if (step % output_file == 0 || finished) {
      ioManager.print_field(step, parameter, physical_time);
      timeSeriesIOManager.print_field(step, parameter, physical_time);
      if (if_collect_statistics && step > collect_statistics_iter_start)
        stat_collector.export_statistical_data(param, parameter.get_bool("perform_spanwise_average"));
      post_process(driver);
      if (if_monitor)
        monitor.output_data();
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
  }
  delete[] bpg;
}

template<MixtureModel mixture_model, class turb_method>
void dual_time_stepping_implicit_treat(const Block &block, DParameter *param, DZone *d_ptr,
                                       DZone *h_ptr, Parameter &parameter,
                                       DBoundCond &bound_cond, real diag_factor) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer dim{extent[2] == 1 ? 2 : 3};
  dim3 tpb{8, 8, 4};
  if (dim == 2) {
    tpb = {16, 16, 1};
  }
  const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
  compute_modified_rhs<<<bpg, tpb>>>(d_ptr, parameter.get_int("n_var"), parameter.get_real("dt"));

  DPLUR<mixture_model, turb_method>(block, param, d_ptr, h_ptr, parameter, bound_cond, diag_factor);
}

}