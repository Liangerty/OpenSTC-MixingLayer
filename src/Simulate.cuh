#pragma once

#include "Driver.cuh"
#include <cstdio>
#include "SteadySim.cuh"
#include "FirstOrderEuler.cuh"
#include "RK.cuh"
#include "DualTimeStepping.cuh"
#include "kernels.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
void simulate(Driver<mix_model, turb> &driver) {
  auto &parameter{driver.parameter};
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    // The methods which use only bv do not need to save cv at all, which is the case in steady simulations.
    // In those methods, such as Roe, AUSM..., we do not store the cv variables.
    steady_simulation<mix_model, turb>(driver);
  } else {
    if (parameter.get_bool("steady_before_transient")) {
      if (parameter.get_int("initial") == 0) {
        // When we start a new simulation, and we want to run a steady simulation before the transient simulation.
        real cfl_transient = parameter.get_real("cfl");
        real cfl_steady = parameter.get_real("cfl_steady");
        bool change_cfl = false;
        if (abs(cfl_steady - cfl_transient) > 1e-10) {
          change_cfl = true;
          modify_cfl<<<1, 1>>>(driver.param, cfl_steady);
        }
        steady_simulation<mix_model, turb>(driver);
        if (change_cfl) {
          modify_cfl<<<1, 1>>>(driver.param, cfl_transient);
        }
      } else {
        parameter.update_parameter("steady_before_transient", false);
      }
    }

    real physical_time = parameter.get_real("solution_time");
    if (real t0 = parameter.get_real("set_current_physical_time");t0 > -1e-10) {
      physical_time = t0;
      parameter.update_parameter("solution_time", physical_time);
    }
    int myid = parameter.get_int("myid");
    if (myid == 0) {
      printf("\n\tCurrent physical time is %es\n", physical_time);
    }
    real length = parameter.get_real("domain_length");
    real u = parameter.get_real("v_inf");
    if (parameter.get_int("problem_type") == 1) {
      u = parameter.get_real("convective_velocity");
    }
    if (real u_char = parameter.get_real("characteristic_velocity");u_char > 0) {
      u = u_char;
    }
    real flowThroughTime = length / u;
    parameter.update_parameter("flow_through_time", flowThroughTime);
    auto n_ftt = parameter.get_real("n_flowThroughTime");
    if (n_ftt > 0) {
      physical_time += n_ftt * flowThroughTime;
      parameter.update_parameter("total_simulation_time", physical_time);
      if (myid == 0) {
        printf("\t-> %10.4es : Flow through time computed with L(%9.3em)/U(%9.3em/s)\n", flowThroughTime, length, u);
        printf("\t-> %10.4e  : Number of flow through time to be computed.\n", n_ftt);
        printf("\t-> %10.4es : Physical time to be simulated.\n", n_ftt * flowThroughTime);
        printf("\t-> %10.4es : The end of the simulation time.\n", physical_time);
      }
    } else {
      // If the flow through time is not set, the total simulation time is added to the solution time.
      physical_time += parameter.get_real("total_simulation_time");
      if (myid == 0) {
        printf("\t-> %10.4es : Physical time to be simulated.\n", parameter.get_real("total_simulation_time"));
        printf("\t-> %10.4es : The end of the simulation time.\n", physical_time);
      }
      parameter.update_parameter("total_simulation_time", physical_time);
    }

    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    switch (temporal_tag) {
      case 1: // Explicit Euler, only first order time accuracy, should be avoided in most cases.
        first_order_euler<mix_model, turb, reconstruct_bv>(driver);
        break;
      case 2:
        dual_time_stepping<mix_model, turb>(driver);
        break;
      case 3:
      default:
        RK3<mix_model, turb>(driver);
        break;
    }
  }
  driver.deallocate();
}
}