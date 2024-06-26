#pragma once

#include "Driver.cuh"
#include <cstdio>
#include "SteadySim.cuh"
#include "FirstOrderEuler.cuh"
#include "RK.cuh"
#include "DualTimeStepping.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
void simulate(Driver<mix_model, turb> &driver) {
  const auto &parameter{driver.parameter};
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    // The methods which use only bv do not need to save cv at all, which is the case in steady simulations.
    // In those methods, such as Roe, AUSM..., we do not store the cv variables.
    steady_simulation<mix_model, turb>(driver);
  } else {
    // In my design, the inviscid method may use cv, or only bv.
    // The methods which use only bv, such as Roe, AUSM..., do not need to save cv at all.
    // But we still store the cv array here. Because in methods such as RK, we need multiple stages to update the cv variables.
    // However, in high-order methods, such as the WENO method, we need to store the cv variables.
    // When this happens, the corresponding boundary conditions, data communications would all involve the update of cv.
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    // Inviscid methods which use only bv
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