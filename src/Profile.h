#pragma once

#include "define.h"
#include "gxl_lib/Array.hpp"

namespace cfd {

struct Profiles{
  ggxl::VectorField2D<real> profile;
};



// This function reads the given profile with filename "profile_name" and assign the values to the given zone.
// The return value is which label is this profile. E.g., there are 2 inflow profiles in the x direction;
// then the return value is used to distinguish between these two profiles.
//integer read_profile(const std::string &profile_name, DZone *zone, integer n);
}