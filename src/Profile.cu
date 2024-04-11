#include "Profile.h"

namespace cfd{

//integer read_profile(const std::string &profile_name, DZone *zone, integer n) {
//  std::ifstream profile_file(profile_name);
//  std::string line;
//  std::getline(profile_file, line);
//  std::istringstream iss(line);
//  integer label;
//  iss >> label;
//  for (integer i = 0; i < n; ++i) {
//    std::getline(profile_file, line);
//    std::istringstream iss(line);
//    iss >> zone[i].x >> zone[i].y >> zone[i].z >> zone[i].u >> zone[i].v >> zone[i].w >> zone[i].p >> zone[i].T;
//    for (integer j = 0; j < n_species; ++j) {
//      iss >> zone[i].sv[j];
//    }
//  }
//  profile_file.close();
//  return 0;
//}
}