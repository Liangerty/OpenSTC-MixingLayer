#pragma once

#include "Define.h"
//#include "mpi.h"
//#include "Field.h"
#include <array>
//#include "ChemData.h"
//#include <filesystem>
//#include "gxl_lib/MyString.h"

namespace cfd {

struct UserDefineIO {
  /**********************************************************************************************/
  constexpr static integer n_auxiliary = 1;
  constexpr static integer n_static_auxiliary = 0;
  constexpr static integer n_dynamic_auxiliary = n_auxiliary - n_static_auxiliary;
  /**********************************************************************************************/
};

struct Field;

void copy_auxiliary_data_from_device(cfd::Field &field, integer size);

}