#include "Parameter.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "gxl_lib/MyString.h"
#include <fmt/format.h>
#include "kernels.cuh"

cfd::Parameter::Parameter(int *argc, char ***argv) {
  int myid, n_proc;
  MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  int_parameters["myid"] = myid;
  int_parameters["n_proc"] = n_proc;

  setup_default_settings();
  read_param_from_file();
  diagnose_parallel_info();
  setup_gpu_device(n_proc, myid);
  deduce_known_info();
}

cfd::Parameter::Parameter(const std::string &filename) {
  std::ifstream file(filename);
  read_one_file(file);
  file.close();
}

void cfd::Parameter::read_param_from_file() {
  for (auto &name: file_names) {
    std::ifstream file(name);
    read_one_file(file);
    file.close();
  }
}

void cfd::Parameter::read_one_file(std::ifstream &file) {
  std::string input{}, type{}, key{}, temp{};
  std::istringstream line(input);
  while (std::getline(file, input)) {
    if (input.starts_with("//") || input.starts_with("!") || input.empty()) {
      continue;
    }
    line.clear();
    line.str(input);
    line >> type;
    line >> key >> temp;
    if (type == "int") {
      int val{};
      line >> val;
      int_parameters[key] = val;
    } else if (type == "real") {
      real val{};
      line >> val;
      real_parameters[key] = val;
    } else if (type == "bool") {
      bool val{};
      line >> val;
      bool_parameters[key] = val;
    } else if (type == "string") {
      std::string val{};
      line >> val;
      string_parameters[key] = val;
    } else if (type == "array") {
      if (key == "int") {
        std::vector<int> arr;
        std::string name{temp};
        line >> temp >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        int_array[name] = arr;
      } else if (key == "real") {
        std::vector<real> arr;
        std::string name{temp};
        line >> temp >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        real_array[name] = arr;
      } else if (key == "string") {
        std::vector<std::string> arr;
        std::string name{temp};
        line >> temp >> temp;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        string_array[name] = arr;
      }
    } else if (type == "struct") {
      auto the_struct = read_struct(file);
      struct_array[key] = the_struct;
    } else if (type == "range") {
      std::string name{temp};
      line >> temp >> temp; // {
      if (key == "real") {
        std::vector<real> arr;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        auto num = arr.size();
        if (num == 2) {
          real_range[name] = Range<real>(arr[0], arr[1]);
        } else if (num == 4) {
          real_range[name] = Range<real>(arr[0], arr[1], arr[2], arr[3]);
        } else if (num == 6) {
          real_range[name] = Range<real>(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
        } else {
          fmt::print("The number of values in the range {} is not correct.\n", name);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      } else if (key == "int") {
        std::vector<int> arr;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        auto num = arr.size();
        if (num == 2) {
          int_range[name] = Range<int>(arr[0], arr[1]);
        } else if (num == 4) {
          int_range[name] = Range<int>(arr[0], arr[1], arr[2], arr[3]);
        } else if (num == 6) {
          int_range[name] = Range<int>(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
        } else {
          fmt::print("The number of values in the range {} is not correct.\n", name);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }
  }
  file.close();
}

template<typename T>
int cfd::Parameter::read_line_to_array(std::istringstream &line, std::vector<T> &arr) {
  std::string temp{};
  while (line >> temp) {
    if (temp == "}") {
      // Which means the array has been read
      return 0;
    }
    if (temp == "//") {
      // which means the array is not over, but values are on the next line
      break;
    }
    T val{};
    if constexpr (std::is_same_v<T, real>) {
      val = std::stod(temp);
    } else if constexpr (std::is_same_v<T, int>) {
      val = std::stoi(temp);
    } else if constexpr (std::is_same_v<T, std::string>) {
      val = temp;
    }
    arr.push_back(val);
  }
  return 1; // Which means we need to read the next line
}

std::map<std::string, std::variant<std::string, int, real>> cfd::Parameter::read_struct(std::ifstream &file) {
  std::map<std::string, std::variant<std::string, int, real>> struct_to_read;
  std::string input{}, key{}, temp{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(file, input, line)) {
    line >> key;
    if (key == "}") { // the "}" must be placed on a separate line.
      break;
    }
    if (key == "string") {
      std::string val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "int") {
      int val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "real") {
      real val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    }
  }
  return struct_to_read;
}

void cfd::Parameter::deduce_known_info() {
  int_parameters.emplace("step", 0);

  // Ghost grid number is decided from the inviscid scheme and viscous scheme.
  // For common 2nd order schemes, we need 2 ghost grids.
  // Here, because we now only implement 2nd order central difference for viscous flux, we need 2 ghost grids.
  // The main concern for the number of ghost grids is the inviscid flux, which in turn is decided from the reconstruction method.
  int ngg{2};
  int reconstruction_scheme = get_int("reconstruction");
  std::string reconstruction_name{};
  if (reconstruction_scheme == 1) {
    reconstruction_name = "1st-order upwind";
  } else if (reconstruction_scheme == 2) {
    reconstruction_name = "MUSCL";
  } else if (reconstruction_scheme == 3) {
    reconstruction_name = "NND2";
  } else if (reconstruction_scheme == 4) {
    reconstruction_name = "AWENO5";
    ngg = 3;
  } else if (reconstruction_scheme == 5) {
    reconstruction_name = "AWENO7";
    ngg = 4;
  }

  if (int_parameters["myid"] == 0)
    fmt::print("\n{:*^80}\n", "Inviscid scheme Information");
  int inviscid_scheme{get_int("inviscid_scheme")};
  std::string inviscid_scheme_name{};
  if (inviscid_scheme == 2) {
    inviscid_scheme_name = "Roe";
    if (int_parameters["myid"] == 0)
      fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
  } else if (inviscid_scheme == 3) {
    inviscid_scheme_name = "AUSM+";
    if (int_parameters["myid"] == 0)
      fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
  } else if (inviscid_scheme == 4) {
    inviscid_scheme_name = "HLLC";
    if (int_parameters["myid"] == 0)
      fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
  } else if (inviscid_scheme == 51) {
    inviscid_scheme_name = "WENO5-cp";
    ngg = 3;
  } else if (inviscid_scheme == 52) {
    inviscid_scheme_name = "WENO5-ch";
    ngg = 3;
  } else if (inviscid_scheme == 71) {
    inviscid_scheme_name = "WENO7-cp";
    ngg = 4;
  } else if (inviscid_scheme == 72) {
    inviscid_scheme_name = "WENO7-ch";
    ngg = 4;
  } else if (inviscid_scheme == 6) {
    inviscid_scheme_name = "energy-preserving-6";
    ngg = 3;
  }

  update_parameter("ngg", ngg);
  if (int_parameters["myid"] == 0) {
    fmt::print("\t->-> {:<20} : inviscid scheme.\n", inviscid_scheme_name);
    fmt::print("\t->-> {:<20} : number of ghost layers\n", ngg);
  }

  if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 71 || inviscid_scheme == 72) {
    if (bool_parameters["positive_preserving"] == 1) {
      fmt::print("\t->-> {:<20} : positive preserving.\n", "Yes");
    }
  }

  // Next, based on the reconstruction scheme and inviscid flux method, we re-assign a new field called "inviscid_tag"
  // to identify the method to be used.
  // 2-Roe, 3-AUSM+, 4-HLLC. These are used by default if the corresponding reconstruction methods are 1stOrder, MUSCL, or NND2.
  // Once the reconstruction method is changed to high-order ones, such as WENO5, WENO7, etc., we would use a new tag to identify
  // them. Because, the reconstructed variables would be conservative variables, and there may be additional operations,
  // such as characteristic reconstruction, high-order central difference, etc.
  // Summary: 2-Roe, 3-AUSM+, 4-HLLC, 14-HLLC+WENO
  int inviscid_tag{get_int("inviscid_scheme")};
  int inviscid_type{0};
  if (inviscid_tag == 2) {
    inviscid_type = 2;
  }
  if (reconstruction_scheme == 4 || reconstruction_scheme == 5) {
    inviscid_type = 1;
    // WENO reconstructions
  }
  if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 71 || inviscid_scheme == 72) {
    inviscid_type = 3;
  }
  if (inviscid_scheme == 6) {
    inviscid_type = 4;
  }
  update_parameter("inviscid_type", inviscid_type);

  if (bool_parameters["steady"] == 0 && int_parameters["temporal_scheme"] == 3) {
    // RK-3, the chemical source should be treated explicitly.
    update_parameter("chemSrcMethod", 0);
  }

  update_parameter("n_var", 5);
  update_parameter("n_turb", 0);
  int n_scalar{0};
  if (bool_parameters["turbulence"] == 1) {
    if (int_parameters["turbulence_method"] == 1 || int_parameters["turbulence_method"] == 2) { // RANS or DES
      if (int_parameters["RANS_model"] == 1) {// SA
        update_parameter("n_turb", 1);
        update_parameter("n_var", 5 + 1);
        n_scalar += 1;
      } else { // SST
        update_parameter("n_turb", 2);
        update_parameter("n_var", 5 + 2);
        n_scalar += 2;
      }
    }
  } else {
    update_parameter("RANS_model", 0);
  }
  update_parameter("n_scalar", n_scalar);
}

void cfd::Parameter::setup_default_settings() {
  int_parameters["problem_type"] = 0;

  int_parameters["mesh_file_type"] = 0;
  int_parameters["gridIsBinary"] = 0;
  real_parameters["gridScale"] = 1.0;
  int_parameters["if_compute_wall_distance"] = 0;
  int_parameters["wall_distance"] = 0;

  bool_parameters["steady"] = true; // Steady simulation is the default choice
  int_parameters["implicit_method"] = 0; // Explicit is used by default
  int_parameters["DPLUR_inner_step"] = 3; // The default DPLUR inner iterations.
  real_parameters["convergence_criteria"] = 1e-8; // The criteria of convergence

  // If we conduct transient simulations, the following parameters are used by default.
  int_parameters["temporal_scheme"] = 3; // RK3 is used by default
  bool_parameters["fixed_time_step"] = false; // The time step is computed with CFL condition.

  // If dual-time stepping is used, the following parameters are needed.
  int_parameters["inner_iteration"] = 20;
  int_parameters["max_inner_iteration"] = 30;
  int_parameters["iteration_adjust_step"] = 20;

  // For spatial discretization
  int_parameters["inviscid_scheme"] = 3; // AUSM+
  int_parameters["reconstruction"] = 2; // MUSCL
  int_parameters["limiter"] = 0; //minmod
  real_parameters["entropy_fix_factor"] = 0.125; // For Roe scheme, we need an entropy fix factor.
  int_parameters["viscous_order"] = 2; // Default viscous order is 2.
  bool_parameters["gradPInDiffusionFlux"] = false;

  // When WENO is used, pp limiter is a choice
  bool_parameters["positive_preserving"] = false;

  int_parameters["species"] = 0;
  int_parameters["reaction"] = 0;
  string_parameters["therm_file"] = "chemistry/therm.dat";
  string_parameters["transport_file"] = "chemistry/tran.dat";
  int_parameters["chemSrcMethod"] = 0; // explicit treatment of the chemical source term is used by default.
  real_parameters["c_chi"] = 1;

  bool_parameters["turbulence"] = false;// laminar
  int_parameters["turbulence_method"] = 1;// 1-RAS, 2-DES
  int_parameters["RANS_model"] = 2; // 1-SA, 2-SST
  int_parameters["turb_implicit"] = 1; // turbulent source term is treated implicitly by default
  int_parameters["compressibility_correction"] = 0; // No compressibility correction is added by default
  int_parameters["des_scale_method"] = 0; // How to compute the grid scale in DES simulations. 0 - cubic root of cell volume, 1 - max of cell dimension

  int_parameters["n_profile"] = 0;

  int_parameters["groups_init"] = 1;
  string_parameters["default_init"] = "freestream";

  int_parameters["diffusivity_method"] = 1;
  real_parameters["schmidt_number"] = 0.5;
  real_parameters["prandtl_number"] = 0.72;
  real_parameters["turbulent_prandtl_number"] = 0.9;
  real_parameters["turbulent_schmidt_number"] = 0.9;

  bool_parameters["if_collect_statistics"] = false;
  bool_parameters["if_continue_collect_statistics"] = false;
  int_parameters["start_collect_statistics_iter"] = 0;
  bool_parameters["perform_spanwise_average"] = false;

  int_array["post_process"] = {};
  int_array["output_bc"] = {};

  int_parameters["if_monitor"] = 0; // 0 - no monitor, 1 - monitor
  string_parameters["monitor_file"] = "input/monitor_points.txt";
  string_array["monitor_var"] = {"density", "u", "v", "w", "pressure", "temperature"};

  int_parameters["n_inflow_fluctuation"] = 0;
  int_array["need_rng"] = {};
  int_array["need_fluctuation_profile"] = {};
  string_array["fluctuation_profile_file"] = {};
  string_array["fluctuation_profile_related_bc_name"] = {};
  bool_parameters["perform_spanwise_average"] = false;
  bool_parameters["positive_preserving"] = false;
}

void cfd::Parameter::diagnose_parallel_info() {
  if (int_parameters["myid"] == 0) {
    fmt::print("{:*^80}\n", "Parallel Information");
    const bool parallel = bool_parameters["parallel"];
    const int n_proc = int_parameters["n_proc"];
    if (parallel) {
      if (n_proc == 1) {
        fmt::print("You chose parallel computation, but the number of processes is equal to 1.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fmt::print("\t->-> Parallel computation chosen! Number of processes: ->-> {}. \n", n_proc);
    } else {
      if (n_proc > 1) {
        fmt::print("You chose serial computation, but the number of processes is not equal to 1, n_proc={}.\n",
                   n_proc);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fmt::print("\t->-> Serial computation chosen!\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void cfd::Parameter::deduce_sim_info() {
  int n_var = 5, n_scalar = 0, n_scalar_transported = 0, n_other_var = 1;
  int i_turb_cv = 5, i_fl_cv = 0;

  if (get_int("species") == 1) {
    n_scalar += get_int("n_spec");
    if (get_int("reaction") != 2) {
      // Mixture / Finite rate chemistry
      n_scalar_transported += get_int("n_spec");
      i_turb_cv += get_int("n_spec");
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
      i_fl_cv = 5 + get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  } else if (get_int("species") == 2) {
    n_scalar += get_int("n_spec");
    if (get_int("reaction") != 2) {
      // Mixture with mixture fraction and variance solved.
      n_scalar_transported += get_int("n_spec") + 2;
      n_scalar += 2;
      i_turb_cv += get_int("n_spec");
      i_fl_cv = i_turb_cv + get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
      i_fl_cv = 5 + get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  }
  if (get_bool("turbulence")) {
    // turbulence simulation
    if (auto turb_method = get_int("turbulence_method");turb_method == 1) {
      // RANS
      n_scalar_transported += get_int("n_turb");
      n_scalar += get_int("n_turb");
      ++n_other_var; // mut
    } else if (turb_method == 2) {
      // DES type
      n_scalar_transported += get_int("n_turb");
      n_scalar += get_int("n_turb");
      ++n_other_var; // mut
    }
  }
  n_var += n_scalar_transported;
  update_parameter("n_var", n_var);
  update_parameter("n_scalar", n_scalar);
  update_parameter("n_scalar_transported", n_scalar_transported);
  update_parameter("i_turb_cv", i_turb_cv);
  update_parameter("i_fl", get_int("n_turb") + get_int("n_spec"));
  update_parameter("i_fl_cv", i_fl_cv);
  update_parameter("n_other_var", n_other_var);

  int myid = get_int("myid");
  if (myid == 0) {
    fmt::print("\n{:*^80}\n", "Simulation Details");
    printf("\t->-> %-20d : number of equations to solve\n", n_var);
    printf("\t->-> %-20d : number of scalar variables\n", n_scalar);

    if (get_bool("steady")) {
      // steady
      fmt::print("\n\t->-> {:<20} : flow simulation will be conducted.\n", "Steady");
      fmt::print("\t\t->-> {:<20} : CFL number\n", get_real("cfl"));
      if (int_parameters["implicit_method"] == 0) {
        // explicit
        fmt::print("\t\t->-> {:<20} : time integration\n", "explicit");
      } else {
        // implicit
        fmt::print("\t\t->-> {:<20} : time integration\n", "implicit DPLUR");
        fmt::print("\t\t->-> {:<20} : inner iterations for DPLUR\n", get_int("DPLUR_inner_step"));
      }
      fmt::print("\t\t->-> {:<20} : convergence criteria\n", get_real("convergence_criteria"));
    } else {
      // transient simulation
      fmt::print("\t->-> {:<20} : flow simulation will be conducted.\n", "Transient");
      if (int_parameters["temporal_scheme"] == 2) {
        // dual-time stepping
        fmt::print("\t\t->-> {:<20} : temporal scheme\n", "Dual-time stepping");
        fmt::print("\t\t->-> {:<20} : inner iterations for dual-time stepping\n", get_int("inner_iteration"));
        fmt::print("\t\t->-> {:<20} : max inner iterations for dual-time stepping\n", get_int("max_inner_iteration"));
        fmt::print("\t\t->-> {:<20} : iteration adjust step\n", get_int("iteration_adjust_step"));
      } else if (int_parameters["temporal_scheme"] == 3) {
        // RK-3
        fmt::print("\t\t->-> {:<25} : temporal scheme\n", "3rd-order Runge-Kutta");
        if (bool_parameters["fixed_time_step"]) {
          fmt::print("\t\t->-> {:<25} : physical time step(s)\n", get_real("dt"));
        } else {
          fmt::print("\t\t->-> {:<25} : CFL number\n", get_real("cfl"));
        }
      }
    }

    fmt::print("\n\t->-> {:<20} : order CDS for viscous flux\n", get_int("viscous_order"));

    if (get_bool("turbulence")) {
      bool need_ras_model{false};
      if (auto method = get_int("turbulence_method");method == 1) {
        fmt::print("\n\t->-> {:<20} : simulation\n", "RAS");
        need_ras_model = true;
      } else if (method == 2) {
        fmt::print("\n\t->-> {:<20} : simulation\n", "DDES");
        need_ras_model = true;
      }
      if (need_ras_model) {
        if (get_int("RANS_model") == 1) {
          fmt::print("\t\t->-> {:<20} : model\n", "SA");
        } else if (get_int("RANS_model") == 2) {
          fmt::print("\t\t->-> {:<20} : model\n", "k-omega SST");
        }
        if (int_parameters["turb_implicit"]) {
          fmt::print("\t\t->-> {:<20} : turbulence source term treatment\n", "implicit");
        }
        std::string cc_method{};
        bool cc_flag = false;
        if (auto cc = get_int("compressibility_correction");cc == 1) {
          cc_flag = true;
          cc_method = "Wilcox";
        } else if (cc == 2) {
          cc_flag = true;
          cc_method = "Sarkar";
        } else if (cc == 3) {
          cc_flag = true;
          cc_method = "Zeman";
        }
        if (cc_flag)
          fmt::print("\t\t->-> {:<20} : compressibility correction\n", cc_method);
      }
    }
  }
}
