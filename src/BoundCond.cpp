#include "BoundCond.h"
#include "Transport.cuh"
#include "gxl_lib/MyString.h"
#include <cmath>
#include "Parallel.h"
#include "MixingLayer.cuh"
#include "MixtureFraction.h"

cfd::Inflow::Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
  if (info.find("inflow_type") != info.end()) inflow_type = std::get<integer>(info.at("inflow_type"));
  if (parameter.get_int("problem_type") == 1)
    inflow_type = 2;
  if (parameter.get_int("problem_type") == 0 && inflow_type == 2)
    inflow_type = 0;
  if (info.find("fluctuation_type") != info.end()) fluctuation_type = std::get<integer>(info.at("fluctuation_type"));

  const integer n_scalar = parameter.get_int("n_scalar");
  const integer n_spec{spec.n_spec};
  real gamma{gamma_air};
  real c{-1};
  if (inflow_type == 2) {
    // Mixing-layer problem.
    // The info should be treated differently from general cases.
    delta_omega = parameter.get_real("delta_omega");

    // Careful: This kind of inflow is only for multi-component, where turbulent part is not included.
    std::vector<real> var_info;
    get_mixing_layer_info(parameter, spec, var_info);

    density = var_info[0];
    u = var_info[1];
    v = var_info[2];
    w = var_info[3];
    velocity = std::sqrt(u * u + v * v + w * w);
    pressure = var_info[4];
    temperature = var_info[5];
    sv = new real[n_spec];
    mw = 0;
    for (int i = 0; i < n_spec; ++i) {
      sv[i] = var_info[6 + i];
      mw += var_info[6 + i] / spec.mw[i];
    }
    mw = 1 / mw;
    if (n_spec > 0) {
      viscosity = compute_viscosity(temperature, mw, sv, spec);
    } else {
      viscosity = Sutherland(temperature);
    }
    reynolds_number = density * velocity / viscosity;
    if (n_spec > 0) {
      std::vector<real> cpi(n_spec, 0);
      spec.compute_cp(temperature, cpi.data());
      real cp{0}, cv{0};
      for (size_t i = 0; i < n_spec; ++i) {
        cp += sv[i] * cpi[i];
        cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
      }
      gamma = cp / cv;  // specific heat ratio
    }
    c = std::sqrt(gamma * R_u / mw * temperature);  // speed of sound

    density_lower = var_info[7 + n_spec];
    u_lower = var_info[8 + n_spec];
    v_lower = var_info[9 + n_spec];
    w_lower = var_info[10 + n_spec];
    p_lower = var_info[11 + n_spec];
    t_lower = var_info[12 + n_spec];
    sv_lower = new real[n_spec];
    for (int i = 0; i < n_spec; ++i) {
      sv_lower[i] = var_info[13 + n_spec + i];
    }

    if (n_spec > 0) {
      mixture_fraction = var_info[6 + n_spec];
      mixture_fraction_lower = var_info[13 + n_spec + n_spec];
      if (mixture_fraction < 1e-10) {
        // the upper stream is oxidizer
        acquire_mixture_fraction_expression(spec, sv_lower, sv, parameter);
      } else {
        // the upper stream is fuel
        acquire_mixture_fraction_expression(spec, sv, sv_lower, parameter);
      }
    }
  } else {
    // In default, the mach number, pressure and temperature should be given.
    // If other combinations are given, then implement it later.
    // Currently, 2 combinations are achieved. One is to give (mach, pressure,
    // temperature) The other is to give (density, velocity, pressure)
    if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
    if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
    if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
    if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
    if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
    if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
    if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
    if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));

    sv = new real[n_scalar];
    for (int i = 0; i < n_scalar; ++i) {
      sv[i] = 0;
    }

    if (n_spec > 0) {
      // Assign the species mass fraction to the corresponding position.
      // Should be done after knowing the order of species.
      for (auto [name, idx]: spec.spec_list) {
        if (info.find(name) != info.cend()) {
          sv[idx] = std::get<real>(info.at(name));
        }
      }
      mw = 0;
      for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
      mw = 1 / mw;
    }

    if (temperature < 0) {
      // The temperature is not given, thus the density and pressure are given
      temperature = pressure * mw / (density * R_u);
    }
    if (n_spec > 0) {
      viscosity = compute_viscosity(temperature, mw, sv, spec);
    } else {
      viscosity = Sutherland(temperature);
    }

    if (n_spec > 0) {
      std::vector<real> cpi(n_spec, 0);
      spec.compute_cp(temperature, cpi.data());
      real cp{0}, cv{0};
      for (size_t i = 0; i < n_spec; ++i) {
        cp += sv[i] * cpi[i];
        cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
      }
      gamma = cp / cv;  // specific heat ratio
    }

    c = std::sqrt(gamma * R_u / mw * temperature);  // speed of sound

    if (mach < 0) {
      // The mach number is not given. The velocity magnitude should be given
      mach = velocity / c;
    } else {
      // The velocity magnitude is not given. The mach number should be given
      velocity = mach * c;
    }
    u *= velocity;
    v *= velocity;
    w *= velocity;
    if (density < 0) {
      // The density is not given, compute it from equation of state
      density = pressure * mw / (R_u * temperature);
    }
    reynolds_number = density * velocity / viscosity;

    if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
      // RANS or DES simulation
      if (parameter.get_int("RANS_model") == 2) {
        // SST
        mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
        if (info.find("turbulence_intensity") != info.end()) {
          // For SST model, we need k and omega. If SA, we compute this for nothing.
          const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
          sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
          sv[n_spec + 1] = density * sv[n_spec] / mut;
        }
      }
    }

    if ((n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2) {
      // flamelet model or z and z prime are transported
      if (parameter.get_int("turbulence_method") == 1) {
        // RANS simulation
        const auto i_fl{parameter.get_int("i_fl")};
        sv[i_fl] = std::get<real>(info.at("mixture_fraction"));
        sv[i_fl + 1] = 0;
      }
    }

    if (const integer n_profile = parameter.get_int("n_profile"); n_profile > 0) {
      // Instead of reading the profiles here, we store the info about all profiles temporarily.
      // We will read these profiles until we initialize the Profile struct.

      auto profile_related_bc_names = parameter.get_string_array("profile_related_bc_names");
      for (int i = 0; i < profile_related_bc_names.size(); ++i) {
        if (profile_related_bc_names[i] == inflow_name) {
          inflow_type = 1;
          profile_idx = i;
          break;
        }
      }
    }
  }

  // This should be re-considered later
  if (inflow_name == parameter.get_string("reference_state")) {
    parameter.update_parameter("rho_inf", density);
    parameter.update_parameter("v_inf", velocity);
    if (abs(velocity) < 1) {
      parameter.update_parameter("v_inf", c);
    }
    parameter.update_parameter("p_inf", pressure);
    parameter.update_parameter("T_inf", temperature);
    parameter.update_parameter("M_inf", mach);
    parameter.update_parameter("Re_unit", reynolds_number);
    parameter.update_parameter("mu_inf", viscosity);
    parameter.update_parameter("speed_of_sound", c);
    parameter.update_parameter("specific_heat_ratio_inf", gamma);
    std::vector<real> sv_inf(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      sv_inf[l] = sv[l];
    }
    parameter.update_parameter("sv_inf", sv_inf);
  }

  if (fluctuation_type != 0) {
    // Last, read the fluctuation info.
    if (fluctuation_type == 1) {
      if (parameter.has_int_array("need_rng")) {
        auto need_rng = parameter.get_int_array("need_rng");
        if (std::find(need_rng.begin(), need_rng.end(), label) == need_rng.end()) {
          need_rng.push_back(label);
          parameter.update_parameter("need_rng", need_rng);
        }
      } else {
        parameter.update_parameter("need_rng", std::vector<integer>{label});
      }
      if (info.find("fluctuation_intensity") != info.end()) {
        fluctuation_intensity = std::get<real>(info.at("fluctuation_intensity"));
      }
    } else if (fluctuation_type == 2) {
      if (info.find("fluctuation_frequency") != info.end()) fluctuation_frequency = std::get<real>(info.at("fluctuation_frequency"));
      if (info.find("fluctuation_intensity") != info.end()) fluctuation_intensity = std::get<real>(info.at("fluctuation_intensity"));
      if (info.find("streamwise_wavelength") != info.end()) streamwise_wavelength = std::get<real>(info.at("streamwise_wavelength"));
      if (info.find("spanwise_wavelength") != info.end()) spanwise_wavelength = std::get<real>(info.at("spanwise_wavelength"));

      // The fluctuation is given by the profile with real and imaginary parts acquired by stability analysis.
      // Only perfect gas is supported for now.
      if (n_spec > 0) {
        printf("Fluctuation with profile is not supported for multi-species simulation.\n");
        MpiParallel::exit();
      }
      if (parameter.has_int_array("need_fluctuation_profile")) {
        auto need_fluctuation_profile = parameter.get_int_array("need_fluctuation_profile");
        if (std::find(need_fluctuation_profile.begin(), need_fluctuation_profile.end(), label) ==
            need_fluctuation_profile.end()) {
          need_fluctuation_profile.push_back(label);
          parameter.update_parameter("need_fluctuation_profile", need_fluctuation_profile);
          auto fluctuation_file = parameter.get_string_array("fluctuation_profile_file");
          fluctuation_file.push_back(std::get<std::string>(info.at("fluctuation_file")));
          parameter.update_parameter("fluctuation_profile_file", fluctuation_file);
          auto fluctuation_profile_related_bc_name = parameter.get_string_array("fluctuation_profile_related_bc_name");
          fluctuation_profile_related_bc_name.push_back(inflow_name);
          parameter.update_parameter("fluctuation_profile_related_bc_name", fluctuation_profile_related_bc_name);
          fluc_prof_idx = (int) (need_fluctuation_profile.size()) - 1;
        }
      } else {
        parameter.update_parameter("need_fluctuation_profile", std::vector<integer>{label});
        parameter.update_parameter("fluctuation_profile_file",
                                   std::vector<std::string>{std::get<std::string>(info.at("fluctuation_file"))});
        parameter.update_parameter("fluctuation_profile_related_bc_name", std::vector<std::string>{inflow_name});
        fluc_prof_idx = 0;
      }
    }
  }
}

std::tuple<real, real, real, real, real, real> cfd::Inflow::var_info() const {
  return std::make_tuple(density, u, v, w, pressure, temperature);
}

cfd::Inflow::Inflow(const std::string &inflow_name, const cfd::Species &spec, const cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));
  if (info.find("fluctuation_type") != info.end()) fluctuation_type = std::get<integer>(info.at("fluctuation_type"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }
  const integer n_spec{spec.n_spec};

  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name) != info.cend()) {
        sv[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if (n_spec > 0) {
    viscosity = compute_viscosity(temperature, mw, sv, spec);
  } else {
    viscosity = Sutherland(temperature);
  }

  real gamma{gamma_air};
  if (n_spec > 0) {
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += sv[i] * cpi[i];
      cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
    }
    gamma = cp / cv;  // specific heat ratio
  }

  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / c;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * c;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (density < 0) {
    // The density is not given, compute it from the equation of state
    density = pressure * mw / (R_u * temperature);
  }
  reynolds_number = density * velocity / viscosity;

  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS or DES simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = density * sv[n_spec] / mut;
      }
    }
  }

  if ((n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2) {
    // flamelet model or z and z prime are transported
    if (parameter.get_int("turbulence_method") == 1) {
      // RANS simulation
      const auto i_fl{parameter.get_int("i_fl")};
      sv[i_fl] = std::get<real>(info.at("mixture_fraction"));
      sv[i_fl + 1] = 0;
    }
  }
}

//cfd::Wall::Wall(integer type_label, std::ifstream &bc_file) : label(type_label) {
//  std::map<std::string, std::string> opt;
//  std::map<std::string, double> par;
//  std::string input{}, key{}, name{};
//  double val{};
//  std::istringstream line(input);
//  while (gxl::getline_to_stream(bc_file, input, line, gxl::Case::lower)) {
//    line >> key;
//    if (key == "double") {
//      line >> name >> key >> val;
//      par.emplace(std::make_pair(name, val));
//    } else if (key == "option") {
//      line >> name >> key >> key;
//      opt.emplace(std::make_pair(name, key));
//    }
//    if (key == "label" || key == "end") {
//      break;
//    }
//  }
//  if (opt.contains("thermal_type")) {
//    if (opt["thermal_type"] == "isothermal")
//      thermal_type = ThermalType::isothermal;
//    else if (opt["thermal_type"] == "adiabatic")
//      thermal_type = ThermalType::adiabatic;
//    else if (opt["thermal_type"] == "equilibrium_radiation")
//      thermal_type = ThermalType::equilibrium_radiation;
//    else {
//      printf("Unknown thermal type, isothermal is set as default.\n");
//      thermal_type = ThermalType::isothermal;
//    }
//  }
//  if (thermal_type == ThermalType::isothermal) {
//    if (par.contains("temperature")) {
//      temperature = par["temperature"];
//    } else {
//      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
//    }
//  } else if (thermal_type == ThermalType::equilibrium_radiation) {
//    if (par.contains("emissivity")) {
//      emissivity = par["emissivity"];
//    } else {
//      printf("Equilibrium radiation wall does not specify emissivity, is set as 0.8 in default.\n");
//    }
//  }
//}

cfd::Wall::Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info, cfd::Parameter &parameter)
    : label(std::get<integer>(info.at("label"))) {
  if (info.contains("thermal_type")) {
    if (std::get<std::string>(info.at("thermal_type")) == "isothermal")
      thermal_type = ThermalType::isothermal;
    else if (std::get<std::string>(info.at("thermal_type")) == "adiabatic")
      thermal_type = ThermalType::adiabatic;
    else if (std::get<std::string>(info.at("thermal_type")) == "equilibrium_radiation")
      thermal_type = ThermalType::equilibrium_radiation;
    else {
      printf("Unknown thermal type, isothermal is set as default.\n");
      thermal_type = ThermalType::isothermal;
    }
  }
  if (thermal_type == ThermalType::isothermal) {
    if (info.contains("temperature")) {
      temperature = std::get<real>(info.at("temperature"));
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  } else if (thermal_type == ThermalType::equilibrium_radiation) {
    if (info.contains("emissivity")) {
      emissivity = std::get<real>(info.at("emissivity"));
    } else {
      printf("Equilibrium radiation wall does not specify emissivity, is set as 0.8 in default.\n");
    }
    temperature = parameter.get_real("T_inf");
    parameter.update_parameter("if_compute_wall_distance", 1);
  }
}

cfd::Symmetry::Symmetry(const std::string &inflow_name, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
}

cfd::Outflow::Outflow(const std::string &inflow_name, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
}

cfd::FarField::FarField(cfd::Species &spec, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct("farfield");
  label = std::get<integer>(info.at("label"));

  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 3 combinations are achieved.
  // 1. (mach number, pressure, temperature)
  // 2. (density, velocity, pressure)
  // 3. (mach number, temperature, reynolds number)
  if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));
  if (info.find("reynolds_number") != info.end()) reynolds_number = std::get<real>(info.at("reynolds_number"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }
  const integer n_spec{spec.n_spec};
  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name) != info.cend()) {
        sv[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if (n_spec > 0) {
    viscosity = compute_viscosity(temperature, mw, sv, spec);
  } else {
    viscosity = Sutherland(temperature);
  }

//  real gamma{gamma_air};
  if (n_spec > 0) {
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += sv[i] * cpi[i];
      cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
    }
    specific_heat_ratio = cp / cv;  // specific heat ratio
  }

  acoustic_speed = std::sqrt(specific_heat_ratio * R_u / mw * temperature);
//  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / acoustic_speed;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * acoustic_speed;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (pressure < 0) {
    // The pressure is not given, which corresponds to case 3, (Ma, temperature, Re)
    density = viscosity * reynolds_number / velocity;
    pressure = density * temperature * R_u / mw;
  }
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (R_u * temperature);
  }
  entropy = pressure / pow(density, specific_heat_ratio);

  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = density * sv[n_spec] / mut;
      }
    }
  }

  // This should be re-considered later
  if (parameter.get_string("reference_state") == "farfield") {
    parameter.update_parameter("rho_inf", density);
    parameter.update_parameter("v_inf", velocity);
    parameter.update_parameter("p_inf", pressure);
    parameter.update_parameter("T_inf", temperature);
    parameter.update_parameter("M_inf", mach);
    parameter.update_parameter("Re_unit", reynolds_number);
    parameter.update_parameter("mu_inf", viscosity);
    parameter.update_parameter("speed_of_sound", acoustic_speed);
    parameter.update_parameter("specific_heat_ratio_inf", specific_heat_ratio);
    std::vector<real> sv_inf(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      sv_inf[l] = sv[l];
    }
    parameter.update_parameter("sv_inf", sv_inf);
  }
}

cfd::SubsonicInflow::SubsonicInflow(const std::string &inflow_name, cfd::Parameter &parameter) {
  const integer n_spec{parameter.get_int("n_spec")};
  if (n_spec > 0) {
    printf("Subsonic inflow boundary condition does not support multi-species simulation.\n");
    MpiParallel::exit();
  }

  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));

  const real pt_pRef{std::get<real>(info.at("pt_pRef_ratio"))};
  const real Tt_TRef{std::get<real>(info.at("Tt_TRef_ratio"))};
  const real pRef{parameter.get_real("p_inf")};
  const real TRef{parameter.get_real("T_inf")};
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }

  total_pressure = pt_pRef * pRef;
  total_temperature = Tt_TRef * TRef;

  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      const real viscosity{Sutherland(TRef)};
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;

      const real velocity{parameter.get_real("v_inf")};
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = parameter.get_real("rho_inf") * sv[n_spec] / mut;
      }
    }
  }
}

cfd::BackPressure::BackPressure(const std::string &name, cfd::Parameter &parameter) {
  const integer n_spec{parameter.get_int("n_spec")};
  if (n_spec > 0) {
    printf("Back pressure boundary condition does not support multi-species simulation.\n");
    MpiParallel::exit();
  }

  auto &info = parameter.get_struct(name);
  label = std::get<integer>(info.at("label"));

  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (pressure < 0) {
    real p_pRef{1};
    if (info.find("p_pRef_ratio") != info.end()) p_pRef = std::get<real>(info.at("p_pRef_ratio"));
    else {
      printf("Back pressure boundary condition does not specify pressure, is set as 1 in default.\n");
    }
    pressure = p_pRef * parameter.get_real("p_inf");
  }
}

cfd::Periodic::Periodic(const std::string &name, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(name);
  label = std::get<integer>(info.at("label"));
}
