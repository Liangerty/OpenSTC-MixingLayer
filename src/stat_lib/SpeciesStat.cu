#include "SpeciesStat.cuh"
#include <mpi.h>
#include "../DParameter.cuh"

__device__ void cfd::collect_species_dissipation_rate(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k) {
  auto &collect = zone->collect_spec_diss_rate;
  const auto &sv = zone->sv;
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  for (int i_s = 0; i_s < param->n_species_stat; ++i_s) {
    int is = param->specStatIndex[i_s];
    int l = i_s * SpeciesDissipationRate::n_collect;

    real zx = 0.5 * (xi_x * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_x * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_x * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    real zy = 0.5 * (xi_y * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_y * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_y * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    real zz = 0.5 * (xi_z * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_z * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_z * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    auto rhoD = zone->rho_D(i, j, k, is);

    // Rho*D*GradH2*GradH2
    collect(i, j, k, l) += rhoD * (zx * zx + zy * zy + zz * zz);
    // Rho*D*H2x
    collect(i, j, k, l + 1) += rhoD * zx;
    // Rho*D*H2y
    collect(i, j, k, l + 2) += rhoD * zy;
    // Rho*D*H2z
    collect(i, j, k, l + 3) += rhoD * zz;
    // Rho*D
    collect(i, j, k, l + 4) += rhoD;
  }
  for (int i_s = 0; i_s < param->n_ps; ++i_s) {
    int is = param->i_ps + i_s;
    int l = (param->n_species_stat + i_s) * SpeciesDissipationRate::n_collect;

    real zx = 0.5 * (xi_x * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_x * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_x * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    real zy = 0.5 * (xi_y * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_y * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_y * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    real zz = 0.5 * (xi_z * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                     eta_z * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                     zeta_z * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    auto rhoD = zone->mul(i, j, k) / param->sc_ps[i_s];

    // Rho*D*GradH2*GradH2
    collect(i, j, k, l) += rhoD * (zx * zx + zy * zy + zz * zz);
    // Rho*D*H2x
    collect(i, j, k, l + 1) += rhoD * zx;
    // Rho*D*H2y
    collect(i, j, k, l + 2) += rhoD * zy;
    // Rho*D*H2z
    collect(i, j, k, l + 3) += rhoD * zz;
    // Rho*D
    collect(i, j, k, l + 4) += rhoD;
  }
}

__device__ void
cfd::collect_species_velocity_correlation(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k) {
  auto &collect = zone->collect_spec_vel_correlation;
  const auto &sv = zone->sv;
  const auto &bv = zone->bv;

  auto rho = bv(i, j, k, 0), u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3);
  for (int i_s = 0; i_s < param->n_species_stat; ++i_s) {
    int is = param->specStatIndex[i_s];
    int l = i_s * SpeciesVelocityCorrelation::n_collect;

    real z = sv(i, j, k, is);
    collect(i, j, k, l) += rho * u * z;
    collect(i, j, k, l + 1) += rho * v * z;
    collect(i, j, k, l + 2) += rho * w * z;
  }
  for (int i_s = 0; i_s < param->n_ps; ++i_s) {
    int is = param->i_ps + i_s;
    int l = (param->n_species_stat + i_s) * SpeciesVelocityCorrelation::n_collect;

    real z = sv(i, j, k, is);
    collect(i, j, k, l) += rho * u * z;
    collect(i, j, k, l + 1) += rho * v * z;
    collect(i, j, k, l + 2) += rho * w * z;
  }
}

void cfd::SpeciesDissipationRate::read(MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count,
                                       MPI_Datatype ty, MPI_Status *status) {
  MPI_File_read_at(fp, offset_read, zone.collect_spec_diss_rate[index], count, ty, status);
}

void cfd::SpeciesDissipationRate::copy_to_device(cfd::Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.h_ptr->collect_spec_diss_rate.data(), zone.collect_spec_diss_rate.data(), sz * nv,
             cudaMemcpyHostToDevice);
}

void cfd::SpeciesDissipationRate::copy_to_host(cfd::Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.collect_spec_diss_rate.data(), zone.h_ptr->collect_spec_diss_rate.data(), sz * nv,
             cudaMemcpyDeviceToHost);
}

void cfd::SpeciesDissipationRate::write(MPI_File &fp, MPI_Offset offset, cfd::Field &zone, int count, MPI_Datatype ty,
                                        MPI_Status *status) {
  MPI_File_write_at(fp, offset, zone.collect_spec_diss_rate.data(), count, ty, status);
}

void cfd::SpeciesVelocityCorrelation::read(MPI_File &fp, MPI_Offset offset_read, cfd::Field &zone, int index, int count,
                                           MPI_Datatype ty, MPI_Status *status) {
  MPI_File_read_at(fp, offset_read, zone.collect_spec_vel_correlation[index], count, ty, status);
}

void cfd::SpeciesVelocityCorrelation::copy_to_device(cfd::Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.h_ptr->collect_spec_vel_correlation.data(), zone.collect_spec_vel_correlation.data(), sz * nv,
             cudaMemcpyHostToDevice);
}

void cfd::SpeciesVelocityCorrelation::copy_to_host(cfd::Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.collect_spec_vel_correlation.data(), zone.h_ptr->collect_spec_vel_correlation.data(), sz * nv,
             cudaMemcpyDeviceToHost);
}

void
cfd::SpeciesVelocityCorrelation::write(MPI_File &fp, MPI_Offset offset, cfd::Field &zone, int count, MPI_Datatype ty,
                                       MPI_Status *status) {
  MPI_File_write_at(fp, offset, zone.collect_spec_vel_correlation.data(), count, ty, status);
}
