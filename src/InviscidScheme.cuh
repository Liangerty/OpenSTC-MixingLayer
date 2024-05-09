#pragma once

#include "Define.h"

namespace cfd {
class Block;

struct DZone;
struct DParameter;

class Parameter;

template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                   const Parameter &parameter);

template<MixtureModel mix_model>
void compute_convective_term_weno(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                  const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_aweno_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_weno_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
__device__ void
compute_lf_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                const real *jac, real *fc, integer i_shared);

template<MixtureModel mix_model>
__device__ void
compute_ausmPlus_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                      const real *jac, real *fc, integer i_shared);

template<MixtureModel mix_model>
__device__ void
compute_hllc_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                  const real *jac, real *fc, integer i_shared);

template<MixtureModel mix_model>
__device__ void
compute_flux(const real *Q, DParameter *param, const real *metric, real jac, real *Fp, real *Fm);

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_ch(const real *cv, DParameter *param, integer tid, const real *metric, const real *jac, real *fc,
                     integer i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, [[maybe_unused]] real *f_1st);

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_cp(const real *cv, DParameter *param, integer tid, const real *metric, const real *jac, real *fc,
                     integer i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st);

template<MixtureModel mix_model>
__device__ void
positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
                            real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac);

} // cfd
