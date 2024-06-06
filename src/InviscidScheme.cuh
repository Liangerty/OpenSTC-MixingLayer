#pragma once

#include "Define.h"

namespace cfd {
class Block;

struct DZone;
struct DParameter;

class Parameter;

template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param);

template<MixtureModel mix_model>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, int idx_shared, DParameter *param);

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, cfd::DZone *zone, DParameter *param, int n_var);

template<MixtureModel mix_model>
void compute_convective_term_weno(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                                  const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_aweno_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param);

template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, int n_var,
                               const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
compute_entropy_fix_delta(cfd::DZone *zone, DParameter *param);

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_weno_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param);

template<MixtureModel mix_model>
__device__ void
compute_flux(const real *Q, DParameter *param, const real *metric, real jac, real *Fp, real *Fm);

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_ch(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, [[maybe_unused]] real *f_1st);

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_cp(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st);

template<MixtureModel mix_model>
__device__ void
positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
                            real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac);

__device__ real WENO5(const real *vp, const real *vm, real eps);
__device__ real WENO7(const real *vp, const real *vm, real eps);

template<MixtureModel mix_model>
void compute_convective_term_ep(const Block &block, cfd::DZone *zone, DParameter *param, int n_var);

template<MixtureModel mix_model>
__global__ void
compute_convective_term_ep_1D(cfd::DZone *zone, int direction, int max_extent, DParameter *param);
} // cfd
