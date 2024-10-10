#pragma once

#include "Define.h"
#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"

namespace cfd{
void initialize_sponge_layer(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field,
                             const Species &spec);

__global__ void update_sponge_layer_value(DZone *zone, DParameter *param);

void update_sponge_iter(DParameter *param, Parameter& parameter);
__global__ void update_sponge_iter_dev(DParameter *param);

__device__ void sponge_layer_source(DZone *zone, int i, int j, int k, DParameter *param);

__device__ real sponge_function(real xi, int method);

void output_sponge_layer(const cfd::Parameter &parameter, const std::vector<Field> &field, const Mesh &mesh,
                         const Species &spec);

void read_sponge_layer(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field,
                       const Species &spec);
}