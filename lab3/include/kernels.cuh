#pragma once

#include <cstdint>

#include "julia.cuh"

namespace lab3 {

void launch_make_grid(float2* grid, const JuliaParams& p, dim3 block);
void launch_julia_from_grid(const float2* grid, std::uint16_t* out, const JuliaParams& p,
                            dim3 block);
void launch_julia_fused(std::uint16_t* out, const JuliaParams& p, dim3 block);

void julia_reference(const JuliaParams& p, std::uint16_t* out);

}
