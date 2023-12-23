#pragma once

#include <sycl/sycl.hpp>
#include "util.hpp"

namespace raytracer {

struct XorShift32State {
    uint32_t a = 2463534242;

    inline float operator()() {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        uint32_t x = this->a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        this->a = x;
        constexpr float scale = 1.f / (uint64_t{1} << 32);
        return this->a * scale;
    }

    inline float operator()(float min, float max) {
        return min + (max - min) * this->operator()();
    }

    inline sycl::float3 vec() {
        return sycl::float3(this->operator()(), this->operator()(), this->operator()());
    }

    inline sycl::float3 vec(float min, float max) {
        return sycl::float3(
            this->operator()(min, max),
            this->operator()(min, max),
            this->operator()(min, max)
        );
    }

    inline sycl::float3 random_unit_vector() {
        return unit_vector(this->vec(-1, 1));
    }

    inline sycl::float3 random_on_hemisphere(const sycl::float3 &normal) {
        sycl::float3 on_unit_sphere = this->random_unit_vector();
        if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
            return on_unit_sphere;
        else
            return -on_unit_sphere;
    }
};

} // namespace raytracer
