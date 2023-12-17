#pragma once

#include <sycl/sycl.hpp>

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
        return sycl::float3(this->operator()(), this->operator()(),
                      this->operator()());
    }

    inline sycl::float3 vec(float min, float max) {
        return sycl::float3(this->operator()(min, max), this->operator()(min, max),
                      this->operator()(min, max));
    }
};
