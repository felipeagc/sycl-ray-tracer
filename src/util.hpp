#pragma once

#include <sycl/sycl.hpp>
#include "stb_image_write.h"

namespace raytracer {

static void
write_image(sycl::queue &q, sycl::image<2> &image, size_t width, size_t height) {
    uint8_t *transfer_buf = sycl::malloc_shared<uint8_t>(width * height * 4, q);
    sycl::range<2> img_size = sycl::range<2>(width, height);

    q.submit([&](sycl::handler &cgh) {
        auto read_accessor =
            image.get_access<sycl::float4, sycl::access::mode::read>(cgh);
        cgh.parallel_for(img_size, [=](sycl::id<2> id) {
            sycl::float4 rgba = read_accessor.read(sycl::int2(id[0], id[1]));
            size_t i = id[1] * width + id[0];
            transfer_buf[i * 4] = (rgba[0] * 255.0f);
            transfer_buf[i * 4 + 1] = (rgba[1] * 255.0f);
            transfer_buf[i * 4 + 2] = (rgba[2] * 255.0f);
            transfer_buf[i * 4 + 3] = (rgba[3] * 255.0f);
        });
    });
    q.wait_and_throw();

    if (!stbi_write_png("out.png", width, height, 4, transfer_buf, width * 4)) {
        std::cout << "Failed to write image to disk." << std::endl;
        std::terminate();
    }

    sycl::free(transfer_buf, q);
}

/*
 * This function allocated USM memory that is writeable by the device.
 */

template <typename T>
T *alignedSYCLMallocDeviceReadWrite(
    const sycl::queue &queue, size_t count, size_t align
) {
    if (count == 0) return nullptr;

    assert((align & (align - 1)) == 0);
    T *ptr = (T *)sycl::aligned_alloc(
        align, count * sizeof(T), queue, sycl::usm::alloc::shared
    );
    if (count != 0 && ptr == nullptr) throw std::bad_alloc();

    return ptr;
}

/*
 * This function allocated USM memory that is only readable by the
 * device. Using this mode many small allocations are possible by the
 * application.
 */

template <typename T>
T *alignedSYCLMallocDeviceReadOnly(const sycl::queue &queue, size_t count, size_t align) {
    if (count == 0) return nullptr;

    assert((align & (align - 1)) == 0);
    T *ptr = (T *)sycl::aligned_alloc_shared(
        align,
        count * sizeof(T),
        queue,
        sycl::ext::oneapi::property::usm::device_read_only()
    );
    if (count != 0 && ptr == nullptr) throw std::bad_alloc();

    return ptr;
}

inline void alignedSYCLFree(const sycl::queue &queue, void *ptr) {
    if (ptr) sycl::free(ptr, queue);
}

// Vector utilities

inline sycl::float3 unit_vector(sycl::float3 v) {
    return v / sycl::length(v);
}

inline float linear_to_gamma(float linear_component) {
    return sycl::sqrt(linear_component);
}

inline sycl::float3 linear_to_gamma(sycl::float3 linear_component) {
    return sycl::float3(
        linear_to_gamma(linear_component[0]),
        linear_to_gamma(linear_component[1]),
        linear_to_gamma(linear_component[2])
    );
}

inline sycl::float4 linear_to_gamma(sycl::float4 linear_component) {
    return sycl::float4(
        linear_to_gamma(linear_component[0]),
        linear_to_gamma(linear_component[1]),
        linear_to_gamma(linear_component[2]),
        linear_component[3]
    );
}

inline bool near_zero(sycl::float3 e) {
    // Return true if the vector is close to zero in all dimensions.
    float s = 1e-8f;
    return (sycl::fabs(e[0]) < s) && (sycl::fabs(e[1]) < s) && (sycl::fabs(e[2]) < s);
}

} // namespace raytracer
