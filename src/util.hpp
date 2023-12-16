#pragma once

#include <sycl/sycl.hpp>
#include "stb_image_write.h"
#include "tiny_gltf.h"

namespace raytracer {

static void write_image(sycl::queue& q, sycl::image<2>& image, size_t width,
                        size_t height) {
    uint8_t* transfer_buf = sycl::malloc_shared<uint8_t>(width * height * 4, q);
    sycl::range<2> img_size = sycl::range<2>(width, height);

    q.submit([&](sycl::handler& cgh) {
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

}  // namespace raytracer
