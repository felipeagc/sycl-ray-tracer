#pragma once

#include "render.hpp"

namespace raytracer {

static uint32_t ZERO = 0;

struct RayData {
    RTCRay ray;
    sycl::float3 attenuation;
    sycl::float3 radiance;
    XorShift32State rng;
};

struct Buffers {
    sycl::buffer<uint32_t> ray_buffer_length;
    RayData *ray_buffer;
    sycl::image<2> image;

    Buffers(App &app, sycl::range<2> img_size)
        : image(
              sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size
          ),
          ray_buffer_length(&ZERO, 1) {
        this->ray_buffer = (RayData *)sycl::aligned_alloc_device(
            alignof(RayData), sizeof(RayData) * img_size.size(), app.queue
        );

        app.queue
            .submit([&](sycl::handler &cgh) {
                auto image_writer =
                    this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);
                cgh.parallel_for(sycl::range<2>(img_size), [=](sycl::item<2> item) {
                    image_writer.write(sycl::int2(item[0], item[1]), sycl::float4(0.0f));
                });
            })
            .wait();
    }
};

struct WavefrontRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> &output_image;

    uint32_t buffer_index = 0;
    std::array<Buffers, 2> buffers;

    WavefrontRenderer(App &app, sycl::range<2> img_size, sycl::image<2> &output_image);

    virtual void render_frame(const Camera &camera, const Scene &scene) override;

    inline Buffers &current_buffer() {
        return buffers[this->buffer_index & 1];
    }
    inline Buffers &prev_buffer() {
        return buffers[~this->buffer_index & 1];
    }

  private:
    void generate_camera_rays(const Camera &camera);
    void shoot_rays(const Camera &camera, const Scene &scene, uint32_t depth);
    void convert_image_to_srgb();
};
} // namespace raytracer
