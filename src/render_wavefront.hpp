#pragma once

#include "render.hpp"
#include "camera.hpp"

namespace raytracer {

static uint32_t ZERO = 0;

struct Buffers {
    sycl::buffer<uint32_t> ray_buffer_length;
    RayData *ray_buffer;

    Buffers(App &app, sycl::range<2> img_size) : ray_buffer_length(&ZERO, 1) {
        this->ray_buffer = (RayData *)sycl::aligned_alloc_device(
            alignof(RayData), sizeof(RayData) * img_size.size(), app.queue
        );
    }
};

struct WavefrontRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> image;
    sycl::image<2> combined_image;
    sycl::image<2> &output_image;

    uint32_t buffer_index = 0;
    std::array<Buffers, 2> buffers;

    XorShift32State *rng_buffer;

    const uint32_t max_depth;
    const uint32_t sample_count;

    WavefrontRenderer(
        App &app,
        sycl::range<2> img_size,
        sycl::image<2> &output_image,
        uint32_t max_depth,
        uint32_t sample_count
    );

    virtual void render_frame(const Camera &camera, const Scene &scene) override;

    inline Buffers &current_buffer() {
        return buffers[this->buffer_index & 1];
    }
    inline Buffers &prev_buffer() {
        return buffers[~this->buffer_index & 1];
    }

  private:
    void generate_camera_rays(const Camera &camera, uint32_t sample);
    void shoot_rays(const Camera &camera, const Scene &scene, uint32_t depth);
    void merge_samples(uint32_t sample);
    void convert_image_to_srgb();
};
} // namespace raytracer