#pragma once

#include "render.hpp"
#include "camera.hpp"

namespace raytracer {

static constexpr uint32_t SAMPLES_PER_RUN = 8;
static uint32_t ZERO = 0;

struct RunBuffers {
    std::array<uint8_t, 4> *image_buffer;

    RunBuffers(App &app, sycl::range<2> img_size) {
        this->image_buffer = (std::array<uint8_t, 4> *)sycl::aligned_alloc_device(
            alignof(sycl::float4),
            sizeof(std::array<uint8_t, 4>) * img_size.size(),
            app.queue
        );
    }
};

struct Buffers {
    uint64_t *ray_buffer_length;
    uint32_t *ray_ids;
    sycl::float3 *ray_origins;
    sycl::half3 *ray_directions;
    sycl::half3 *ray_attenuations;
    sycl::half3 *ray_radiances;

    Buffers(App &app, sycl::range<2> img_size) {
        size_t buf_size = img_size.size() * SAMPLES_PER_RUN;
        this->ray_buffer_length = (uint64_t *)sycl::aligned_alloc_shared(
            alignof(uint64_t), sizeof(uint64_t), app.queue
        );
        this->ray_ids = (uint32_t *)sycl::aligned_alloc_device(
            alignof(uint32_t), sizeof(uint32_t) * buf_size, app.queue
        );
        this->ray_origins = (sycl::float3 *)sycl::aligned_alloc_device(
            alignof(sycl::float3), sizeof(sycl::float3) * buf_size, app.queue
        );
        this->ray_directions = (sycl::half3 *)sycl::aligned_alloc_device(
            alignof(sycl::half3), sizeof(sycl::half3) * buf_size, app.queue
        );
        this->ray_attenuations = (sycl::half3 *)sycl::aligned_alloc_device(
            alignof(sycl::half3), sizeof(sycl::half3) * buf_size, app.queue
        );
        this->ray_radiances = (sycl::half3 *)sycl::aligned_alloc_device(
            alignof(sycl::half3), sizeof(sycl::half3) * buf_size, app.queue
        );
    }
};

struct WavefrontRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> combined_image;
    sycl::image<2> &output_image;

    uint32_t buffer_index = 0;
    std::array<Buffers, 2> buffers;
    RunBuffers* run_buffers;

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
    void generate_camera_rays(const Camera &camera);
    void shoot_rays(const Camera &camera, const Scene &scene, uint32_t depth);
    void merge_samples();
    void convert_image_to_srgb();
};
} // namespace raytracer
