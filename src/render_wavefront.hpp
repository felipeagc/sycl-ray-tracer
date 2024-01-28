#pragma once

#include "render.hpp"
#include "camera.hpp"

namespace raytracer {

static constexpr uint32_t SAMPLES_PER_RUN = 8;
static uint32_t ZERO = 0;

struct ColorBuffer {
  private:
    sycl::range<2> img_size;
    std::array<std::array<uint8_t, 4>, SAMPLES_PER_RUN> *const color_buffer = nullptr;

  public:
    ColorBuffer(App &app, sycl::range<2> img_size)
        : img_size(img_size),
          color_buffer(
              (std::array<std::array<uint8_t, 4>, SAMPLES_PER_RUN> *)
                  sycl::aligned_alloc_device(
                      alignof(sycl::float4),
                      sizeof(std::array<std::array<uint8_t, 4>, SAMPLES_PER_RUN>) *
                          img_size.size(),
                      app.queue
                  )
          ) {}

    inline void
    write(uint32_t run_index, sycl::int2 pixel_coords, sycl::float4 value) const {
        uint32_t pixel_linear_pos = pixel_coords[0] + (pixel_coords[1] * img_size[0]);

        std::array<uint8_t, 4> val = {
            (uint8_t)(value.r() * 255.0f),
            (uint8_t)(value.g() * 255.0f),
            (uint8_t)(value.b() * 255.0f),
            (uint8_t)(value.a() * 255.0f),
        };
        this->color_buffer[pixel_linear_pos][run_index] = val;
    }

    inline sycl::float4 read(uint32_t run_index, sycl::int2 pixel_coords) const {
        uint32_t pixel_linear_pos = pixel_coords[0] + (pixel_coords[1] * img_size[0]);
        std::array<uint8_t, 4> val = this->color_buffer[pixel_linear_pos][run_index];
        return sycl::float4(
            float(val[0]) / 255.0f,
            float(val[1]) / 255.0f,
            float(val[2]) / 255.0f,
            float(val[3]) / 255.0f
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
    ColorBuffer color_buffer;

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
