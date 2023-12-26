#pragma once

#include "render.hpp"

namespace raytracer {
struct WavefrontRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> &image;

    sycl::buffer<uint32_t> total_ray_count_buffer;

    sycl::buffer<uint32_t> ray_buffer_length;
    RTCRay *ray_buffer;

    WavefrontRenderer(App &app, sycl::range<2> img_size, sycl::image<2> &image);

    virtual void render_frame(const Camera &camera, const Scene &scene) override;

  private:
    void generate_camera_rays(const Camera &camera);
    void shoot_rays(const Camera &camera, const Scene &scene);
    void convert_image_to_srgb();
};
} // namespace raytracer
