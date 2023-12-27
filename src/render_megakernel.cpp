#include "render_megakernel.hpp"

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <fmt/ostream.h>

#include "util.hpp"
#include "trace_ray.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

constexpr sycl::specialization_id<uint32_t> max_depth_spec_id;
constexpr sycl::specialization_id<uint32_t> sample_count_spec_id;

static float3 render_pixel(
    const RenderContext &ctx,
    XorShift32State &rng,
    int2 pixel_coords,
    uint32_t max_depth,
    uint32_t &ray_count
) {
    float3 attenuation = float3(1.0f);
    float3 radiance = float3(0.0f);

    RayData ray_data = ctx.camera.get_ray(pixel_coords, rng);
    for (uint32_t i = 0; i < max_depth; ++i) {
        ray_count++;

        float3 attenuation =
            float3(ray_data.att_r, ray_data.att_g, ray_data.att_b);
        float3 radiance =
            float3(ray_data.rad_r, ray_data.rad_g, ray_data.rad_b);

        auto res = trace_ray(ctx, rng, ray_data, attenuation, radiance);
        ray_data.att_r = attenuation.x();
        ray_data.att_g = attenuation.y();
        ray_data.att_b = attenuation.z();

        ray_data.rad_r = radiance.x();
        ray_data.rad_g = radiance.y();
        ray_data.rad_b = radiance.z();

        if (res) {
            return *res;
        }
    }

    return float3(0, 0, 0);
}

MegakernelRenderer::MegakernelRenderer(
    App &app,
    sycl::range<2> img_size,
    sycl::image<2> &image,
    uint32_t max_depth,
    uint32_t sample_count
)
    : app(app), img_size(img_size), image(image), max_depth(max_depth),
      sample_count(sample_count) {}

void MegakernelRenderer::render_frame(const Camera &camera, const Scene &scene) {
    uint32_t initial_ray_count = 0;
    sycl::buffer<uint32_t> ray_count_buffer{&initial_ray_count, 1};

    auto begin = std::chrono::high_resolution_clock::now();

    auto e = app.queue.submit([&](sycl::handler &cgh) {
        cgh.set_specialization_constant<max_depth_spec_id>(this->max_depth);
        cgh.set_specialization_constant<sample_count_spec_id>(this->sample_count);

        sycl::stream os(8192, 256, cgh);

        auto image_writer = image.get_access<float4, sycl::access::mode::write>(cgh);
        auto ray_count = ray_count_buffer.get_access<sycl::access_mode::write>(cgh);

        range<2> local_size{8, 8};
        range<2> n_groups = img_size;
        n_groups[0] = ((img_size[0] + local_size[0] - 1) / local_size[0]);
        n_groups[1] = ((img_size[1] + local_size[1] - 1) / local_size[1]);

        RenderContext ctx = {
            .camera = camera,
            .sky_color = scene.sky_color,
            .scene = scene.scene,
            .sampler = sycl::sampler(
                sycl::coordinate_normalization_mode::normalized,
                sycl::addressing_mode::repeat,
                sycl::filtering_mode::nearest
            ),
            .image_reader = ImageReadAccessor(scene.image_array.value(), cgh),
            .os = os,
        };

        const auto img_size = this->img_size;

        cgh.parallel_for(
            sycl::nd_range<2>(n_groups * local_size, local_size),
            [=](sycl::nd_item<2> id, sycl::kernel_handler h) {
                auto global_id = id.get_global_id();
                if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                    return;
                }

                uint32_t max_depth = h.get_specialization_constant<max_depth_spec_id>();
                uint32_t sample_count =
                    h.get_specialization_constant<sample_count_spec_id>();

                int2 pixel_coords = {global_id[0], global_id[1]};

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    ray_count_ref(ray_count[0]);

                auto init_generator_state =
                    std::hash<std::size_t>{}(id.get_global_linear_id());
                auto rng = XorShift32State{(uint32_t)init_generator_state};

                uint32_t ray_count = 0;
                float3 pixel_color = float3(0, 0, 0);
                for (uint32_t i = 0; i < sample_count; ++i) {
                    pixel_color +=
                        render_pixel(ctx, rng, pixel_coords, max_depth, ray_count);
                }
                pixel_color /= (float)sample_count;

                pixel_color = linear_to_gamma(pixel_color);

                image_writer.write(pixel_coords, float4(pixel_color, 1.0f));

                ray_count_ref.fetch_add(ray_count);
            }
        );
    });

    e.wait_and_throw();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto ray_count = ray_count_buffer.get_host_access();

    double secs = elapsed.count() * 1e-9;
    double rays_per_sec = (double)ray_count[0] / secs;

    fmt::println("Time measured: {:.6f} seconds", secs);
    fmt::println("Total rays: {}", ray_count[0]);
    fmt::println("Rays/sec: {:.2f}M", rays_per_sec / 1000000.0);

    fmt::println("Writing image to disk");
    write_image(app.queue, image, img_size[0], img_size[1]);
}
