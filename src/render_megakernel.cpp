#include "render_megakernel.hpp"

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <fmt/ostream.h>

#include "util.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

static float3 render_pixel(
    const RenderContext &ctx, XorShift32State &rng, int2 pixel_coords, uint32_t &ray_count
) {
    float3 attenuation = float3(1.0f);
    float3 radiance = float3(0.0f);

    constexpr uint32_t max_bounces = 50;

    RTCRayHit rayhit;
    rayhit.ray = ctx.camera.get_ray(pixel_coords, rng);
    for (uint32_t i = 0; i < max_bounces; ++i) {
        ray_count++;

        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(ctx.scene, &rayhit);

        // If not hit, return sky color
        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            return attenuation * (ctx.sky_color + radiance);
        }

        GeometryData *user_data = (GeometryData *)rtcGetGeometryUserDataFromScene(
            ctx.scene, rayhit.hit.instID[0]
        );

        glm::vec2 bary = {rayhit.hit.u, rayhit.hit.v};

        const uint32_t *prim_indices = &user_data->index_buffer[rayhit.hit.primID * 3];
        std::array<glm::vec3, 3> vertex_normals = {
            user_data->normal_buffer[prim_indices[0]],
            user_data->normal_buffer[prim_indices[1]],
            user_data->normal_buffer[prim_indices[2]],
        };

        std::array<float2, 3> vertex_uvs = {
            user_data->uv_buffer[prim_indices[0]],
            user_data->uv_buffer[prim_indices[1]],
            user_data->uv_buffer[prim_indices[2]],
        };

        // Calculate UVs
        float2 vertex_uv = (1 - bary.x - bary.y) * vertex_uvs[0] +
                           bary.x * vertex_uvs[1] + bary.y * vertex_uvs[2];

        // Calculate normals
        glm::vec3 vertex_normal = glm::normalize(
            (1 - bary.x - bary.y) * vertex_normals[0] + bary.x * vertex_normals[1] +
            bary.y * vertex_normals[2]
        );

        glm::vec3 g_normal = user_data->obj_to_world * vertex_normal;
        const float3 normal = normalize(float3(g_normal.x, g_normal.y, g_normal.z));

        const float3 dir =
            normalize(float3(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z));

        radiance += user_data->material.emitted();

        ScatterResult result;
        if (user_data->material.scatter(ctx, rng, dir, normal, vertex_uv, result)) {
            rayhit.ray.org_x = rayhit.ray.org_x + rayhit.ray.dir_x * rayhit.ray.tfar;
            rayhit.ray.org_z = rayhit.ray.org_z + rayhit.ray.dir_z * rayhit.ray.tfar;
            rayhit.ray.org_y = rayhit.ray.org_y + rayhit.ray.dir_y * rayhit.ray.tfar;

            rayhit.ray.tnear = 0.0001f;
            rayhit.ray.tfar = std::numeric_limits<float>::infinity();

            rayhit.ray.dir_x = result.dir.x();
            rayhit.ray.dir_y = result.dir.y();
            rayhit.ray.dir_z = result.dir.z();

            attenuation = attenuation * result.attenuation;
        } else {
            return attenuation * radiance;
        }
    }

    return float3(0, 0, 0);
}

MegakernelRenderer::MegakernelRenderer(App &app) : app(app) {}

void MegakernelRenderer::render_frame(
    const Camera &camera, const Scene &scene, range<2> img_size, sycl::image<2> &image
) {
    uint32_t initial_ray_count = 0;
    sycl::buffer<uint32_t> ray_count_buffer{&initial_ray_count, 1};

    auto begin = std::chrono::high_resolution_clock::now();

    auto e = app.queue.submit([&](sycl::handler &cgh) {
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

        cgh.parallel_for(
            sycl::nd_range<2>(n_groups * local_size, local_size),
            [=](sycl::nd_item<2> id) {
                auto global_id = id.get_global_id();
                if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                    return;
                }

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

                constexpr uint32_t sample_count = 64;

                uint32_t ray_count = 0;
                float3 pixel_color = float3(0, 0, 0);
                for (uint32_t i = 0; i < sample_count; ++i) {
                    pixel_color += render_pixel(ctx, rng, pixel_coords, ray_count);
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
