#include "render.hpp"

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <fmt/ostream.h>

#include "util.hpp"

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

namespace raytracer {

static float4 render_pixel(
    const Camera &camera,
    XorShift32State &rng,
    RTCScene scene,
    int x,
    int y,
    uint32_t &ray_count,
    sycl::stream os
) {
    uint32_t local_ray_count = 0;
    float4 color = float4(1, 1, 1, 1);

    constexpr uint32_t max_bounces = 10;

    RTCRayHit rayhit;
    rayhit.ray = camera.get_ray(x, y, rng);
    for (uint32_t i = 0; i < max_bounces; ++i) {
        ray_count++;

        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(scene, &rayhit);
        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            return color;
        }

        GeometryData *user_data =
            (GeometryData *)rtcGetGeometryUserDataFromScene(scene, rayhit.hit.instID[0]);

        glm::vec2 bary = {rayhit.hit.u, rayhit.hit.v};

        glm::mat4 transform;
        rtcGetGeometryTransformFromScene(
            scene, rayhit.hit.geomID, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &transform
        );
        glm::mat3 normal_transform = glm::transpose(glm::inverse(glm::mat3(transform)));

        const uint32_t *prim_indices = &user_data->index_buffer[rayhit.hit.primID * 3];
        std::array<glm::vec3, 3> vertex_normals = {
            user_data->normal_buffer[prim_indices[0]],
            user_data->normal_buffer[prim_indices[1]],
            user_data->normal_buffer[prim_indices[2]],
        };

        // os << "N[0] = " << vertex_normals[0].x << ", " << vertex_normals[0].y << ", "
        //    << vertex_normals[0].z << sycl::endl;

        // os << "N[1] = " << vertex_normals[1].x << ", " << vertex_normals[1].y << ", "
        //    << vertex_normals[1].z << sycl::endl;

        // os << "N[2] = " << vertex_normals[2].x << ", " << vertex_normals[2].y << ", "
        //    << vertex_normals[2].z << sycl::endl;

        // TODO: transform into world space
        glm::vec3 vertex_normal = glm::normalize(
            (1 - bary.x - bary.y) * vertex_normals[0] + bary.x * vertex_normals[1] +
            bary.y * vertex_normals[2]
        );
        // vertex_normal = normal_transform * vertex_normal;

        glm::vec3 g_normal = glm::normalize(glm::vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));
        // g_normal = (g_normal * vertex_normal);
        // g_normal = user_data->obj_to_world * vertex_normal;
        g_normal = normal_transform * vertex_normal;

        // const float3 normal = normalize(
        //     normalize(float3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z)) *
        //     normalize(float3(vertex_normal.x, vertex_normal.y, vertex_normal.z))
        // );

        const float3 normal = float3(g_normal.x, g_normal.y, g_normal.z);

        const float3 dir =
            normalize(float3(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z));

        ScatterResult result;
        if (user_data->material.scatter(rng, dir, normal, result)) {
            rayhit.ray.org_x = rayhit.ray.org_x + rayhit.ray.dir_x * rayhit.ray.tfar;
            rayhit.ray.org_z = rayhit.ray.org_z + rayhit.ray.dir_z * rayhit.ray.tfar;
            rayhit.ray.org_y = rayhit.ray.org_y + rayhit.ray.dir_y * rayhit.ray.tfar;

            rayhit.ray.tnear = 0.0001f;
            rayhit.ray.tfar = std::numeric_limits<float>::infinity();

            rayhit.ray.dir_x = result.dir.x();
            rayhit.ray.dir_y = result.dir.y();
            rayhit.ray.dir_z = result.dir.z();

            color *= result.attenuation;
        } else {
            return float4(0, 0, 0, 1);
        }
    }

    return float4(0, 0, 0, 1);
}

void render_frame(
    App &app,
    const Camera &camera,
    const Scene &scene,
    range<2> img_size,
    sycl::image<2> &image
) {
    uint32_t initial_ray_count = 0;
    sycl::buffer<uint32_t> ray_count_buffer{&initial_ray_count, 1};

    auto begin = std::chrono::high_resolution_clock::now();

    auto e = app.queue.submit([&](sycl::handler &cgh) {
        sycl::stream os(8192, 256, cgh);
        RTCScene r_scene = scene.scene;

        auto image_writer = image.get_access<float4, sycl::access::mode::write>(cgh);
        auto ray_count = ray_count_buffer.get_access<sycl::access_mode::write>(cgh);

        range<2> local_size{8, 8};
        range<2> n_groups = img_size;
        n_groups[0] = ((img_size[0] + local_size[0] - 1) / local_size[0]);
        n_groups[1] = ((img_size[1] + local_size[1] - 1) / local_size[1]);

        cgh.parallel_for(
            sycl::nd_range<2>(n_groups * local_size, local_size),
            [=](sycl::nd_item<2> id) {
                auto global_id = id.get_global_id();
                if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                    return;
                }

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    ray_count_ref(ray_count[0]);

                int x = global_id[0];
                int y = global_id[1];

                auto init_generator_state =
                    std::hash<std::size_t>{}(id.get_global_linear_id());
                auto rng = XorShift32State{(uint32_t)init_generator_state};

                constexpr uint32_t sample_count = 256;

                uint32_t ray_count = 0;
                float4 pixel_color = float4(0, 0, 0, 0);
                for (uint32_t i = 0; i < sample_count; ++i) {
                    pixel_color +=
                        render_pixel(camera, rng, r_scene, x, y, ray_count, os);
                }
                pixel_color /= (float)sample_count;

                pixel_color = linear_to_gamma(pixel_color);

                image_writer.write(int2(x, y), pixel_color);

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

    printf("Time measured: %.6lf seconds\n", secs);
    printf("Total rays: %u\n", ray_count[0]);
    printf("Rays/sec: %.2lfM\n", rays_per_sec / 1000000.0);

    printf("Writing image to disk\n");
    write_image(app.queue, image, img_size[0], img_size[1]);
}
} // namespace raytracer
