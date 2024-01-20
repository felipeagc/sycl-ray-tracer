#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

#include "render.hpp"
#include "xorshift.hpp"

namespace raytracer {

static inline std::optional<sycl::float3> trace_ray(
    const RenderContext &ctx,
    XorShift32State &rng,
    RTCRay &ray,
    sycl::float3 &attenuation,
    sycl::float3 &radiance
) {
    RTCRayHit rayhit;
    rayhit.ray = ray;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(ctx.scene, &rayhit);

    // If not hit, return sky color
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return attenuation * (ctx.sky_color + radiance);
    }

    GeometryData *user_data =
        (GeometryData *)rtcGetGeometryUserDataFromScene(ctx.scene, rayhit.hit.instID[0]);

    glm::vec2 bary = {rayhit.hit.u, rayhit.hit.v};

    const uint32_t *prim_indices = &user_data->index_buffer[rayhit.hit.primID * 3];
    std::array<glm::vec3, 3> vertex_normals = {
        user_data->normal_buffer[prim_indices[0]],
        user_data->normal_buffer[prim_indices[1]],
        user_data->normal_buffer[prim_indices[2]],
    };

    std::array<sycl::float2, 3> vertex_uvs = {
        user_data->uv_buffer[prim_indices[0]],
        user_data->uv_buffer[prim_indices[1]],
        user_data->uv_buffer[prim_indices[2]],
    };

    // Calculate UVs
    sycl::float2 vertex_uv = (1 - bary.x - bary.y) * vertex_uvs[0] +
                             bary.x * vertex_uvs[1] + bary.y * vertex_uvs[2];

    // Calculate normals
    glm::vec3 vertex_normal = glm::normalize(
        (1 - bary.x - bary.y) * vertex_normals[0] + bary.x * vertex_normals[1] +
        bary.y * vertex_normals[2]
    );

    glm::vec3 g_normal = user_data->obj_to_world * vertex_normal;
    const sycl::float3 normal =
        normalize(sycl::float3(g_normal.x, g_normal.y, g_normal.z));

    const sycl::float3 dir =
        normalize(sycl::float3(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z));

    radiance += user_data->material.emitted();

    ScatterResult result;
    if (user_data->material.scatter(ctx, rng, dir, normal, vertex_uv, result)) {
        ray.org_x = rayhit.ray.org_x + rayhit.ray.dir_x * rayhit.ray.tfar;
        ray.org_z = rayhit.ray.org_z + rayhit.ray.dir_z * rayhit.ray.tfar;
        ray.org_y = rayhit.ray.org_y + rayhit.ray.dir_y * rayhit.ray.tfar;

        ray.dir_x = result.dir.x();
        ray.dir_y = result.dir.y();
        ray.dir_z = result.dir.z();

        attenuation = attenuation * result.attenuation;
    } else {
        return attenuation * radiance;
    }

    return {};
}

} // namespace raytracer
