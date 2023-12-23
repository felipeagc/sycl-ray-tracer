#pragma once

#include <embree4/rtcore.h>

#include "xorshift.hpp"
#include "util.hpp"

namespace raytracer {

enum class MaterialType : uint8_t {
    eDiffuse,
    eDielectric,
};

struct MaterialDiffuse {
    sycl::float4 albedo;

    inline bool scatter(
        sycl::float3 &new_dir,
        sycl::float3 normal,
        XorShift32State &rng,
        sycl::float4 &attenuation
    ) const {
        new_dir = normal + rng.random_unit_vector();
        if (near_zero(new_dir)) {
            new_dir = normal;
        }
        attenuation = this->albedo;
        return true;
    }
};

struct MaterialDielectric {
    inline bool scatter(
        sycl::float3 &new_dir,
        sycl::float3 normal,
        XorShift32State &rng,
        sycl::float4 &attenuation
    ) const {
        return false;
    }
};

struct Material {
    MaterialType type;
    union {
        MaterialDiffuse diffuse;
        MaterialDielectric dielectric;
    };

    Material(MaterialDiffuse diffuse) {
        type = MaterialType::eDiffuse;
        this->diffuse = diffuse;
    }

    Material(MaterialDielectric dielectric) {
        type = MaterialType::eDielectric;
        this->dielectric = dielectric;
    }

    bool scatter(
        const RTCRayHit &rayhit,
        sycl::float4 &attenuation,
        RTCRay &scattered,
        XorShift32State &rng
    ) const {
        scattered.org_x = rayhit.ray.org_x + rayhit.ray.dir_x * rayhit.ray.tfar;
        scattered.org_y = rayhit.ray.org_y + rayhit.ray.dir_y * rayhit.ray.tfar;
        scattered.org_z = rayhit.ray.org_z + rayhit.ray.dir_z * rayhit.ray.tfar;

        scattered.tnear = 0.0001f;
        scattered.tfar = std::numeric_limits<float>::infinity();

        sycl::float3 new_dir;

        sycl::float3 normal =
            normalize(sycl::float3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));

        bool res = false;
        switch (this->type) {
        case MaterialType::eDiffuse:
            res = this->diffuse.scatter(new_dir, normal, rng, attenuation);
            break;
        case MaterialType::eDielectric:
            res = this->dielectric.scatter(new_dir, normal, rng, attenuation);
            break;
        }

        scattered.dir_x = new_dir.x();
        scattered.dir_y = new_dir.y();
        scattered.dir_z = new_dir.z();

        return res;
    }
};

} // namespace raytracer
