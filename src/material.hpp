#pragma once

#include <embree4/rtcore.h>

#include "xorshift.hpp"
#include "util.hpp"

namespace raytracer {

enum class MaterialType : uint8_t {
    eDiffuse,
    eMetallic,
    eDielectric,
};

struct MaterialDiffuse {
    sycl::float4 albedo;

    inline bool scatter(
        sycl::float3 &dir,
        sycl::float3 normal,
        XorShift32State &rng,
        sycl::float4 &attenuation
    ) const {
        dir = normal + rng.random_unit_vector();
        if (near_zero(dir)) {
            dir = normal;
        }
        attenuation = this->albedo;
        return true;
    }
};

struct MaterialMetallic {
    sycl::float4 albedo;
    float roughness;

    inline bool scatter(
        sycl::float3 &dir,
        sycl::float3 normal,
        XorShift32State &rng,
        sycl::float4 &attenuation
    ) const {
        sycl::float3 reflected = reflect(dir, normal);
        dir = reflected + this->roughness * rng.random_unit_vector();
        attenuation = this->albedo;
        return (dot(dir, normal) > 0);
    }
};

struct MaterialDielectric {
    float ior;

    inline bool scatter(
        sycl::float3 &dir,
        sycl::float3 outward_normal,
        XorShift32State &rng,
        sycl::float4 &attenuation
    ) const {
        attenuation = sycl::float4(1, 1, 1, 1);
        bool front_face = dot(dir, outward_normal) < 0;

        sycl::float3 normal = front_face ? outward_normal : -outward_normal;
        float refraction_ratio = front_face ? (1.0f / this->ior) : this->ior;

        sycl::float3 unit_direction = normalize(dir);
        sycl::float3 refracted = refract(unit_direction, normal, refraction_ratio);

        dir = refracted;
        return true;
    }
};

struct Material {
    MaterialType type;
    union {
        MaterialDiffuse diffuse;
        MaterialMetallic metallic;
        MaterialDielectric dielectric;
    };

    Material() {
        *this = Material(MaterialDiffuse{.albedo = sycl::float4(1, 1, 1, 1)});
    }

    Material(const Material &other) {
        this->type = other.type;
        switch (this->type) {
        case MaterialType::eDiffuse: this->diffuse = other.diffuse; break;
        case MaterialType::eMetallic: this->metallic = other.metallic; break;
        case MaterialType::eDielectric: this->dielectric = other.dielectric; break;
        }
    }

    Material &operator=(const Material &other) {
        this->type = other.type;
        switch (this->type) {
        case MaterialType::eDiffuse: this->diffuse = other.diffuse; break;
        case MaterialType::eMetallic: this->metallic = other.metallic; break;
        case MaterialType::eDielectric: this->dielectric = other.dielectric; break;
        }
        return *this;
    }

    Material(MaterialDiffuse diffuse) {
        type = MaterialType::eDiffuse;
        this->diffuse = diffuse;
    }

    Material(MaterialMetallic metallic) {
        type = MaterialType::eMetallic;
        this->metallic = metallic;
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

        sycl::float3 dir =
            normalize(sycl::float3(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z));

        sycl::float3 normal =
            normalize(sycl::float3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));

        bool res = false;
        switch (this->type) {
        case MaterialType::eDiffuse:
            res = this->diffuse.scatter(dir, normal, rng, attenuation);
            break;
        case MaterialType::eMetallic:
            res = this->metallic.scatter(dir, normal, rng, attenuation);
            break;
        case MaterialType::eDielectric:
            res = this->dielectric.scatter(dir, normal, rng, attenuation);
            break;
        }

        scattered.dir_x = dir.x();
        scattered.dir_y = dir.y();
        scattered.dir_z = dir.z();

        return res;
    }
};

} // namespace raytracer
