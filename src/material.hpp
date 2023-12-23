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
        float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
        float sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > rng(0.0f, 1.0f)) {
            dir = reflect(unit_direction, normal);
        } else {
            dir = refract(unit_direction, normal, refraction_ratio);
        }

        return true;
    }

    static float reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * sycl::pow((1.0f - cosine), 5.0f);
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
