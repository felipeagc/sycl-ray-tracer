#pragma once

#include <embree4/rtcore.h>

#include "xorshift.hpp"
#include "util.hpp"

namespace raytracer {

struct ScatterResult {
    sycl::float3 dir;
    sycl::float4 attenuation;
};

enum class MaterialType : uint8_t {
    eDiffuse,
    eMetallic,
    eDielectric,
};

struct MaterialDiffuse {
    sycl::float4 albedo;

    inline std::optional<ScatterResult> scatter(
        XorShift32State &rng, const sycl::float3 &dir, const sycl::float3 &normal
    ) const {
        ScatterResult result;
        result.dir = normal + rng.random_unit_vector();
        if (near_zero(dir)) {
            result.dir = normal;
        }
        result.attenuation = this->albedo;
        return result;
    }
};

struct MaterialMetallic {
    sycl::float4 albedo;
    float roughness;

    inline std::optional<ScatterResult> scatter(
        XorShift32State &rng, const sycl::float3 &dir, const sycl::float3 &normal
    ) const {
        ScatterResult result;

        sycl::float3 reflected = reflect(dir, normal);
        result.dir = reflected + this->roughness * rng.random_unit_vector();
        if (dot(result.dir, normal) <= 0) {
            return {};
        }

        result.attenuation = this->albedo;
        return result;
    }
};

struct MaterialDielectric {
    float ior;

    inline std::optional<ScatterResult> scatter(
        XorShift32State &rng, const sycl::float3 &dir, const sycl::float3 &outward_normal
    ) const {
        ScatterResult result;

        result.attenuation = sycl::float4(1, 1, 1, 1);

        bool front_face = dot(dir, outward_normal) < 0;

        sycl::float3 normal = front_face ? outward_normal : -outward_normal;
        float refraction_ratio = front_face ? (1.0f / this->ior) : this->ior;

        sycl::float3 unit_direction = normalize(dir);
        float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
        float sin_theta = sycl::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > rng(0.0f, 1.0f)) {
            result.dir = reflect(unit_direction, normal);
        } else {
            result.dir = refract(unit_direction, normal, refraction_ratio);
        }

        return result;
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

    inline std::optional<ScatterResult> scatter(
        XorShift32State &rng, const sycl::float3 &dir, const sycl::float3 &normal
    ) const {
        switch (this->type) {
        case MaterialType::eDiffuse: return this->diffuse.scatter(rng, dir, normal);
        case MaterialType::eMetallic: return this->metallic.scatter(rng, dir, normal);
        case MaterialType::eDielectric: return this->dielectric.scatter(rng, dir, normal);
        }
    }
};

} // namespace raytracer
