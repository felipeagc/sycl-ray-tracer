#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

#include "image_manager.hpp"
#include "xorshift.hpp"
#include "util.hpp"
#include "render_context.hpp"

namespace raytracer {

enum class TextureType : uint8_t {
    eColor,
    eImage,
};

struct Texture {
    TextureType type;
    union {
        sycl::float3 color;
        ImageRef image_ref;
    };

    Texture(sycl::float3 color) : type(TextureType::eColor), color(color) {}

    Texture(ImageRef image_ref) : type(TextureType::eImage), image_ref(image_ref) {}

    inline sycl::float3 sample(const RenderContext &ctx, sycl::float2 uv) const {
        switch (this->type) {
        case TextureType::eColor: return this->color;
        case TextureType::eImage:
            sycl::float4 color = ctx.image_reader[this->image_ref.index].read(
                sycl::float2(0.0f, 0.0f), ctx.sampler
            );
            return sycl::float3(color.x(), color.y(), color.z());
        }
    }
};

struct ScatterResult {
    sycl::float3 dir;
    sycl::float3 attenuation;
};

enum class MaterialType : uint8_t {
    eNone,
    eDiffuse,
    eMetallic,
    eDielectric,
};

struct MaterialDiffuse {
    Texture albedo;
    sycl::float3 emissive;

    inline bool scatter(
        const RenderContext &ctx,
        XorShift32State &rng,
        const sycl::float3 &dir,
        const sycl::float3 &normal,
        ScatterResult &result
    ) const {
        result.dir = normal + rng.random_unit_vector();
        if (near_zero(dir)) {
            result.dir = normal;
        }
        result.attenuation = this->albedo.sample(ctx, sycl::float2(0.0f, 0.0f));
        return true;
    }

    inline sycl::float3 emitted() const {
        return this->emissive;
    }
};

struct MaterialMetallic {
    sycl::float3 albedo;
    float roughness;

    inline bool scatter(
        const RenderContext &ctx,
        XorShift32State &rng,
        const sycl::float3 &dir,
        const sycl::float3 &normal,
        ScatterResult &result
    ) const {
        sycl::float3 reflected = reflect(dir, normal);
        result.dir = reflected + this->roughness * rng.random_unit_vector();
        result.attenuation = this->albedo;
        return dot(result.dir, normal) > 0;
    }

    inline sycl::float3 emitted() const {
        return sycl::float3(0, 0, 0);
    }
};

struct MaterialDielectric {
    float ior;

    static float reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * sycl::pow((1.0f - cosine), 5.0f);
    }

    inline bool scatter(
        const RenderContext &ctx,
        XorShift32State &rng,
        const sycl::float3 &dir,
        const sycl::float3 &outward_normal,
        ScatterResult &result
    ) const {
        result.attenuation = sycl::float3(1, 1, 1);

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

        return true;
    }

    inline sycl::float3 emitted() const {
        return sycl::float3(0, 0, 0);
    }
};

struct Material {
    MaterialType type = MaterialType::eNone;
    union {
        MaterialDiffuse diffuse;
        MaterialMetallic metallic;
        MaterialDielectric dielectric;
    };

    Material(const Material &other) {
        this->type = other.type;
        switch (this->type) {
        case MaterialType::eDiffuse: this->diffuse = other.diffuse; break;
        case MaterialType::eMetallic: this->metallic = other.metallic; break;
        case MaterialType::eDielectric: this->dielectric = other.dielectric; break;
        case MaterialType::eNone: break;
        }
    }

    Material &operator=(const Material &other) {
        this->type = other.type;
        switch (this->type) {
        case MaterialType::eDiffuse: this->diffuse = other.diffuse; break;
        case MaterialType::eMetallic: this->metallic = other.metallic; break;
        case MaterialType::eDielectric: this->dielectric = other.dielectric; break;
        case MaterialType::eNone: break;
        }
        return *this;
    }

    Material() {
        type = MaterialType::eNone;
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

    inline bool scatter(
        const RenderContext &ctx,
        XorShift32State &rng,
        const sycl::float3 &dir,
        const sycl::float3 &normal,
        ScatterResult &result
    ) const {
        switch (this->type) {
        case MaterialType::eDiffuse:
            return this->diffuse.scatter(ctx, rng, dir, normal, result);
        case MaterialType::eMetallic:
            return this->metallic.scatter(ctx, rng, dir, normal, result);
        case MaterialType::eDielectric:
            return this->dielectric.scatter(ctx, rng, dir, normal, result);
        case MaterialType::eNone: return false;
        }
    }

    inline sycl::float3 emitted() const {
        switch (this->type) {
        case MaterialType::eDiffuse: return this->diffuse.emitted();
        case MaterialType::eMetallic: return this->metallic.emitted();
        case MaterialType::eDielectric: return this->dielectric.emitted();
        case MaterialType::eNone: return sycl::float3(0.0f);
        }
    }
};

} // namespace raytracer
