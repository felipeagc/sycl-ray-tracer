#pragma once

#include <fmt/core.h>
#include <sycl/sycl.hpp>
#include <glm/glm.hpp>

template <> struct ::fmt::formatter<glm::vec3> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }

    auto format(glm::vec3 t, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "[{} {} {}]", t[0], t[1], t[2]);
    }
};

template <> struct ::fmt::formatter<glm::mat4> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }

    auto format(glm::mat4 m, fmt::format_context &ctx) {
        auto appender = ctx.out();
        for (int i = 0; i < 4; ++i) {
            appender = fmt::format_to(
                appender, "{} {} {} {}\n", m[i][0], m[i][1], m[i][2], m[i][3]
            );
        }
        return appender;
    }
};

template <> struct ::fmt::formatter<sycl::float3> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }

    auto format(sycl::float3 t, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "[{} {} {}]", t[0], t[1], t[2]);
    }
};

template <> struct ::fmt::formatter<sycl::float4> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }

    auto format(sycl::float4 t, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "[{} {} {} {}]", t[0], t[1], t[2], t[3]);
    }
};
