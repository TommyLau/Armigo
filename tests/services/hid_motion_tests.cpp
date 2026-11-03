// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic coverage for the six-axis sensor fusion:
// a controller at rest, a steady turn about each axis, the zero-drift dead
// bands, a controller without a gyroscope, and gaps too long to integrate.

#include "hid_motion.h"

#include <cmath>
#include <cstdio>

namespace {

using armigo::services::hid::detail::MotionFusion;
using Vec3 = MotionFusion::Vec3;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

bool near(float a, float b, float tolerance) { return std::fabs(a - b) <= tolerance; }

float dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

bool orthonormal(const std::array<Vec3, 3>& rows) {
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (!near(dot(rows[i], rows[j]), i == j ? 1.0f : 0.0f, 1e-3f)) return false;
    return true;
}

// 200 samples of 5 ms each: one second.
void run(MotionFusion& fusion, const Vec3& acceleration, const Vec3& angular_velocity, int samples = 200) {
    for (int i = 0; i < samples; ++i) fusion.update(acceleration, angular_velocity, 5000);
}

void test_rest() {
    MotionFusion fusion;
    run(fusion, {0.0f, 0.0f, -1.0f}, {});
    const auto rows = fusion.orientation();
    CHECK(near(rows[0][0], 1.0f, 1e-5f) && near(rows[1][1], 1.0f, 1e-5f) && near(rows[2][2], 1.0f, 1e-5f));
    CHECK(fusion.angle() == (Vec3{}));
    CHECK(fusion.angular_velocity() == (Vec3{}));
    CHECK(fusion.acceleration() == (Vec3{0.0f, 0.0f, -1.0f}));
    CHECK(fusion.at_rest());
}

// A quarter turn per second about z for one second: the angle reads a quarter
// revolution, the orientation turned 90 degrees about z and still levels the
// z row with gravity.
void test_turn_about_z() {
    MotionFusion fusion;
    run(fusion, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.25f});
    CHECK(near(fusion.angle()[2], 0.25f, 1e-3f));
    CHECK(near(fusion.angle()[0], 0.0f, 1e-6f) && near(fusion.angle()[1], 0.0f, 1e-6f));
    CHECK(fusion.angular_velocity()[2] == 0.25f);
    CHECK(!fusion.at_rest());
    const auto rows = fusion.orientation();
    CHECK(orthonormal(rows));
    CHECK(near(rows[2][2], 1.0f, 1e-3f));
    // A positive turn about z (counter-clockwise seen from above) turns the
    // x row into +y.
    CHECK(near(rows[0][1], 1.0f, 2e-2f) && near(rows[0][0], 0.0f, 2e-2f));
}

// A turn about x tilts the controller: the gravity correction cannot hold the
// z row up against a steady gyroscope rate, so the orientation leaves level.
void test_turn_about_x() {
    MotionFusion fusion;
    run(fusion, {0.0f, 0.0f, -1.0f}, {0.125f, 0.0f, 0.0f});
    CHECK(near(fusion.angle()[0], 0.125f, 1e-3f));
    const auto rows = fusion.orientation();
    CHECK(orthonormal(rows));
    CHECK(rows[2][2] < 0.9f);
}

// Rates inside the zero-drift mode's dead band read as zero.
void test_dead_bands() {
    MotionFusion tight;
    tight.set_zero_drift_mode(2);
    tight.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.0015f}, 5000);
    CHECK(tight.angular_velocity() == (Vec3{}));
    tight.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.003f}, 5000);
    CHECK(tight.angular_velocity()[2] > 0.0f);
    MotionFusion standard;
    standard.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.005f}, 5000);
    CHECK(standard.angular_velocity() == (Vec3{}));
    MotionFusion loose;
    loose.set_zero_drift_mode(0);
    loose.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.009f}, 5000);
    CHECK(loose.angular_velocity() == (Vec3{}));
}

// Without a gyroscope, tilting the accelerometer turns the orientation toward
// the measured gravity.
void test_accelerometer_only() {
    MotionFusion fusion;
    const float s = std::sqrt(0.5f);
    for (int i = 0; i < 400; ++i) fusion.update({0.0f, -s, -s}, {}, 5000, false);
    const auto rows = fusion.orientation();
    CHECK(orthonormal(rows));
    CHECK(rows[2][2] < 0.95f);
}

// A controller with a gyroscope that starts tilted, with rates inside the
// dead band, reports no rotation: it does not fall back to the
// accelerometer-only estimate.
void test_tilted_start_with_gyroscope() {
    MotionFusion fusion;
    fusion.set_zero_drift_mode(2);
    const float s = std::sqrt(0.5f);
    for (int i = 0; i < 40; ++i) {
        fusion.update({0.0f, -s, -s}, {0.0005f, 0.0f, 0.0f}, 5000);
        CHECK(fusion.angular_velocity() == (Vec3{}));
    }
    CHECK(fusion.angle() == (Vec3{}));
}

// A gap longer than 0.1 s does not integrate.
void test_long_gap() {
    MotionFusion fusion;
    fusion.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, 200000);
    CHECK(fusion.angle() == (Vec3{}));
    fusion.update({0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, 10000);
    CHECK(near(fusion.angle()[2], 0.01f, 1e-5f));
}

// Out-of-range samples are clamped (5 rev/s, 7 G).
void test_clamps() {
    MotionFusion fusion;
    fusion.update({0.0f, 0.0f, -20.0f}, {0.0f, 0.0f, 50.0f}, 5000);
    CHECK(fusion.acceleration()[2] == -7.0f);
    CHECK(fusion.angular_velocity()[2] == 5.0f);
}

} // namespace

int main() {
    test_rest();
    test_turn_about_z();
    test_turn_about_x();
    test_dead_bands();
    test_accelerometer_only();
    test_tilted_start_with_gyroscope();
    test_long_gap();
    test_clamps();
    std::printf("HID motion failures: %u\n", failures);
    return failures ? 1 : 0;
}
