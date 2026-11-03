// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "hid_motion.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace armigo::services::hid::detail {
namespace {

// Gyroscope dead bands (revolutions per second) and at-rest thresholds per
// zero-drift mode; the relaxed threshold gates bias learning.
constexpr float threshold_loose = 0.01f;
constexpr float threshold_standard = 0.007f;
constexpr float threshold_tight = 0.002f;
constexpr float at_rest_relaxed = 0.05f;
constexpr float at_rest_loose = 0.02f;
constexpr float at_rest_standard = 0.01f;
constexpr float at_rest_tight = 0.005f;
constexpr float gyro_max = 5.0f;
constexpr float accel_max = 7.0f;

using Vec3 = MotionFusion::Vec3;

float length(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vec3 normalized(const Vec3& v) {
    const float l = length(v);
    if (l == 0.0f) return v;
    return {v[0] / l, v[1] / l, v[2] / l};
}

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 operator*(const Vec3& a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 operator*(float s, const Vec3& a) { return a * s; }

std::array<float, 4> normalized(const std::array<float, 4>& q) {
    const float l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (l == 0.0f) return q;
    return {q[0] / l, q[1] / l, q[2] / l, q[3] / l};
}

// Rotation matrix (row-major 4x4) of a quaternion stored as x, y, z, w.
std::array<float, 16> to_matrix(const std::array<float, 4>& q) {
    const float x2 = q[0] * q[0], y2 = q[1] * q[1], z2 = q[2] * q[2];
    const float xy = q[0] * q[1], wz = q[3] * q[2], xz = q[0] * q[2];
    const float wy = q[3] * q[1], yz = q[1] * q[2], wx = q[3] * q[0];
    return {1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz), 2.0f * (xz - wy), 0.0f,
            2.0f * (xy - wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx), 0.0f,
            2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (x2 + y2), 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
}

} // namespace

MotionFusion::MotionFusion() {
    set_zero_drift_mode(1);
}

void MotionFusion::set_zero_drift_mode(std::uint32_t mode) {
    switch (mode) {
    case 0:
        gyro_threshold_ = threshold_loose;
        rest_threshold_ = at_rest_loose;
        break;
    case 2:
        gyro_threshold_ = threshold_tight;
        rest_threshold_ = at_rest_tight;
        break;
    default:
        gyro_threshold_ = threshold_standard;
        rest_threshold_ = at_rest_standard;
        break;
    }
}

void MotionFusion::update(const Vec3& acceleration, const Vec3& angular_velocity, std::uint64_t elapsed_us,
                          bool has_gyroscope) {
    // A sensor with a gyroscope leaves accelerometer-only mode on its first
    // sample, whatever the rate.
    if (has_gyroscope) only_accelerometer_ = false;
    set_acceleration(acceleration);
    set_gyroscope(angular_velocity);
    update_rotation(elapsed_us);
    update_orientation(elapsed_us);
}

void MotionFusion::set_acceleration(const Vec3& acceleration) {
    for (std::size_t i = 0; i < 3; ++i) accel_[i] = std::clamp(acceleration[i], -accel_max, accel_max);
}

void MotionFusion::set_gyroscope(const Vec3& angular_velocity) {
    gyro_ = angular_velocity - gyro_bias_;
    for (auto& axis : gyro_) axis = std::clamp(axis, -gyro_max, gyro_max);
    // Learn the bias slowly while the controller rests.
    if (!moving(at_rest_relaxed)) gyro_bias_ = (gyro_bias_ * 0.9999f) + (angular_velocity * 0.0001f);
    if (length(gyro_) < gyro_threshold_) gyro_ = {};
    else only_accelerometer_ = false;
}

void MotionFusion::update_rotation(std::uint64_t elapsed_us) {
    const float period = static_cast<float>(elapsed_us) / 1000000.0f;
    if (period > 0.1f) return;
    rotations_ = rotations_ + gyro_ * period;
}

void MotionFusion::update_orientation(std::uint64_t elapsed_us) {
    if (!calibrated(0.1f)) reset_orientation();
    float q1 = quat_[3], q2 = quat_[0], q3 = quat_[1], q4 = quat_[2];
    const float period = static_cast<float>(elapsed_us) / 1000000.0f;
    if (period > 0.1f) return;

    const auto normal = normalized(accel_);
    Vec3 rad_gyro = gyro_ * (std::numbers::pi_v<float> * 2.0f);
    const float swap = rad_gyro[0];
    rad_gyro[0] = rad_gyro[1];
    rad_gyro[1] = -swap;
    rad_gyro[2] = -rad_gyro[2];
    if (only_accelerometer_) rad_gyro = {};

    // Drift correction only while the accelerometer reads about one G.
    const float magnitude = length(accel_);
    if (magnitude >= 0.75f && magnitude <= 1.25f) {
        const float ax = -normal[0], ay = normal[1], az = -normal[2];
        // Gravity as the current orientation predicts it.
        const float vx = 2.0f * (q2 * q4 - q1 * q3);
        const float vy = 2.0f * (q1 * q2 + q3 * q4);
        const float vz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
        // The error is the cross product of the predicted and measured gravity.
        const Vec3 error{az * vx - ax * vz, ay * vz - az * vy, ax * vy - ay * vx};
        derivative_error_ = error - real_error_;
        real_error_ = error;
        if (ki_ != 0.0f && !calibrated(0.05f)) integral_error_ = integral_error_ + real_error_;
        else integral_error_ = {};
        if (!only_accelerometer_) {
            rad_gyro = rad_gyro + kp_ * real_error_;
            rad_gyro = rad_gyro + ki_ * integral_error_;
            rad_gyro = rad_gyro + kd_ * derivative_error_;
        } else {
            // Without a gyroscope the accelerometer weighs more, and the rate
            // the correction implies stands in for the missing gyroscope.
            rad_gyro = rad_gyro + 35.0f * kp_ * real_error_;
            rad_gyro = rad_gyro + 10.0f * ki_ * integral_error_;
            rad_gyro = rad_gyro + 10.0f * kd_ * derivative_error_;
            gyro_ = {-rad_gyro[1], rad_gyro[0], -rad_gyro[2]};
            update_rotation(elapsed_us);
        }
    }

    const float gx = rad_gyro[1], gy = rad_gyro[0], gz = rad_gyro[2];
    const float pa = q2, pb = q3, pc = q4;
    q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5f * period);
    q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5f * period);
    q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5f * period);
    q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5f * period);
    quat_ = normalized(Quat{q2, q3, q4, q1});
}

std::array<Vec3, 3> MotionFusion::orientation() const {
    const auto m = to_matrix({-quat_[1], -quat_[0], -quat_[3], -quat_[2]});
    return {Vec3{m[0], m[1], -m[2]}, Vec3{m[4], m[5], -m[6]}, Vec3{-m[8], -m[9], m[10]}};
}

bool MotionFusion::at_rest() const {
    return !moving(rest_threshold_);
}

void MotionFusion::reset_orientation() {
    if (only_accelerometer_) return;
    // A controller left face up and still for 900 samples is re-levelled
    // from its accelerometer.
    if (!moving(at_rest_relaxed) && accel_[2] <= -0.9f) {
        if (++reset_counter_ > 900) {
            quat_ = {0.0f, 0.0f, -1.0f, 0.0f};
            orientation_from_accelerometer();
            integral_error_ = {};
            reset_counter_ = 0;
        }
    } else {
        reset_counter_ = 0;
    }
}

void MotionFusion::orientation_from_accelerometer() {
    const float period = 0.015f;
    const auto normal = normalized(accel_);
    for (int iterations = 0; !calibrated(0.01f) && ++iterations < 100;) {
        float q1 = quat_[3], q2 = quat_[0], q3 = quat_[1], q4 = quat_[2];
        const float ax = -normal[0], ay = normal[1], az = -normal[2];
        const float vx = 2.0f * (q2 * q4 - q1 * q3);
        const float vy = 2.0f * (q1 * q2 + q3 * q4);
        const float vz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
        const Vec3 error{az * vx - ax * vz, ay * vz - az * vy, ax * vy - ay * vx};
        derivative_error_ = error - real_error_;
        real_error_ = error;
        Vec3 rad_gyro{};
        rad_gyro = rad_gyro + 10.0f * kp_ * real_error_;
        rad_gyro = rad_gyro + 5.0f * ki_ * integral_error_;
        rad_gyro = rad_gyro + 10.0f * kd_ * derivative_error_;
        const float gx = rad_gyro[1], gy = rad_gyro[0], gz = rad_gyro[2];
        const float pa = q2, pb = q3, pc = q4;
        q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5f * period);
        q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5f * period);
        q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5f * period);
        q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5f * period);
        quat_ = normalized(Quat{q2, q3, q4, q1});
    }
}

bool MotionFusion::moving(float sensitivity) const {
    const float magnitude = length(accel_);
    return length(gyro_) >= sensitivity || magnitude <= 0.9f || magnitude >= 1.1f;
}

bool MotionFusion::calibrated(float sensitivity) const {
    return length(real_error_) < sensitivity;
}

} // namespace armigo::services::hid::detail
