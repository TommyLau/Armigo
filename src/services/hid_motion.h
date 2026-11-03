// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <cstdint>

namespace armigo::services::hid::detail {

// Sensor fusion for one six-axis sensor, in the controller's frame (x right,
// y forward, z up). Each sample integrates the gyroscope into an accumulated
// angle and into an orientation that a proportional-integral term pulls
// toward the measured gravity; slow gyroscope bias is removed while the
// controller rests, and rates inside the zero-drift mode's dead band read as
// zero. Without a gyroscope the orientation follows the accelerometer alone.
// At rest, face up, the orientation is the identity.
class MotionFusion {
public:
    using Vec3 = std::array<float, 3>;

    MotionFusion();

    // The guest's zero-drift mode: 0 loose, 1 standard, 2 tight (others are
    // treated as standard).
    void set_zero_drift_mode(std::uint32_t mode);

    // One sample: acceleration in G, angular velocity in revolutions per
    // second, and the time since the previous sample in microseconds. A gap
    // longer than 0.1 s does not integrate. A sensor without a gyroscope
    // (has_gyroscope false) steers the orientation by the accelerometer alone
    // and reports the rate that correction implies.
    void update(const Vec3& acceleration, const Vec3& angular_velocity, std::uint64_t elapsed_us,
                bool has_gyroscope = true);

    // The sample as the guest reads it: clamped acceleration, angular velocity
    // after bias removal and the dead band, the accumulated angle in
    // revolutions, and the orientation as three rows of a rotation matrix.
    const Vec3& acceleration() const noexcept { return accel_; }
    const Vec3& angular_velocity() const noexcept { return gyro_; }
    const Vec3& angle() const noexcept { return rotations_; }
    std::array<Vec3, 3> orientation() const;
    // Whether the sensor rests by the current zero-drift mode's threshold.
    bool at_rest() const;

private:
    // A quaternion as x, y, z, w.
    using Quat = std::array<float, 4>;

    void set_acceleration(const Vec3& acceleration);
    void set_gyroscope(const Vec3& angular_velocity);
    void update_rotation(std::uint64_t elapsed_us);
    void update_orientation(std::uint64_t elapsed_us);
    void reset_orientation();
    void orientation_from_accelerometer();
    bool moving(float sensitivity) const;
    bool calibrated(float sensitivity) const;

    float kp_ = 0.3f, ki_ = 0.005f, kd_ = 0.0f;
    Vec3 real_error_{}, integral_error_{}, derivative_error_{};
    Quat quat_{0.0f, 0.0f, -1.0f, 0.0f};
    Vec3 rotations_{};
    Vec3 accel_{0.0f, 0.0f, -1.0f};
    Vec3 gyro_{};
    Vec3 gyro_bias_{};
    float gyro_threshold_ = 0.0f;
    float rest_threshold_ = 0.0f;
    std::uint32_t reset_counter_ = 0;
    bool only_accelerometer_ = true;
};

} // namespace armigo::services::hid::detail
