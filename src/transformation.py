"""
Core transformation algorithms for converting phone sensor data to vehicle reference frame.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from data_simulator import SensorData


@dataclass
class TransformationResult:
    """Result of sensor data transformation"""
    timestamp: float
    vehicle_acceleration: np.ndarray  # [forward, left, up] in vehicle frame
    estimated_orientation: np.ndarray  # [roll, pitch, yaw] phone relative to vehicle
    quality_score: float  # 0-1, higher is better
    confidence_metrics: dict  # Additional quality indicators


class OrientationEstimator:
    """Estimates phone orientation relative to vehicle using GPS and accelerometer data"""
    
    def __init__(self):
        self.gravity = 9.81
        self.previous_gps_velocity = None
        self.previous_timestamp = None
        
    def estimate_vehicle_heading_from_gps(self, gps_velocity: np.ndarray) -> float:
        """Estimate vehicle heading from GPS velocity"""
        if np.linalg.norm(gps_velocity[:2]) < 1.0:  # Too slow for reliable heading
            return None
        
        heading = np.arctan2(gps_velocity[1], gps_velocity[0])  # North = 0, East = π/2
        return heading
    
    def estimate_vehicle_acceleration_from_gps(self, current_velocity: np.ndarray, 
                                             previous_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Estimate vehicle acceleration from GPS velocity changes"""
        if dt <= 0:
            return np.zeros(3)
        
        accel_earth = (current_velocity - previous_velocity) / dt
        return accel_earth
    
    def align_with_gravity(self, accelerometer: np.ndarray) -> Tuple[float, float]:
        """Estimate roll and pitch from accelerometer assuming static conditions"""
        ax, ay, az = accelerometer
        
        # Normalize accelerometer reading
        norm = np.linalg.norm(accelerometer)
        if norm < 0.1:  # Too small to be reliable
            return 0.0, 0.0
        
        ax_norm, ay_norm, az_norm = accelerometer / norm
        
        # Calculate roll and pitch
        roll = np.arctan2(ay_norm, az_norm)
        pitch = np.arctan2(-ax_norm, np.sqrt(ay_norm**2 + az_norm**2))
        
        return roll, pitch
    
    def estimate_yaw_from_motion(self, phone_accel: np.ndarray, vehicle_accel_earth: np.ndarray,
                                vehicle_heading: float, roll: float, pitch: float) -> float:
        """Estimate yaw by comparing phone and vehicle accelerations"""
        if vehicle_heading is None or np.linalg.norm(vehicle_accel_earth) < 0.5:
            return 0.0
        
        # Remove gravity from phone accelerometer
        gravity_phone = self._gravity_in_phone_frame(roll, pitch, 0.0)
        phone_accel_corrected = phone_accel + gravity_phone
        
        # Vehicle acceleration in vehicle frame (forward, left, up)
        vehicle_accel_magnitude = np.linalg.norm(vehicle_accel_earth[:2])
        if vehicle_accel_magnitude < 0.1:
            return 0.0
        
        # Try different yaw angles and find best match
        best_yaw = 0.0
        best_error = float('inf')
        
        for yaw_test in np.linspace(-np.pi, np.pi, 36):  # 10-degree steps
            R_phone_to_vehicle = self._rotation_matrix_phone_to_vehicle(roll, pitch, yaw_test)
            vehicle_accel_from_phone = R_phone_to_vehicle @ phone_accel_corrected
            
            # Compare with expected vehicle acceleration direction
            expected_direction = np.array([1.0, 0.0, 0.0])  # Forward
            actual_direction = vehicle_accel_from_phone[:2]
            actual_direction = actual_direction / (np.linalg.norm(actual_direction) + 1e-6)
            
            error = np.linalg.norm(expected_direction[:2] - actual_direction)
            if error < best_error:
                best_error = error
                best_yaw = yaw_test
        
        return best_yaw
    
    def _gravity_in_phone_frame(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Calculate gravity vector in phone frame"""
        R_earth_to_phone = self._rotation_matrix_earth_to_phone(roll, pitch, yaw)
        gravity_earth = np.array([0, 0, -self.gravity])
        return R_earth_to_phone @ gravity_earth
    
    def _rotation_matrix_earth_to_phone(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Rotation matrix from earth frame to phone frame"""
        return R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    
    def _rotation_matrix_phone_to_vehicle(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Rotation matrix from phone frame to vehicle frame"""
        # This assumes vehicle frame: X=forward, Y=left, Z=up
        # Phone frame: X=right, Y=up, Z=out of screen (when held upright)
        return R.from_euler('ZYX', [-yaw, -pitch, -roll]).as_matrix()


class KalmanOrientationFilter:
    """Kalman filter for smoothing orientation estimates"""
    
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        
        # State transition matrix (constant velocity model)
        dt = 0.1  # Will be updated with actual dt
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe orientation directly)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise
        self.kf.Q *= 0.01
        
        # Measurement noise
        self.kf.R = np.diag([0.1, 0.1, 0.2])  # Higher uncertainty in yaw
        
        # Initial state covariance
        self.kf.P *= 100
        
        self.initialized = False
    
    def update(self, orientation: np.ndarray, dt: float) -> np.ndarray:
        """Update filter with new orientation measurement"""
        if not self.initialized:
            self.kf.x[:3, 0] = orientation
            self.kf.x[3:, 0] = 0  # Initial rates
            self.initialized = True
            return orientation
        
        # Update state transition matrix with actual dt
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        
        # Predict and update
        self.kf.predict()
        self.kf.update(orientation)
        
        return self.kf.x[:3, 0].copy()


class VehicleFrameTransformer:
    """Main class for transforming sensor data to vehicle reference frame"""
    
    def __init__(self):
        self.orientation_estimator = OrientationEstimator()
        self.kalman_filter = KalmanOrientationFilter()
        self.previous_gps_data = None
        self.previous_timestamp = None
        
    def transform_sensor_data(self, sensor_data: List[SensorData]) -> List[TransformationResult]:
        """Transform complete sensor dataset"""
        results = []
        
        for i, data in enumerate(sensor_data):
            result = self._transform_single_sample(data, i)
            results.append(result)
            
        return results
    
    def _transform_single_sample(self, data: SensorData, index: int) -> TransformationResult:
        """Transform a single sensor sample"""
        
        # Estimate vehicle motion from GPS
        vehicle_heading = None
        vehicle_accel_earth = np.zeros(3)
        
        if self.previous_gps_data is not None and self.previous_timestamp is not None:
            dt = data.timestamp - self.previous_timestamp
            if dt > 0:
                vehicle_heading = self.orientation_estimator.estimate_vehicle_heading_from_gps(data.gps_velocity)
                vehicle_accel_earth = self.orientation_estimator.estimate_vehicle_acceleration_from_gps(
                    data.gps_velocity, self.previous_gps_data.gps_velocity, dt
                )
        
        # Estimate phone orientation relative to vehicle
        roll, pitch = self.orientation_estimator.align_with_gravity(data.accelerometer)
        yaw = self.orientation_estimator.estimate_yaw_from_motion(
            data.accelerometer, vehicle_accel_earth, vehicle_heading, roll, pitch
        )
        
        raw_orientation = np.array([roll, pitch, yaw])
        
        # Apply Kalman filtering for smoothing
        dt = data.timestamp - self.previous_timestamp if self.previous_timestamp else 0.1
        filtered_orientation = self.kalman_filter.update(raw_orientation, dt)
        
        # Transform acceleration to vehicle frame
        R_phone_to_vehicle = self._rotation_matrix_phone_to_vehicle(*filtered_orientation)
        
        # Remove gravity from phone accelerometer
        gravity_phone = self._gravity_in_phone_frame(*filtered_orientation)
        phone_accel_corrected = data.accelerometer + gravity_phone
        
        # Transform to vehicle frame
        vehicle_acceleration = R_phone_to_vehicle @ phone_accel_corrected
        
        # Calculate quality metrics
        quality_score, confidence_metrics = self._calculate_quality_metrics(
            data, vehicle_heading, vehicle_accel_earth, filtered_orientation, raw_orientation
        )
        
        # Update previous data
        self.previous_gps_data = data
        self.previous_timestamp = data.timestamp
        
        return TransformationResult(
            timestamp=data.timestamp,
            vehicle_acceleration=vehicle_acceleration,
            estimated_orientation=filtered_orientation,
            quality_score=quality_score,
            confidence_metrics=confidence_metrics
        )
    
    def _rotation_matrix_phone_to_vehicle(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Rotation matrix from phone frame to vehicle frame"""
        return R.from_euler('ZYX', [-yaw, -pitch, -roll]).as_matrix()
    
    def _gravity_in_phone_frame(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Calculate gravity vector in phone frame"""
        R_earth_to_phone = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
        gravity_earth = np.array([0, 0, -9.81])
        return R_earth_to_phone @ gravity_earth
    
    def _calculate_quality_metrics(self, data: SensorData, vehicle_heading: Optional[float],
                                 vehicle_accel_earth: np.ndarray, filtered_orientation: np.ndarray,
                                 raw_orientation: np.ndarray) -> Tuple[float, dict]:
        """Calculate quality score and confidence metrics"""
        
        metrics = {}
        quality_factors = []
        
        # GPS quality factor
        gps_quality = min(1.0, 10.0 / max(data.gps_accuracy, 1.0))  # Better with lower accuracy values
        metrics['gps_quality'] = gps_quality
        quality_factors.append(gps_quality)
        
        # Motion consistency factor
        vehicle_speed = np.linalg.norm(data.gps_velocity[:2])
        if vehicle_speed > 2.0:  # Only meaningful when moving
            motion_quality = min(1.0, vehicle_speed / 20.0)  # Better at higher speeds
        else:
            motion_quality = 0.3  # Low quality when stationary
        metrics['motion_quality'] = motion_quality
        quality_factors.append(motion_quality)
        
        # Accelerometer magnitude consistency (should be close to gravity when stationary)
        accel_magnitude = np.linalg.norm(data.accelerometer)
        gravity_consistency = 1.0 - min(1.0, abs(accel_magnitude - 9.81) / 5.0)
        metrics['gravity_consistency'] = gravity_consistency
        quality_factors.append(gravity_consistency)
        
        # Orientation stability (difference between raw and filtered)
        orientation_stability = 1.0 - min(1.0, np.linalg.norm(filtered_orientation - raw_orientation) / np.pi)
        metrics['orientation_stability'] = orientation_stability
        quality_factors.append(orientation_stability)
        
        # Gyroscope consistency
        gyro_magnitude = np.linalg.norm(data.gyroscope)
        gyro_quality = 1.0 - min(1.0, gyro_magnitude / 2.0)  # Lower is better for gyro noise
        metrics['gyro_quality'] = gyro_quality
        quality_factors.append(gyro_quality)
        
        # Overall quality score (weighted average)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # GPS and motion are most important
        overall_quality = np.average(quality_factors, weights=weights)
        
        return overall_quality, metrics


if __name__ == "__main__":
    # Test with simulated data
    from data_simulator import DataSimulator
    
    simulator = DataSimulator(duration=10.0)
    sensor_data = simulator.generate_complete_dataset("changing")
    
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    print(f"Transformed {len(results)} samples")
    print(f"Average quality score: {np.mean([r.quality_score for r in results]):.3f}")