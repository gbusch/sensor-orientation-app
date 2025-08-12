"""
Data simulator for generating realistic mobile phone sensor data in a vehicle.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import json


@dataclass
class SensorData:
    """Container for sensor measurements"""
    timestamp: float
    accelerometer: np.ndarray  # [ax, ay, az] in phone frame (m/s²)
    gyroscope: np.ndarray      # [gx, gy, gz] in phone frame (rad/s)
    gps_position: np.ndarray   # [lat, lon, alt] (degrees, degrees, meters)
    gps_velocity: np.ndarray   # [vx, vy, vz] in earth frame (m/s)
    gps_accuracy: float        # GPS accuracy in meters


class VehicleMotionSimulator:
    """Simulates realistic vehicle motion patterns"""
    
    def __init__(self, duration: float = 60.0, dt: float = 0.1):
        self.duration = duration
        self.dt = dt
        self.time_steps = int(duration / dt)
        
    def generate_vehicle_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic vehicle trajectory with turns and acceleration changes"""
        t = np.linspace(0, self.duration, self.time_steps)
        
        # Create a path with turns and speed changes
        # Vehicle follows a curved path with varying speed
        speed_profile = 15 + 10 * np.sin(0.1 * t) + 5 * np.sin(0.3 * t)  # 10-30 m/s
        speed_profile = np.maximum(speed_profile, 5)  # Minimum 5 m/s
        
        # Heading changes (turns)
        heading_rate = 0.05 * np.sin(0.2 * t) + 0.02 * np.sin(0.5 * t)  # rad/s
        heading = np.cumsum(heading_rate * self.dt)
        
        # Calculate vehicle acceleration in vehicle frame
        speed_change = np.gradient(speed_profile, self.dt)
        lateral_accel = speed_profile * heading_rate
        
        # Vehicle acceleration in vehicle frame [forward, left, up]
        vehicle_accel = np.column_stack([
            speed_change,  # Forward acceleration
            lateral_accel,  # Lateral acceleration
            np.zeros_like(t)  # Vertical acceleration (assume flat road)
        ])
        
        # Vehicle velocity in earth frame
        vx = speed_profile * np.cos(heading)
        vy = speed_profile * np.sin(heading)
        vz = np.zeros_like(t)
        vehicle_velocity = np.column_stack([vx, vy, vz])
        
        return t, vehicle_accel, vehicle_velocity, heading


class PhoneOrientationSimulator:
    """Simulates phone orientation relative to vehicle"""
    
    def __init__(self, time_steps: int):
        self.time_steps = time_steps
        
    def generate_orientation_sequence(self, orientation_type: str = "changing") -> np.ndarray:
        """Generate phone orientation relative to vehicle over time"""
        
        if orientation_type == "fixed":
            # Fixed orientation (e.g., phone in dashboard mount)
            roll = np.full(self.time_steps, np.pi/6)    # 30 degrees
            pitch = np.full(self.time_steps, np.pi/12)  # 15 degrees  
            yaw = np.full(self.time_steps, np.pi/4)     # 45 degrees
            
        elif orientation_type == "slowly_changing":
            # Slowly changing orientation (e.g., phone sliding on seat)
            t = np.linspace(0, 1, self.time_steps)
            roll = np.pi/6 + 0.2 * np.sin(2 * np.pi * t * 0.1)
            pitch = np.pi/12 + 0.1 * np.sin(2 * np.pi * t * 0.15)
            yaw = np.pi/4 + 0.3 * np.sin(2 * np.pi * t * 0.05)
            
        else:  # "changing"
            # More dynamic orientation changes (e.g., phone in pocket/hand)
            t = np.linspace(0, 1, self.time_steps)
            roll = np.pi/6 + 0.4 * np.sin(2 * np.pi * t * 0.3) + 0.1 * np.random.randn(self.time_steps)
            pitch = np.pi/12 + 0.3 * np.sin(2 * np.pi * t * 0.4) + 0.1 * np.random.randn(self.time_steps)
            yaw = np.pi/4 + 0.5 * np.sin(2 * np.pi * t * 0.2) + 0.15 * np.random.randn(self.time_steps)
        
        return np.column_stack([roll, pitch, yaw])


class SensorNoiseSimulator:
    """Adds realistic noise to sensor measurements"""
    
    @staticmethod
    def add_accelerometer_noise(accel: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Add noise to accelerometer data"""
        noise = np.random.normal(0, noise_std, accel.shape)
        return accel + noise
    
    @staticmethod
    def add_gyroscope_noise(gyro: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Add noise to gyroscope data"""
        noise = np.random.normal(0, noise_std, gyro.shape)
        bias = np.random.normal(0, 0.01, (1, 3))  # Small bias
        return gyro + noise + bias
    
    @staticmethod
    def add_gps_noise(position: np.ndarray, velocity: np.ndarray, 
                     pos_accuracy: float = 3.0, vel_accuracy: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        """Add noise to GPS data"""
        pos_noise = np.random.normal(0, pos_accuracy/111000, position.shape)  # Convert m to degrees
        vel_noise = np.random.normal(0, vel_accuracy, velocity.shape)
        
        # Simulate varying GPS accuracy
        accuracy = pos_accuracy + np.random.exponential(2.0)
        
        return position + pos_noise, velocity + vel_noise, accuracy


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create rotation matrix from Euler angles (ZYX convention)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R


class DataSimulator:
    """Main class for generating complete sensor data simulation"""
    
    def __init__(self, duration: float = 60.0, accel_freq: float = 10.0, gps_freq: float = 1.0):
        self.duration = duration
        self.accel_freq = accel_freq
        self.gps_freq = gps_freq
        self.accel_dt = 1.0 / accel_freq
        self.gps_dt = 1.0 / gps_freq
        
    def generate_complete_dataset(self, orientation_type: str = "changing") -> List[SensorData]:
        """Generate complete sensor dataset"""
        
        # Generate vehicle motion at high frequency for accurate simulation
        motion_sim = VehicleMotionSimulator(self.duration, self.accel_dt)
        t_fine, vehicle_accel, vehicle_velocity, vehicle_heading = motion_sim.generate_vehicle_trajectory()
        
        # Generate phone orientation
        orientation_sim = PhoneOrientationSimulator(len(t_fine))
        phone_orientation = orientation_sim.generate_orientation_sequence(orientation_type)
        
        # Generate GPS trajectory (lower frequency)
        gps_indices = np.arange(0, len(t_fine), int(self.accel_freq / self.gps_freq))
        
        # Starting GPS position (arbitrary location)
        start_lat, start_lon = 37.7749, -122.4194  # San Francisco
        gps_positions = []
        current_lat, current_lon = start_lat, start_lon
        
        for i in gps_indices:
            if i < len(vehicle_velocity):
                # Convert velocity to lat/lon changes
                dt = self.gps_dt
                dlat = vehicle_velocity[i, 1] * dt / 111000  # m to degrees
                dlon = vehicle_velocity[i, 0] * dt / (111000 * np.cos(np.radians(current_lat)))
                current_lat += dlat
                current_lon += dlon
                gps_positions.append([current_lat, current_lon, 100.0])  # Assume 100m altitude
        
        gps_positions = np.array(gps_positions)
        
        # Generate sensor data
        sensor_data = []
        noise_sim = SensorNoiseSimulator()
        
        for i, t in enumerate(t_fine):
            # Transform vehicle acceleration to phone frame
            R_vehicle_to_phone = rotation_matrix_from_euler(*phone_orientation[i])
            
            # Add gravity in phone frame
            gravity_earth = np.array([0, 0, -9.81])  # Earth frame gravity
            gravity_phone = R_vehicle_to_phone @ gravity_earth
            
            # Vehicle acceleration in phone frame
            vehicle_accel_phone = R_vehicle_to_phone @ vehicle_accel[i]
            
            # Total acceleration in phone frame (vehicle + gravity)
            phone_accel = vehicle_accel_phone - gravity_phone  # Accelerometer measures specific force
            
            # Add noise
            phone_accel_noisy = noise_sim.add_accelerometer_noise(phone_accel)
            
            # Gyroscope (angular velocity of phone relative to earth)
            # For simplicity, assume phone rotates with vehicle plus some additional rotation
            vehicle_angular_vel = np.array([0, 0, np.gradient(vehicle_heading)[i] / self.accel_dt])
            phone_orientation_rate = np.gradient(phone_orientation, axis=0)[i] / self.accel_dt
            phone_gyro = vehicle_angular_vel + phone_orientation_rate
            phone_gyro_noisy = noise_sim.add_gyroscope_noise(phone_gyro.reshape(1, -1)).flatten()
            
            # GPS data (at lower frequency)
            gps_idx = min(i // int(self.accel_freq / self.gps_freq), len(gps_positions) - 1)
            gps_pos = gps_positions[gps_idx]
            gps_vel = vehicle_velocity[i] if i < len(vehicle_velocity) else vehicle_velocity[-1]
            
            # Add GPS noise
            gps_pos_noisy, gps_vel_noisy, gps_accuracy = noise_sim.add_gps_noise(gps_pos, gps_vel)
            
            sensor_data.append(SensorData(
                timestamp=t,
                accelerometer=phone_accel_noisy,
                gyroscope=phone_gyro_noisy,
                gps_position=gps_pos_noisy,
                gps_velocity=gps_vel_noisy,
                gps_accuracy=gps_accuracy
            ))
        
        return sensor_data
    
    def save_dataset(self, sensor_data: List[SensorData], filename: str):
        """Save dataset to JSON file"""
        data_dict = {
            'metadata': {
                'duration': self.duration,
                'accel_freq': self.accel_freq,
                'gps_freq': self.gps_freq,
                'num_samples': len(sensor_data)
            },
            'data': []
        }
        
        for sample in sensor_data:
            data_dict['data'].append({
                'timestamp': sample.timestamp,
                'accelerometer': sample.accelerometer.tolist(),
                'gyroscope': sample.gyroscope.tolist(),
                'gps_position': sample.gps_position.tolist(),
                'gps_velocity': sample.gps_velocity.tolist(),
                'gps_accuracy': sample.gps_accuracy
            })
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def load_dataset(self, filename: str) -> List[SensorData]:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            data_dict = json.load(f)
        
        sensor_data = []
        for sample in data_dict['data']:
            sensor_data.append(SensorData(
                timestamp=sample['timestamp'],
                accelerometer=np.array(sample['accelerometer']),
                gyroscope=np.array(sample['gyroscope']),
                gps_position=np.array(sample['gps_position']),
                gps_velocity=np.array(sample['gps_velocity']),
                gps_accuracy=sample['gps_accuracy']
            ))
        
        return sensor_data


if __name__ == "__main__":
    # Generate sample data
    simulator = DataSimulator(duration=30.0)
    data = simulator.generate_complete_dataset("changing")
    simulator.save_dataset(data, "../data/sample_data.json")
    print(f"Generated {len(data)} sensor samples")