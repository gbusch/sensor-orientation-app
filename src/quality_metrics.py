"""
Quality assessment and metrics for transformation results.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformation import TransformationResult
from data_simulator import SensorData


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    temporal_consistency: float
    physical_plausibility: float
    sensor_reliability: float
    transformation_stability: float
    detailed_metrics: Dict[str, float]
    recommendations: List[str]


class QualityAssessment:
    """Advanced quality assessment for transformation results"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def assess_transformation_quality(self, sensor_data: List[SensorData], 
                                    results: List[TransformationResult]) -> QualityReport:
        """Comprehensive quality assessment of transformation results"""
        
        if len(results) < self.window_size:
            return self._create_insufficient_data_report()
        
        # Calculate various quality metrics
        temporal_consistency = self._assess_temporal_consistency(results)
        physical_plausibility = self._assess_physical_plausibility(results)
        sensor_reliability = self._assess_sensor_reliability(sensor_data)
        transformation_stability = self._assess_transformation_stability(results)
        
        # Detailed metrics
        detailed_metrics = self._calculate_detailed_metrics(sensor_data, results)
        
        # Overall score (weighted combination)
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = [temporal_consistency, physical_plausibility, sensor_reliability, transformation_stability]
        overall_score = np.average(scores, weights=weights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scores, detailed_metrics)
        
        return QualityReport(
            overall_score=overall_score,
            temporal_consistency=temporal_consistency,
            physical_plausibility=physical_plausibility,
            sensor_reliability=sensor_reliability,
            transformation_stability=transformation_stability,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations
        )
    
    def _assess_temporal_consistency(self, results: List[TransformationResult]) -> float:
        """Assess temporal consistency of transformations"""
        
        # Check for smooth changes in orientation
        orientations = np.array([r.estimated_orientation for r in results])
        orientation_changes = np.diff(orientations, axis=0)
        orientation_smoothness = 1.0 - np.mean(np.linalg.norm(orientation_changes, axis=1)) / np.pi
        orientation_smoothness = max(0.0, min(1.0, orientation_smoothness))
        
        # Check for smooth changes in acceleration
        accelerations = np.array([r.vehicle_acceleration for r in results])
        accel_changes = np.diff(accelerations, axis=0)
        accel_smoothness = 1.0 - np.mean(np.linalg.norm(accel_changes, axis=1)) / 20.0
        accel_smoothness = max(0.0, min(1.0, accel_smoothness))
        
        # Check quality score consistency
        quality_scores = [r.quality_score for r in results]
        quality_std = np.std(quality_scores)
        quality_consistency = 1.0 - min(1.0, quality_std * 2)
        
        return np.mean([orientation_smoothness, accel_smoothness, quality_consistency])
    
    def _assess_physical_plausibility(self, results: List[TransformationResult]) -> float:
        """Assess physical plausibility of results"""
        
        scores = []
        
        # Check acceleration magnitudes (should be reasonable for vehicle motion)
        accelerations = np.array([r.vehicle_acceleration for r in results])
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Reasonable vehicle accelerations: -10 to +10 m/s²
        reasonable_accel = np.mean((accel_magnitudes >= 0) & (accel_magnitudes <= 15))
        scores.append(reasonable_accel)
        
        # Check for predominant forward/backward acceleration (typical for vehicles)
        forward_accel = accelerations[:, 0]  # Forward component
        lateral_accel = accelerations[:, 1]   # Lateral component
        
        forward_dominance = np.mean(np.abs(forward_accel) >= np.abs(lateral_accel) * 0.5)
        scores.append(forward_dominance)
        
        # Check orientation ranges (should be within reasonable bounds)
        orientations = np.array([r.estimated_orientation for r in results])
        
        # Roll and pitch should typically be within ±45 degrees for normal driving
        roll_reasonable = np.mean(np.abs(orientations[:, 0]) <= np.pi/4)
        pitch_reasonable = np.mean(np.abs(orientations[:, 1]) <= np.pi/4)
        scores.extend([roll_reasonable, pitch_reasonable])
        
        return np.mean(scores)
    
    def _assess_sensor_reliability(self, sensor_data: List[SensorData]) -> float:
        """Assess reliability of sensor measurements"""
        
        scores = []
        
        # GPS accuracy assessment
        gps_accuracies = [data.gps_accuracy for data in sensor_data]
        avg_gps_accuracy = np.mean(gps_accuracies)
        gps_score = min(1.0, 10.0 / max(avg_gps_accuracy, 1.0))
        scores.append(gps_score)
        
        # GPS velocity consistency
        gps_velocities = np.array([data.gps_velocity for data in sensor_data])
        velocity_changes = np.diff(gps_velocities, axis=0)
        velocity_smoothness = 1.0 - np.mean(np.linalg.norm(velocity_changes, axis=1)) / 10.0
        velocity_smoothness = max(0.0, min(1.0, velocity_smoothness))
        scores.append(velocity_smoothness)
        
        # Accelerometer noise assessment
        accelerometers = np.array([data.accelerometer for data in sensor_data])
        accel_noise = np.std(accelerometers, axis=0)
        accel_score = 1.0 - min(1.0, np.mean(accel_noise) / 2.0)
        scores.append(accel_score)
        
        # Gyroscope noise assessment
        gyroscopes = np.array([data.gyroscope for data in sensor_data])
        gyro_noise = np.std(gyroscopes, axis=0)
        gyro_score = 1.0 - min(1.0, np.mean(gyro_noise) / 1.0)
        scores.append(gyro_score)
        
        return np.mean(scores)
    
    def _assess_transformation_stability(self, results: List[TransformationResult]) -> float:
        """Assess stability of transformation process"""
        
        # Quality score stability
        quality_scores = [r.quality_score for r in results]
        quality_mean = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        quality_stability = 1.0 - min(1.0, quality_std / max(quality_mean, 0.1))
        
        # Confidence metrics stability
        confidence_keys = results[0].confidence_metrics.keys()
        confidence_stabilities = []
        
        for key in confidence_keys:
            values = [r.confidence_metrics[key] for r in results]
            std_val = np.std(values)
            mean_val = np.mean(values)
            stability = 1.0 - min(1.0, std_val / max(mean_val, 0.1))
            confidence_stabilities.append(stability)
        
        confidence_stability = np.mean(confidence_stabilities)
        
        return np.mean([quality_stability, confidence_stability])
    
    def _calculate_detailed_metrics(self, sensor_data: List[SensorData], 
                                  results: List[TransformationResult]) -> Dict[str, float]:
        """Calculate detailed quality metrics"""
        
        metrics = {}
        
        # Transformation metrics
        quality_scores = [r.quality_score for r in results]
        metrics['mean_quality_score'] = np.mean(quality_scores)
        metrics['quality_score_std'] = np.std(quality_scores)
        metrics['min_quality_score'] = np.min(quality_scores)
        
        # Orientation metrics
        orientations = np.array([r.estimated_orientation for r in results])
        metrics['mean_roll'] = np.mean(np.abs(orientations[:, 0]))
        metrics['mean_pitch'] = np.mean(np.abs(orientations[:, 1]))
        metrics['mean_yaw'] = np.mean(np.abs(orientations[:, 2]))
        metrics['orientation_stability'] = 1.0 - np.mean(np.std(orientations, axis=0)) / np.pi
        
        # Acceleration metrics
        accelerations = np.array([r.vehicle_acceleration for r in results])
        metrics['mean_forward_accel'] = np.mean(accelerations[:, 0])
        metrics['mean_lateral_accel'] = np.mean(np.abs(accelerations[:, 1]))
        metrics['mean_vertical_accel'] = np.mean(np.abs(accelerations[:, 2]))
        metrics['max_acceleration'] = np.max(np.linalg.norm(accelerations, axis=1))
        
        # GPS metrics
        gps_accuracies = [data.gps_accuracy for data in sensor_data]
        metrics['mean_gps_accuracy'] = np.mean(gps_accuracies)
        metrics['gps_accuracy_std'] = np.std(gps_accuracies)
        
        gps_speeds = [np.linalg.norm(data.gps_velocity[:2]) for data in sensor_data]
        metrics['mean_speed'] = np.mean(gps_speeds)
        metrics['max_speed'] = np.max(gps_speeds)
        
        # Sensor noise metrics
        accelerometers = np.array([data.accelerometer for data in sensor_data])
        metrics['accelerometer_noise'] = np.mean(np.std(accelerometers, axis=0))
        
        gyroscopes = np.array([data.gyroscope for data in sensor_data])
        metrics['gyroscope_noise'] = np.mean(np.std(gyroscopes, axis=0))
        
        return metrics
    
    def _generate_recommendations(self, scores: List[float], 
                                detailed_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        
        recommendations = []
        temporal_consistency, physical_plausibility, sensor_reliability, transformation_stability = scores
        
        if temporal_consistency < 0.7:
            recommendations.append("Consider increasing Kalman filter process noise to improve temporal smoothing")
            recommendations.append("Check for sudden orientation changes that may indicate phone movement")
        
        if physical_plausibility < 0.6:
            recommendations.append("Review orientation estimation algorithm - results may not be physically realistic")
            recommendations.append("Consider constraining orientation estimates to reasonable vehicle motion ranges")
        
        if sensor_reliability < 0.6:
            recommendations.append("GPS accuracy is poor - consider using additional sensors or filtering")
            recommendations.append("High sensor noise detected - check sensor calibration")
        
        if transformation_stability < 0.7:
            recommendations.append("Transformation quality varies significantly - review estimation parameters")
        
        if detailed_metrics['mean_gps_accuracy'] > 10.0:
            recommendations.append("GPS accuracy is poor (>10m) - results may be unreliable")
        
        if detailed_metrics['max_acceleration'] > 20.0:
            recommendations.append("Detected very high accelerations - check for sensor artifacts")
        
        if detailed_metrics['mean_speed'] < 2.0:
            recommendations.append("Low vehicle speed detected - orientation estimation may be less reliable")
        
        if not recommendations:
            recommendations.append("Transformation quality is good - no specific recommendations")
        
        return recommendations
    
    def _create_insufficient_data_report(self) -> QualityReport:
        """Create report for insufficient data"""
        return QualityReport(
            overall_score=0.0,
            temporal_consistency=0.0,
            physical_plausibility=0.0,
            sensor_reliability=0.0,
            transformation_stability=0.0,
            detailed_metrics={},
            recommendations=["Insufficient data for quality assessment - need more samples"]
        )


class RealTimeQualityMonitor:
    """Real-time quality monitoring for streaming data"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.recent_results = []
        self.recent_sensor_data = []
        
    def update(self, sensor_data: SensorData, result: TransformationResult) -> Dict[str, float]:
        """Update with new data and return current quality metrics"""
        
        self.recent_sensor_data.append(sensor_data)
        self.recent_results.append(result)
        
        # Keep only recent data
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)
            self.recent_sensor_data.pop(0)
        
        if len(self.recent_results) < 5:  # Need minimum data
            return {'current_quality': result.quality_score, 'status': 'insufficient_data'}
        
        # Calculate rolling metrics
        quality_scores = [r.quality_score for r in self.recent_results]
        orientations = np.array([r.estimated_orientation for r in self.recent_results])
        
        metrics = {
            'current_quality': result.quality_score,
            'mean_quality': np.mean(quality_scores),
            'quality_trend': self._calculate_trend(quality_scores),
            'orientation_stability': 1.0 - np.mean(np.std(orientations, axis=0)) / np.pi,
            'gps_accuracy': sensor_data.gps_accuracy,
            'status': self._determine_status(quality_scores[-1], np.mean(quality_scores))
        }
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in quality scores (-1 to 1, negative is declining)"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return np.clip(slope * 10, -1, 1)  # Scale and clip
    
    def _determine_status(self, current_quality: float, mean_quality: float) -> str:
        """Determine current status"""
        if current_quality > 0.8 and mean_quality > 0.7:
            return 'excellent'
        elif current_quality > 0.6 and mean_quality > 0.5:
            return 'good'
        elif current_quality > 0.4 and mean_quality > 0.3:
            return 'fair'
        else:
            return 'poor'


if __name__ == "__main__":
    # Test quality assessment
    from data_simulator import DataSimulator
    from transformation import VehicleFrameTransformer
    
    simulator = DataSimulator(duration=20.0)
    sensor_data = simulator.generate_complete_dataset("changing")
    
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    assessor = QualityAssessment()
    report = assessor.assess_transformation_quality(sensor_data, results)
    
    print(f"Overall Quality Score: {report.overall_score:.3f}")
    print(f"Temporal Consistency: {report.temporal_consistency:.3f}")
    print(f"Physical Plausibility: {report.physical_plausibility:.3f}")
    print(f"Sensor Reliability: {report.sensor_reliability:.3f}")
    print(f"Transformation Stability: {report.transformation_stability:.3f}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")