#!/usr/bin/env python3
"""
System test script to verify all components work correctly.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from data_simulator import DataSimulator, SensorData
from transformation import VehicleFrameTransformer, TransformationResult
from quality_metrics import QualityAssessment, RealTimeQualityMonitor


def test_data_simulation():
    """Test data simulation functionality"""
    print("Testing data simulation...")
    
    simulator = DataSimulator(duration=5.0, accel_freq=10.0, gps_freq=1.0)
    
    # Test different orientation types
    for orientation_type in ['fixed', 'slowly_changing', 'changing']:
        print(f"  Testing {orientation_type} orientation...")
        data = simulator.generate_complete_dataset(orientation_type)
        
        assert len(data) == 50, f"Expected 50 samples, got {len(data)}"
        assert all(isinstance(sample, SensorData) for sample in data), "Invalid data type"
        
        # Check data ranges
        for sample in data[:5]:  # Check first few samples
            assert sample.accelerometer.shape == (3,), "Invalid accelerometer shape"
            assert sample.gyroscope.shape == (3,), "Invalid gyroscope shape"
            assert sample.gps_position.shape == (3,), "Invalid GPS position shape"
            assert sample.gps_velocity.shape == (3,), "Invalid GPS velocity shape"
            assert sample.gps_accuracy > 0, "Invalid GPS accuracy"
    
    print("  ✓ Data simulation tests passed")


def test_transformation():
    """Test transformation functionality"""
    print("Testing transformation...")
    
    # Generate test data
    simulator = DataSimulator(duration=3.0)
    sensor_data = simulator.generate_complete_dataset("changing")
    
    # Transform data
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    assert len(results) == len(sensor_data), "Mismatch in result count"
    assert all(isinstance(result, TransformationResult) for result in results), "Invalid result type"
    
    # Check result properties
    for result in results[:5]:
        assert result.vehicle_acceleration.shape == (3,), "Invalid acceleration shape"
        assert result.estimated_orientation.shape == (3,), "Invalid orientation shape"
        assert 0 <= result.quality_score <= 1, f"Invalid quality score: {result.quality_score}"
        assert isinstance(result.confidence_metrics, dict), "Invalid confidence metrics"
    
    print("  ✓ Transformation tests passed")


def test_quality_assessment():
    """Test quality assessment functionality"""
    print("Testing quality assessment...")
    
    # Generate test data
    simulator = DataSimulator(duration=10.0)
    sensor_data = simulator.generate_complete_dataset("fixed")
    
    # Transform data
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    # Assess quality
    assessor = QualityAssessment()
    report = assessor.assess_transformation_quality(sensor_data, results)
    
    assert 0 <= report.overall_score <= 1, "Invalid overall score"
    assert 0 <= report.temporal_consistency <= 1, "Invalid temporal consistency"
    assert 0 <= report.physical_plausibility <= 1, "Invalid physical plausibility"
    assert 0 <= report.sensor_reliability <= 1, "Invalid sensor reliability"
    assert 0 <= report.transformation_stability <= 1, "Invalid transformation stability"
    assert isinstance(report.detailed_metrics, dict), "Invalid detailed metrics"
    assert isinstance(report.recommendations, list), "Invalid recommendations"
    
    print("  ✓ Quality assessment tests passed")


def test_real_time_monitoring():
    """Test real-time quality monitoring"""
    print("Testing real-time monitoring...")
    
    # Generate test data
    simulator = DataSimulator(duration=5.0)
    sensor_data = simulator.generate_complete_dataset("slowly_changing")
    
    # Transform data
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    # Test real-time monitor
    monitor = RealTimeQualityMonitor(window_size=10)
    
    for sensor_sample, result in zip(sensor_data, results):
        metrics = monitor.update(sensor_sample, result)
        
        assert isinstance(metrics, dict), "Invalid metrics type"
        assert 'current_quality' in metrics, "Missing current quality"
        assert 'status' in metrics, "Missing status"
        assert metrics['status'] in ['insufficient_data', 'excellent', 'good', 'fair', 'poor'], "Invalid status"
    
    print("  ✓ Real-time monitoring tests passed")


def test_data_export_import():
    """Test data export and import functionality"""
    print("Testing data export/import...")
    
    # Generate and save data
    simulator = DataSimulator(duration=2.0)
    original_data = simulator.generate_complete_dataset("fixed")
    
    test_file = "test_data.json"
    simulator.save_dataset(original_data, test_file)
    
    # Load data back
    loaded_data = simulator.load_dataset(test_file)
    
    assert len(loaded_data) == len(original_data), "Data length mismatch"
    
    # Compare first sample
    orig = original_data[0]
    loaded = loaded_data[0]
    
    assert abs(orig.timestamp - loaded.timestamp) < 1e-6, "Timestamp mismatch"
    assert np.allclose(orig.accelerometer, loaded.accelerometer), "Accelerometer mismatch"
    assert np.allclose(orig.gyroscope, loaded.gyroscope), "Gyroscope mismatch"
    assert np.allclose(orig.gps_position, loaded.gps_position), "GPS position mismatch"
    assert np.allclose(orig.gps_velocity, loaded.gps_velocity), "GPS velocity mismatch"
    assert abs(orig.gps_accuracy - loaded.gps_accuracy) < 1e-6, "GPS accuracy mismatch"
    
    # Clean up
    os.remove(test_file)
    
    print("  ✓ Data export/import tests passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test with very short duration
    simulator = DataSimulator(duration=0.5)
    data = simulator.generate_complete_dataset("fixed")
    assert len(data) > 0, "No data generated for short duration"
    
    # Test transformation with minimal data
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(data)
    assert len(results) == len(data), "Result count mismatch for short data"
    
    # Test quality assessment with insufficient data
    assessor = QualityAssessment()
    report = assessor.assess_transformation_quality(data, results)
    # Should handle gracefully even with insufficient data
    
    print("  ✓ Edge case tests passed")


def run_performance_test():
    """Test performance with larger dataset"""
    print("Testing performance...")
    
    import time
    
    # Generate larger dataset
    simulator = DataSimulator(duration=60.0)  # 1 minute
    start_time = time.time()
    sensor_data = simulator.generate_complete_dataset("changing")
    generation_time = time.time() - start_time
    
    # Transform data
    transformer = VehicleFrameTransformer()
    start_time = time.time()
    results = transformer.transform_sensor_data(sensor_data)
    transformation_time = time.time() - start_time
    
    # Assess quality
    assessor = QualityAssessment()
    start_time = time.time()
    report = assessor.assess_transformation_quality(sensor_data, results)
    assessment_time = time.time() - start_time
    
    print(f"  Performance results for {len(sensor_data)} samples:")
    print(f"    Data generation: {generation_time:.2f}s ({len(sensor_data)/generation_time:.0f} samples/s)")
    print(f"    Transformation: {transformation_time:.2f}s ({len(sensor_data)/transformation_time:.0f} samples/s)")
    print(f"    Quality assessment: {assessment_time:.2f}s")
    print(f"    Overall quality score: {report.overall_score:.3f}")
    
    print("  ✓ Performance test completed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("VEHICLE REFERENCE FRAME TRANSFORMATION - SYSTEM TEST")
    print("=" * 60)
    
    try:
        test_data_simulation()
        test_transformation()
        test_quality_assessment()
        test_real_time_monitoring()
        test_data_export_import()
        test_edge_cases()
        run_performance_test()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - SYSTEM IS WORKING CORRECTLY")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Run the interactive dashboard: python src/main.py dashboard")
        print("2. Open http://localhost:52739 in your browser")
        print("3. Explore different orientation scenarios and quality metrics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)