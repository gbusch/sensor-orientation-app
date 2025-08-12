# Usage Examples

## Quick Start

### 1. Run Interactive Dashboard
```bash
python src/main.py dashboard
```
Then open http://localhost:52739 in your browser.

### 2. Generate Sample Data
```bash
# Generate 60 seconds of data with changing phone orientation
python src/main.py generate --duration 60 --orientation changing

# Generate data with fixed orientation (dashboard mount)
python src/main.py generate --duration 30 --orientation fixed

# Generate data with slowly changing orientation
python src/main.py generate --duration 45 --orientation slowly_changing
```

### 3. Run Batch Analysis
```bash
python src/main.py batch
```
This compares all three orientation scenarios.

## Programmatic Usage

### Basic Data Generation and Transformation

```python
from src.data_simulator import DataSimulator
from src.transformation import VehicleFrameTransformer
from src.quality_metrics import QualityAssessment

# Generate sensor data
simulator = DataSimulator(duration=30.0, accel_freq=10.0, gps_freq=1.0)
sensor_data = simulator.generate_complete_dataset("changing")

# Transform to vehicle frame
transformer = VehicleFrameTransformer()
results = transformer.transform_sensor_data(sensor_data)

# Assess quality
assessor = QualityAssessment()
quality_report = assessor.assess_transformation_quality(sensor_data, results)

print(f"Overall quality: {quality_report.overall_score:.3f}")
```

### Real-time Processing Simulation

```python
from src.quality_metrics import RealTimeQualityMonitor

monitor = RealTimeQualityMonitor(window_size=20)

for sensor_sample, result in zip(sensor_data, results):
    metrics = monitor.update(sensor_sample, result)
    print(f"Current quality: {metrics['current_quality']:.3f}, "
          f"Status: {metrics['status']}")
```

### Custom Data Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract vehicle accelerations
timestamps = [r.timestamp for r in results]
forward_accel = [r.vehicle_acceleration[0] for r in results]
lateral_accel = [r.vehicle_acceleration[1] for r in results]
quality_scores = [r.quality_score for r in results]

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(timestamps, forward_accel, label='Forward', color='red')
ax1.plot(timestamps, lateral_accel, label='Lateral', color='green')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.set_title('Vehicle Frame Acceleration')
ax1.legend()
ax1.grid(True)

ax2.plot(timestamps, quality_scores, label='Quality Score', color='blue')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Quality Score')
ax2.set_title('Transformation Quality')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Dashboard Features

### Navigation
- **Raw Sensor Data**: View accelerometer, gyroscope, and GPS measurements
- **Transformation Results**: See vehicle frame accelerations and estimated orientations
- **Quality Assessment**: Radar chart and detailed metrics
- **GPS Trajectory**: Vehicle path with speed visualization

### Interactive Controls
- **Duration Slider**: Adjust simulation length (10-120 seconds)
- **Orientation Type**: Choose phone placement scenario
- **Generate New Data**: Create fresh dataset with current parameters
- **Export Results**: Save data and results to JSON files

### Quality Metrics Interpretation

**Overall Quality Score (0-1)**:
- 0.9-1.0: Excellent transformation quality
- 0.7-0.9: Good quality, minor issues
- 0.5-0.7: Fair quality, some concerns
- 0.0-0.5: Poor quality, significant problems

**Component Scores**:
- **Temporal Consistency**: Smoothness of orientation and acceleration changes
- **Physical Plausibility**: Realistic vehicle motion patterns
- **Sensor Reliability**: GPS accuracy and sensor noise levels
- **Transformation Stability**: Consistency of quality metrics over time

## Data Format

### Sensor Data Structure
```json
{
  "timestamp": 1.5,
  "accelerometer": [2.1, -0.5, 9.2],
  "gyroscope": [0.02, -0.01, 0.15],
  "gps_position": [37.7749, -122.4194, 100.0],
  "gps_velocity": [15.2, 8.3, 0.0],
  "gps_accuracy": 4.2
}
```

### Transformation Results Structure
```json
{
  "timestamp": 1.5,
  "vehicle_acceleration": [2.8, -1.2, 0.1],
  "estimated_orientation": [0.15, 0.08, 0.75],
  "quality_score": 0.85,
  "confidence_metrics": {
    "gps_quality": 0.8,
    "motion_quality": 0.9,
    "gravity_consistency": 0.85,
    "orientation_stability": 0.88,
    "gyro_quality": 0.82
  }
}
```

## Performance Optimization

### For Large Datasets
```python
# Process data in chunks for memory efficiency
chunk_size = 1000
for i in range(0, len(sensor_data), chunk_size):
    chunk = sensor_data[i:i+chunk_size]
    chunk_results = transformer.transform_sensor_data(chunk)
    # Process chunk_results...
```

### Real-time Processing
```python
# Initialize transformer once
transformer = VehicleFrameTransformer()

# Process individual samples
for sensor_sample in sensor_stream:
    result = transformer._transform_single_sample(sensor_sample, 0)
    # Use result immediately...
```

## Troubleshooting

### Common Issues

**Low Quality Scores**:
- Check GPS accuracy (should be < 10m)
- Ensure vehicle is moving (> 2 m/s for reliable heading)
- Verify sensor data is reasonable

**Unrealistic Accelerations**:
- Check orientation estimation parameters
- Verify gravity compensation
- Consider sensor calibration

**Dashboard Not Loading**:
- Ensure all dependencies are installed
- Check port 52739 is available
- Try different browser

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python src/main.py dashboard --debug
```

## Integration Examples

### With Vehicle CAN Data
```python
# Combine with vehicle speed/steering data
def enhance_with_can_data(results, can_data):
    for result, can_sample in zip(results, can_data):
        # Validate against vehicle speed
        gps_speed = np.linalg.norm(sensor_data.gps_velocity[:2])
        can_speed = can_sample['vehicle_speed']
        speed_consistency = 1.0 - abs(gps_speed - can_speed) / max(can_speed, 1.0)
        
        # Update quality score
        result.quality_score *= speed_consistency
```

### With Map Data
```python
# Use road geometry for validation
def validate_with_map(results, map_data):
    for result in results:
        # Check if lateral acceleration matches road curvature
        expected_lateral = calculate_expected_lateral_accel(
            result.timestamp, map_data
        )
        actual_lateral = abs(result.vehicle_acceleration[1])
        
        # Adjust quality based on consistency
        consistency = 1.0 - abs(expected_lateral - actual_lateral) / 5.0
        result.confidence_metrics['map_consistency'] = max(0, consistency)
```