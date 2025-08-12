# Vehicle Reference Frame Transformation

This application transforms mobile phone sensor data (accelerometer, gyroscope, GPS) from the phone's reference frame to the vehicle's reference frame, providing quality metrics for each transformation.

## Features

- **Data Transformation**: Converts phone sensor data to vehicle reference frame
- **Quality Metrics**: Provides confidence scores for each transformation
- **Simulated Data**: Generates realistic sensor data for testing
- **Interactive Dashboard**: Real-time visualization of data and transformations
- **Orientation Estimation**: Handles unknown and changing phone orientation

## Architecture

- `src/data_simulator.py`: Generates realistic sensor data
- `src/transformation.py`: Core transformation algorithms
- `src/quality_metrics.py`: Quality assessment functions
- `src/dashboard.py`: Interactive web dashboard
- `src/main.py`: Main application entry point

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

Then open http://localhost:52739 to view the dashboard.

## Technical Approach

1. **Orientation Estimation**: Uses GPS velocity and accelerometer data to estimate vehicle orientation
2. **Kalman Filtering**: Smooths orientation estimates over time
3. **Coordinate Transformation**: Applies rotation matrices to transform accelerometer data
4. **Quality Assessment**: Evaluates transformation confidence based on GPS quality, motion consistency, and sensor noise

## Data Format

- Accelerometer/Gyroscope: 10 Hz sampling rate
- GPS: 1 Hz sampling rate with location and velocity
- All timestamps synchronized