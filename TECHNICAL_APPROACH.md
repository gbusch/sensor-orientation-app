# Technical Approach: Vehicle Reference Frame Transformation

## Problem Statement

Transform mobile phone sensor data (accelerometer, gyroscope, GPS) from the phone's reference frame to the vehicle's reference frame, accounting for unknown and changing phone orientation.

## Key Challenges

1. **Unknown Orientation**: Phone orientation relative to vehicle is unknown and may change
2. **Multi-rate Data**: Accelerometer/gyroscope at 10Hz, GPS at 1Hz
3. **Noise and Accuracy**: GPS accuracy varies, sensors have noise
4. **Real-time Processing**: Need quality metrics for each transformation

## Solution Architecture

### 1. Data Simulation (`data_simulator.py`)

**Vehicle Motion Model**:
- Realistic speed profiles (10-30 m/s with variations)
- Curved trajectories with heading changes
- Forward and lateral accelerations based on speed and turning

**Phone Orientation Scenarios**:
- **Fixed**: Dashboard-mounted phone (stable orientation)
- **Slowly Changing**: Phone on seat (gradual orientation drift)
- **Dynamic**: Phone in pocket/hand (frequent orientation changes)

**Sensor Noise Models**:
- Accelerometer: Gaussian noise (σ = 0.1 m/s²)
- Gyroscope: Gaussian noise + bias (σ = 0.05 rad/s)
- GPS: Position accuracy 3-5m, velocity accuracy 0.5 m/s

### 2. Orientation Estimation (`transformation.py`)

**Multi-step Approach**:

1. **Gravity Alignment**: Use accelerometer to estimate roll and pitch
   ```
   roll = atan2(ay, az)
   pitch = atan2(-ax, sqrt(ay² + az²))
   ```

2. **Heading from GPS**: Vehicle heading from GPS velocity
   ```
   heading = atan2(vy_gps, vx_gps)
   ```

3. **Yaw Estimation**: Compare phone and vehicle accelerations
   - Try different yaw angles
   - Find best match between transformed phone acceleration and expected vehicle motion

4. **Kalman Filtering**: Smooth orientation estimates over time
   - State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
   - Constant velocity model for orientation changes

### 3. Coordinate Transformation

**Reference Frames**:
- **Earth Frame**: X=East, Y=North, Z=Up
- **Vehicle Frame**: X=Forward, Y=Left, Z=Up
- **Phone Frame**: X=Right, Y=Up, Z=Out (when held upright)

**Transformation Steps**:
1. Remove gravity from phone accelerometer
2. Apply rotation matrix from phone to vehicle frame
3. Result: acceleration in vehicle coordinates

**Rotation Matrix** (ZYX Euler angles):
```
R = Rz(-yaw) * Ry(-pitch) * Rx(-roll)
```

### 4. Quality Assessment (`quality_metrics.py`)

**Quality Components**:

1. **GPS Quality** (30% weight):
   - Based on GPS accuracy values
   - Higher weight due to importance for orientation estimation

2. **Motion Quality** (25% weight):
   - Vehicle speed (better at higher speeds)
   - Motion consistency

3. **Gravity Consistency** (20% weight):
   - Accelerometer magnitude should be ~9.81 m/s² when stationary
   - Indicates proper gravity compensation

4. **Orientation Stability** (15% weight):
   - Difference between raw and filtered orientation estimates
   - Smooth changes indicate good estimation

5. **Gyroscope Quality** (10% weight):
   - Lower gyroscope noise indicates better sensor conditions

**Real-time Monitoring**:
- Rolling window quality assessment
- Trend analysis
- Status classification (excellent/good/fair/poor)

### 5. Interactive Dashboard (`dashboard.py`)

**Visualization Components**:
- Raw sensor data plots
- Transformation results
- Quality metrics radar chart
- GPS trajectory with speed coloring
- Detailed metrics tables
- Quality recommendations

**Interactive Features**:
- Adjustable simulation parameters
- Real-time data generation
- Export functionality
- Multiple visualization tabs

## Algorithm Performance

**Quality Results by Scenario**:

| Scenario | Overall Quality | Temporal Consistency | Physical Plausibility | Sensor Reliability | Transformation Stability |
|----------|----------------|---------------------|----------------------|-------------------|-------------------------|
| Fixed | 0.955 | 0.985 | 1.000 | 0.852 | 0.982 |
| Slowly Changing | 0.944 | 0.982 | 0.968 | 0.847 | 0.978 |
| Dynamic | 0.785 | 0.921 | 0.735 | 0.586 | 0.892 |

**Key Insights**:
- Fixed orientation provides best transformation quality
- Dynamic orientation is more challenging but still achievable
- GPS quality is the main limiting factor
- Temporal consistency remains high across all scenarios

## Limitations and Future Improvements

**Current Limitations**:
1. Assumes vehicle motion on flat roads (no elevation changes)
2. Simplified yaw estimation (could use magnetometer)
3. GPS-dependent (poor in tunnels, urban canyons)

**Potential Improvements**:
1. **Sensor Fusion**: Add magnetometer for absolute heading
2. **Machine Learning**: Train orientation classifier on labeled data
3. **Map Matching**: Use road geometry to constrain estimates
4. **IMU Integration**: Full 6-DOF orientation tracking
5. **Adaptive Filtering**: Adjust filter parameters based on driving conditions

## Validation Approach

**Simulation-based Validation**:
- Generate ground truth vehicle motion
- Add realistic sensor noise and phone orientation
- Compare transformed results with ground truth
- Assess quality metrics accuracy

**Real-world Validation** (future work):
- Collect data with known phone orientation
- Compare with reference IMU or vehicle CAN data
- Validate in different driving scenarios
- Test with various phone placements