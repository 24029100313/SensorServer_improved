## Visual-Inertial Odometry (VIO) Implementation

This project also includes two Visual-Inertial Odometry (VIO) system implementations for integrating IMU data, gyroscope data, and video streams to estimate camera motion trajectories.

### What is VIO?

Visual-Inertial Odometry (VIO) is a positioning technology that combines visual information with Inertial Measurement Unit (IMU) data to achieve precise position tracking in environments without GPS. VIO obtains more accurate position estimates by fusing two complementary sensor data sources:

- **Visual Component**: Estimates camera motion by analyzing feature point movements across consecutive video frames
- **Inertial Component**: Uses IMU (accelerometer and gyroscope) data to predict short-term camera motion

### Two VIO Implementations

This project provides two different VIO implementations:

1. **Simple VIO Implementation** (`simple_vio.py`)
   - Basic visual-inertial fusion
   - Uses ORB features and simple weighted fusion strategy

2. **VINS-Mono Architecture-based VIO System** (`vins_inspired/`)
   - References the architecture of the VINS-Mono project from Hong Kong University of Science and Technology
   - Includes feature tracking, IMU preintegration, initialization, sliding window optimization, and other modules
   - More comprehensive state estimation and fusion strategy

### Data Collection

Before using the VIO system, you need to collect sensor data:

```bash
python data_collector.py
```

This script will guide you through collecting data for six different motion patterns:
- Positive and negative movement along the x-axis
- Positive and negative movement along the y-axis
- Positive and negative movement along the z-axis
- Rotation around the x-axis
- Rotation around the y-axis
- Rotation around the z-axis

Each motion pattern will collect IMU data, gyroscope data, and video frames, saving them in a structured directory with detailed metadata.

### Running the VIO System

After collecting the data, you can choose to run either VIO system:

```bash
# Run the simple VIO implementation
python simple_vio.py

# Run the VINS-Mono architecture-based VIO system
python run_vins_inspired_vio.py
```

The program will automatically process the most recently collected data and generate trajectories and visualization results for each motion type.

### Output Results

The VIO system generates the following outputs:

1. **Trajectory Plots**: Displays the estimated 3D camera motion trajectory
2. **Visualization Videos**: Videos containing feature tracking and state information
3. **Trajectory Data**: Saved trajectory coordinate data
4. **HTML Report**: Summary of processing results for all motion types (VINS architecture version)

All results are saved in the `vio_results` subdirectory for the corresponding motion type.

### Technical Details

The VINS architecture VIO system includes the following core modules:

1. **Feature Tracker** (`feature_tracker.py`)
   - Tracks feature points using optical flow
   - Manages feature point IDs and lifecycles

2. **IMU Preintegration** (`imu_preintegration.py`)
   - Performs IMU data preintegration
   - Handles bias correction

3. **Initializer** (`initializer.py`)
   - Performs visual SFM initialization
   - Estimates gravity direction and scale

4. **Sliding Window Optimizer** (`sliding_window_optimizer.py`)
   - Maintains sliding window state
   - Fuses visual and IMU constraints

5. **VIO System** (`vio_system.py`)
   - Integrates all modules
   - Processes sensor data

### System Requirements

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- Matplotlib
