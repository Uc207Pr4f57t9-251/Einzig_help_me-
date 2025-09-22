# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MicroPython project for ESP32S3 devices that implements an intelligent maimai arcade game queue management system. The system uses accelerometer data from a BNO085 sensor to detect horizontal card movements and automatically track queue positions.

## Core Architecture

### Movement Detection Pipeline
1. **HorizontalMovementDetector** (`movementDetector.py`) - Core algorithm that analyzes accelerometer peaks to detect left/right movements
2. **QueueManager** (`queueManager.py`) - Manages queue position state and persistence to local files
3. **SensorDataStream** (`sensorDataStream.py`) - Abstracts sensor data input for both real hardware and file-based testing

### Key Integration Points
- `movementDetector.py` automatically integrates with `queueManager.py` when `enable_queue=True`
- Movement directions: `forward` (left move) = position-1, `backward` (right move) = position+1
- Queue state persists to `queue_status.txt` with sync status for server communication

### Hardware Integration Files
- `bno08x.py` - BNO085 sensor driver
- `bmp280.py` - BMP280 pressure sensor support
- `ssd1306.py` - OLED display driver
- `boot.py` - MicroPython boot configuration
- `main.py` - Entry point with command-line interface

## Development Commands

### Testing
```bash
# Run comprehensive queue system tests
python testQueue.py

# Run manual interactive queue testing
python testQueue.py manual

# Test movement detection with sensor data
python main.py realtime [speed] [data_file]
python main.py batch [data_file]
```

### Hardware Deployment
- This project uses Pymakr for ESP32S3 deployment (see `pymakr.conf`)
- Upload all `.py` files to the MicroPython device
- The system will auto-start via `boot.py` and `main.py`

## File Dependencies and Data Flow

### Core Data Flow
1. Sensor data → `SensorDataStream` → `HorizontalMovementDetector`
2. Movement detection → `QueueManager` position updates
3. Queue state → local file persistence + server sync status
4. Status reporting → combined movement + queue information

### Important State Files
- `queue_status.txt` - Persistent queue state (position, timing, sync status)
- `test.csv` - Sample accelerometer data for testing

### Testing Strategy
- `testFunctions.py` provides real-time and batch testing modes
- `testQueue.py` includes comprehensive integration tests and manual testing interface
- All tests clean up temporary files automatically

## Key Configuration Parameters

### Movement Detection Tuning
Located in `HorizontalMovementDetector.__init__()`:
- `peak_threshold=1.5` - Minimum acceleration for valid peaks
- `trigger_threshold=0.8` - Threshold to start detection
- `peak_time_window=1.0` - Time window for peak pairing
- `sample_rate=200` - Expected sensor sampling rate

### Queue Management
- Positions start at 1 (front of queue)
- Position 1 blocks forward movement (boundary condition)
- All movements automatically update sync status to pending