try: 
    import utime
except ImportError:
    import time as utime
from movementDetector import HorizontalMovementDetector
from sensorDataStream import SensorDataStream

def test_realtime_detector(data_file="test.csv", speed_factor=1.0):
    """
    实时模拟传感器数据读取的测试
    
    data_file: 数据文件路径
    speed_factor: 播放速度倍数（1.0=实时，2.0=2倍速，0.5=慢速）
    """
    # 创建检测器
    sample_rate = 200
    detector = HorizontalMovementDetector(
        noise_threshold=0.3,
        peak_threshold=1.5,
        peak_max_threshold=5.0,
        window_size=400,
        peak_time_window=1.0,
        sample_rate=sample_rate,
        ma_window=5,
        trigger_threshold=0.8,
        peak_confirmation_delay=0.3
    )
    
    # 创建数据流
    stream = SensorDataStream(data_file, sample_rate)
    
    print("\n"*2 + "=" * 60)
    print("Real-time Movement Detection Simulation")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Speed factor: {speed_factor}x")
    print(f"Actual interval: {1000/(sample_rate*speed_factor):.1f} ms")
    print("=" * 60 + "\n"*2)
    print("\nStarting real-time simulation...")
    print("-" * 50)
    
    # 统计变量
    samples_processed = 0
    # 两个状态打印时间：激活时(1s)与非激活时(5s)
    last_status_time_active = 0.0
    last_status_time_inactive = 0.0
    active_interval = 1.0
    inactive_interval = 5.0
    
    simulation_start = utime.time()
    
    # 主循环 - 模拟实时数据读取
    while stream.has_more_data():
        # 获取下一个样本
        sample_data = stream.get_next_sample()
        if sample_data is None:
            break
        
        timestamp, ax, ay, az = sample_data
        
        # 模拟实时延迟（根据速度因子调整）
        if speed_factor > 0:
            # 计算应该等待的时间
            expected_time = simulation_start + (samples_processed * stream.interval / speed_factor)
            current_time = utime.time()
            
            if current_time < expected_time:
                sleep_time = expected_time - current_time
                utime.sleep(sleep_time)
        
        # 处理样本
        movement_detected, direction, distance = detector.add_sample(
            ax, ay, az, timestamp
        )
        
        samples_processed += 1
        
        # 检测到移动时的处理
        if movement_detected:
            # 移动信息已在 _register_movement 中打印
            pass
        
        # 定期输出状态：
        # - 当 detector.detection_triggered == True 时，每 active_interval 秒输出一次
        # - 否则每 inactive_interval 秒输出一次
        status = detector.get_status()
        sim_time_str = detector._format_time(timestamp)
        
        
        if status['detection_triggered'] and timestamp - last_status_time_active >= active_interval or (not status['detection_triggered']) and timestamp - last_status_time_inactive >= inactive_interval:
            elapsed_real = utime.time() - simulation_start
            print(f"\n[RT Status] Sim time: {elapsed_real:.1f}s, Real time: {sim_time_str}")
            print(f"  Samples: {samples_processed}, Rate: {sample_rate:.1f} Hz")
            print(f"  Movements: {status['movement_count']} "
                    f"(F: {status['forward_count']}, B: {status['backward_count']})")
            print(f"  Peaks: {status['peaks_confirmed']} confirmed, "
                    f"{status['peaks_pending']} pending")
            print(f"  Detection: {'ACTIVE' if status['detection_triggered'] else 'IDLE'}")
            last_status_time_active = timestamp
            # 同步重置非激活计时，避免刚刚激活又立刻打印非激活信息
            last_status_time_inactive = timestamp

    
    # 最终统计
    simulation_duration = utime.time() - simulation_start
    final_status = detector.get_status()
    
    print("\n"*2 + "=" * 60)
    print("Simulation Complete!")
    print(f"  Total samples: {samples_processed}")
    print(f"  Simulation duration: {simulation_duration:.2f}s")
    print(f"  Effective rate: {samples_processed/simulation_duration:.1f} Hz")
    print(f"\nMovement Statistics:")
    print(f"  Total movements: {final_status['movement_count']}")
    print(f"  Forward: {final_status['forward_count']}")
    print(f"  Backward: {final_status['backward_count']}")
    if final_status['last_distance']:
        print(f"  Last distance: {final_status['last_distance']:.3f} m "
              f"({final_status['last_distance']*100:.1f} cm)")
    print("=" * 60 + "\n"*2)


# 批处理测试（保留原有功能）
def test_batch_detector(data_file="test.csv"):
    """批量处理测试（原有的快速测试模式）"""
    detector = HorizontalMovementDetector(
        noise_threshold=0.3,
        peak_threshold=1.5,
        peak_max_threshold=5.0,
        window_size=400,
        peak_time_window=1.0,
        sample_rate=200,
        ma_window=5,
        trigger_threshold=0.8,
        peak_confirmation_delay=0.3
    )
    
    print("Batch Movement Detector Test")
    print(f"Sample rate: {detector.sample_rate} Hz")
    print("-" * 50)
    
    # 加载测试数据
    try:
        with open(data_file, "r") as f:
            lines = f.readlines()
        test_data = []
        for line in lines:
            parts = line.strip().split(",")
            test_data.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        print(f"Loaded {len(test_data)} samples")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 批处理所有数据
    for i, sample in enumerate(test_data):
        movement_detected, direction, distance = detector.add_sample(
            sample[1], sample[2], sample[3], 
            timestamp=sample[0]
        )
        
        # 简单进度显示
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{len(test_data)} samples...")
    
    # 输出结果
    final_status = detector.get_status()
    print(f"\nTotal movements: {final_status['movement_count']}")
    print(f"Forward: {final_status['forward_count']}, Backward: {final_status['backward_count']}")