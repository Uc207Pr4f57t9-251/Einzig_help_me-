import math
try: 
    import utime
except ImportError:
    import time as utime

class HorizontalMovementDetector:
    def __init__(self, 
                 noise_threshold=0.3,      # 噪声阈值 (m/s²)
                 peak_threshold=1.5,        # 峰值阈值 (m/s²)
                 peak_max_threshold=5.0,    # 峰值最大阈值 (m/s²)
                 window_size=400,           # 滑动窗口大小（采样点数）
                 peak_time_window=1.0,      # 双峰时间窗口（秒）
                 sample_rate=200,           # 采样率 (Hz)
                 ma_window=5,               # 移动平均窗口
                 trigger_threshold=0.8,     # 触发检测的阈值 (m/s²)
                 peak_confirmation_delay=0.3):  # 峰值确认延迟（秒）
        
        self.noise_threshold = noise_threshold
        self.peak_threshold = peak_threshold
        self.peak_max_threshold = peak_max_threshold
        self.window_size = window_size
        self.peak_time_window = peak_time_window
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.ma_window = ma_window
        self.trigger_threshold = trigger_threshold  # 新增：触发阈值
        self.peak_confirmation_delay = peak_confirmation_delay  # 新增：确认延迟
        
        # 缓冲区
        self.accel_buffer = []
        self.time_buffer = []
        
        # 峰值检测状态
        self.peaks_detected = []
        self.pending_peaks = []  # 新增：待确认的峰值
        
        # 触发状态
        self.detection_triggered = False  # 新增：检测是否被触发
        self.trigger_start_time = None    # 新增：触发开始时间
        self.last_detection_time = 0      # 新增：上次检测时间
        self.detection_interval = 0.1     # 新增：检测间隔（秒）
        
        # 移动检测结果
        self.movement_count = 0
        self.forward_movement_count = 0
        self.backward_movement_count = 0
        self.last_movement_time = None
        self.last_movement_direction = None
        self.last_movement_distance = None  # 新增：记录移动距离
    def _format_time(self, ts):
        """
        将 epoch 时间或浮点时间转换为 "HH:MM:SS" 字符串。
        如果 utime.localtime 可用则使用本地时间；否则以秒数取模转为 hh:mm:ss。
        """
        try:
            # MicroPython/time 模块通常支持 localtime(tuple)
            t = utime.localtime(int(ts))
            return "{:02d}:{:02d}:{:02d}".format(t[3], t[4], t[5])
        except Exception:
            sec = int(ts) % 86400
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            return "{:02d}:{:02d}:{:02d}".format(h, m, s)
        
    def _append_buffer(self, buf, value):
        """向缓冲区添加数据"""
        buf.append(value)
        if len(buf) > self.window_size:
            buf.pop(0)
        
    def add_sample(self, ax, ay, az, timestamp=None):
        """
        添加新的加速度采样数据
        返回: (movement_detected, direction, distance) 元组
        """
        if timestamp is None:
            timestamp = utime.time()
            
        # 保存数据
        self._append_buffer(self.accel_buffer, ax)
        self._append_buffer(self.time_buffer, timestamp)
        
        # 检查是否需要触发检测
        if not self.detection_triggered:
            if abs(ax) > self.trigger_threshold:
                self.detection_triggered = True
                self.trigger_start_time = timestamp
                print(f"Detection triggered at {timestamp:.2f}s with ax={ax:.2f}")
                return False, None, 0
        
        # 如果已触发，检查是否应该进行检测
        if self.detection_triggered:
            # 控制检测频率
            if timestamp - self.last_detection_time < self.detection_interval:
                return False, None, 0
                
            # 检查是否有足够的数据
            if len(self.accel_buffer) >= 30:
                self.last_detection_time = timestamp
                
                # 确认待定峰值
                self._confirm_pending_peaks(timestamp)
                
                # 检测新峰值
                result = self._detect_movement()
                
                # 检查是否可以关闭触发状态
                if timestamp - self.trigger_start_time > 2.0:  # 2秒后检查
                    recent_data = self.accel_buffer[-20:]
                    if all(abs(x) < self.trigger_threshold for x in recent_data):
                        self.detection_triggered = False
                        print(f"Detection trigger ended at {timestamp:.2f}s")
                
                return result
        
        return False, None, 0
        
    def _detect_movement(self):
        """
        检测横向移动的核心算法
        返回: (movement_detected, direction, distance) 元组
        """
        buffer_list = self.accel_buffer
        current_time = self.time_buffer[-1]
        
        # 获取足够的历史数据用于分析
        analysis_window = min(100, len(buffer_list))  # 分析最近100个点
        recent_data = buffer_list[-analysis_window:]
        
        if len(recent_data) < 30:
            return False, None, 0
        
        # 噪声过滤
        filtered_data = self._moving_average(recent_data, self.ma_window)
        
        # 清理过期的峰值
        self._clean_old_peaks(current_time)
        
        # 检测潜在峰值（但不立即确认）
        potential_peaks = self._detect_potential_peaks(
            filtered_data, 
            current_time, 
            len(buffer_list) - analysis_window
        )
        
        # 将潜在峰值加入待确认列表
        for peak in potential_peaks:
            if not self._is_in_pending(peak):
                self.pending_peaks.append(peak)
                print(f"Potential peak found: {peak['value']:.2f} at {peak['time']:.2f}s")
        
        # 分析已确认的峰值序列
        if len(self.peaks_detected) >= 2:
            movement_result = self._analyze_peak_sequence()
            if movement_result[0]:
                return movement_result
        
        return False, None, 0
    
    def _confirm_pending_peaks(self, current_time):
        """
        确认待定峰值（确保有足够的后续数据）
        """
        confirmed = []
        for peak in self.pending_peaks:
            # 检查是否已过确认延迟时间
            if current_time - peak['time'] >= self.peak_confirmation_delay:
                # 重新计算完整的积分（现在有更多数据）
                peak_idx = self._find_peak_index(peak['time'])
                if peak_idx is not None:
                    # 重新计算速度变化和距离
                    velocity_change = self._calculate_velocity_change_full(peak_idx)
                    distance = self._estimate_distance_full(peak_idx)
                    
                    peak['velocity_change'] = velocity_change
                    peak['distance'] = distance
                    
                    self.peaks_detected.append(peak)
                    confirmed.append(peak)
                    
                    print(f"Peak confirmed: {peak['value']:.2f} at {peak['time']:.2f}s")
                    print(f"  Complete integration: ΔV={velocity_change:.3f} m/s, Δd={distance:.3f} m")
        
        # 移除已确认的峰值
        for peak in confirmed:
            self.pending_peaks.remove(peak)
    
    def _find_peak_index(self, peak_time):
        """
        在时间缓冲区中找到峰值的索引
        """
        for i, t in enumerate(self.time_buffer):
            if abs(t - peak_time) < 0.01:  # 时间匹配容差
                return i
        return None
    
    def _detect_potential_peaks(self, filtered_data, current_time, start_idx):
        """
        检测潜在峰值（不立即计算积分）
        """
        peaks = []
        
        # 需要足够的边界数据
        for i in range(5, len(filtered_data)-5):
            val = filtered_data[i]
            
            # 使用7点窗口检测更稳定的峰值
            is_local_max = all(val > filtered_data[j] for j in range(i-3, i)) and \
                          all(val > filtered_data[j] for j in range(i+1, i+4))
            
            is_local_min = all(val < filtered_data[j] for j in range(i-3, i)) and \
                          all(val < filtered_data[j] for j in range(i+1, i+4))
            
            if (is_local_max or is_local_min) and self._is_valid_peak(val):
                # 计算峰值时间
                orig_idx = start_idx + i + self.ma_window // 2
                if orig_idx >= len(self.time_buffer):
                    peak_time = self.time_buffer[-1]
                else:
                    peak_time = self.time_buffer[orig_idx]
                
                # 检查是否重复
                if not self._is_duplicate_peak(val, peak_time):
                    peak_info = {
                        'value': val,
                        'time': peak_time,
                        'type': 'positive' if val > 0 else 'negative',
                        'abs_value': abs(val),
                        'velocity_change': 0,  # 稍后计算
                        'distance': 0  # 稍后计算
                    }
                    peaks.append(peak_info)
        
        return peaks
    
    def _is_in_pending(self, peak):
        """
        检查峰值是否已在待确认列表中
        """
        for p in self.pending_peaks:
            if abs(p['time'] - peak['time']) < 0.05 and p['type'] == peak['type']:
                return True
        return False
    
    def _calculate_velocity_change_full(self, peak_idx, window_time=0.5):
        """
        计算完整的速度变化（确保有足够的前后数据）
        """
        window = int(window_time * self.sample_rate)
        start = max(0, peak_idx - window)
        end = min(len(self.accel_buffer), peak_idx + window + 1)
        
        # 确保有足够的数据点
        if end - start < 10:
            return 0
        
        # 梯形积分
        velocity_change = 0.0
        for i in range(start, end - 1):
            a1 = self.accel_buffer[i]
            a2 = self.accel_buffer[i + 1]
            velocity_change += (a1 + a2) * self.dt / 2.0
        
        return velocity_change
    
    def _estimate_distance_full(self, peak_idx, window_time=0.5):
        """
        计算完整的距离（双重积分）
        """
        window = int(window_time * self.sample_rate)
        start = max(0, peak_idx - window)
        end = min(len(self.accel_buffer), peak_idx + window + 1)
        
        if end - start < 10:
            return 0
        
        # 第一次积分：加速度 → 速度
        velocity = [0.0]
        for i in range(start, end - 1):
            a1 = self.accel_buffer[i]
            a2 = self.accel_buffer[i + 1]
            dv = (a1 + a2) * self.dt / 2.0
            velocity.append(velocity[-1] + dv)
        
        # 第二次积分：速度 → 位移
        distance = 0.0
        for i in range(len(velocity) - 1):
            v1 = velocity[i]
            v2 = velocity[i + 1]
            distance += abs((v1 + v2) * self.dt / 2.0)
        
        return distance
    
    def _is_valid_peak(self, val):
        """检查值是否为有效峰值"""
        abs_val = abs(val)
        return abs_val > self.peak_threshold and abs_val < self.peak_max_threshold
    
    def _is_duplicate_peak(self, val, peak_time, time_threshold=0.15):
        """检查是否与已存在的峰值重复"""
        # 检查已确认峰值
        for peak in self.peaks_detected:
            if abs(peak['time'] - peak_time) < time_threshold:
                if (val > 0 and peak['type'] == 'positive') or \
                   (val < 0 and peak['type'] == 'negative'):
                    return True
        
        # 检查待确认峰值
        for peak in self.pending_peaks:
            if abs(peak['time'] - peak_time) < time_threshold:
                if (val > 0 and peak['type'] == 'positive') or \
                   (val < 0 and peak['type'] == 'negative'):
                    return True
        
        return False
    
    def _analyze_peak_sequence(self):
        """
        分析峰值序列，判断是否形成有效的移动模式
        返回: (movement_detected, direction, distance)
        """
        if len(self.peaks_detected) < 2:
            return False, None, 0
        
        recent_peaks = self.peaks_detected[-4:]
        
        for i in range(len(recent_peaks)-1):
            first_peak = recent_peaks[i]
            
            for j in range(i+1, len(recent_peaks)):
                second_peak = recent_peaks[j]
                
                # 必须是不同类型的峰值
                if first_peak['type'] == second_peak['type']:
                    continue
                
                # 检查时间间隔
                time_interval = second_peak['time'] - first_peak['time']
                if time_interval > self.peak_time_window:
                    continue
                
                # 验证峰值对
                if self._validate_peak_pair(first_peak, second_peak):
                    # 计算总变化
                    total_velocity_change = first_peak['velocity_change'] + second_peak['velocity_change']
                    total_distance = (first_peak['distance'] + second_peak['distance']) / 2.0
                    
                    # 判断方向
                    direction = self._determine_direction(first_peak, second_peak, total_velocity_change)
                    
                    if direction:
                        # 注册移动
                        self._register_movement(
                            first_peak, second_peak, 
                            direction, total_velocity_change, total_distance
                        )
                        
                        # 清理已使用的峰值
                        self._clean_used_peaks(first_peak, second_peak)
                        
                        return True, direction, total_distance
        
        return False, None, 0
    
    def _validate_peak_pair(self, peak1, peak2):
        """验证两个峰值是否能构成有效的移动模式"""
        ratio = peak1['abs_value'] / peak2['abs_value'] if peak2['abs_value'] > 0 else 999
        if not (0.3 < ratio < 3.0):
            return False
        
        time_diff = abs(peak2['time'] - peak1['time'])
        if time_diff < 0.1 or time_diff > self.peak_time_window:
            return False
        
        return True
    
    def _determine_direction(self, first_peak, second_peak, total_velocity_change):
        """基于峰值序列和速度变化判断移动方向"""
        if abs(total_velocity_change) < 0.001:
            return None
        
        if first_peak['type'] == 'negative' and second_peak['type'] == 'positive':
            return 'forward' 
        elif first_peak['type'] == 'positive' and second_peak['type'] == 'negative':
            return 'backward' 
        
        return None
    
    def _register_movement(self, first_peak, second_peak, direction, velocity_change, distance):
        """注册检测到的移动"""
        self.movement_count += 1
        if direction == 'forward':
            self.forward_movement_count += 1
        else:
            self.backward_movement_count += 1
        
        self.last_movement_time = second_peak['time']
        self.last_movement_direction = direction
        self.last_movement_distance = distance
        
        time_interval = second_peak['time'] - first_peak['time']
        
        print("\n"*2 + "=" * 50)
        print(f"| Movement Detected: {direction} direction")
        print(f"| Peak sequence:     {first_peak['type']} ({first_peak['value']:.2f}) "
              f"@{self._format_time(first_peak['time'])} # {first_peak['time']} ->")
        print("|" + " "*20 + f"{second_peak['type']} ({second_peak['value']:.2f}) "
              f" @{self._format_time(second_peak['time'])} # {second_peak['time']}")
        print(f"| Time interval:     {time_interval:.3f}s")
        print(f"| Velocity change:   {velocity_change:.3f} m/s")
        print(f"| Distance:          {distance:.3f} m ({distance * 100:.1f} cm)")
        print(f"| Total movements:   {self.movement_count} "
              f"(Forward: {self.forward_movement_count}, "
              f"Backward: {self.backward_movement_count})")
        print("=" * 50 + "\n"*2)
    
    def _clean_old_peaks(self, current_time):
        """清理过期的峰值"""
        self.peaks_detected = [
            peak for peak in self.peaks_detected 
            if current_time - peak['time'] < self.peak_time_window * 2
        ]
        
        self.pending_peaks = [
            peak for peak in self.pending_peaks
            if current_time - peak['time'] < self.peak_time_window * 2
        ]
    
    def _clean_used_peaks(self, peak1, peak2):
        """清理已经配对使用的峰值"""
        self.peaks_detected = [
            peak for peak in self.peaks_detected 
            if peak['time'] != peak1['time'] and peak['time'] != peak2['time']
        ]
    
    def _moving_average(self, data, window):
        """移动平均滤波"""
        if len(data) < window:
            return data[:]
        
        result = []
        for i in range(len(data) - window + 1):
            s = sum(data[i:i+window])
            avg = s / window
            result.append(avg)
        
        return result
    
    def _calculate_velocity_change(self, peak_idx, data, window_time=0.5):
        """兼容旧接口的速度计算"""
        window = int(window_time * self.sample_rate)
        start = max(0, peak_idx - window)
        end = min(len(data), peak_idx + window + 1)
        
        velocity_change = 0.0
        for i in range(start, end - 1):
            v1 = data[i]
            v2 = data[i + 1]
            velocity_change += (v1 + v2) * self.dt / 2.0
        
        return velocity_change
    
    def get_status(self):
        """获取当前状态"""
        return {
            'movement_count': self.movement_count,
            'forward_count': self.forward_movement_count,
            'backward_count': self.backward_movement_count,
            'last_movement_time': self.last_movement_time,
            'last_direction': self.last_movement_direction,
            'last_distance': self.last_movement_distance,
            'peaks_confirmed': len(self.peaks_detected),
            'peaks_pending': len(self.pending_peaks),
            'detection_triggered': self.detection_triggered,
            'buffer_size': len(self.accel_buffer)
        }
        
    def report_status(self):
        """打印当前状态"""
        status = self.get_status()
        print("=" * 50)
        print(f"  Total movements: {status['movement_count']}")
        print(f"    Forward: {status['forward_count']}, Backward: {status['backward_count']}")
        if status['last_movement_time']:
            print(f"  Last movement at {status['last_movement_time']:.2f}s, "
                  f"Direction: {status['last_direction']}, "
                  f"Distance: {status['last_distance']:.3f} m")
        print(f"  Confirmed peaks: {status['peaks_confirmed']}, "
              f"Pending peaks: {status['peaks_pending']}")
        print(f"  Detection triggered: {'Yes' if status['detection_triggered'] else 'No'}")
        print(f"  Buffer size: {status['buffer_size']} samples")
        print("=" * 50)


# 数据源类 - 模拟传感器数据流
class SensorDataStream:
    """模拟传感器实时数据流"""
    def __init__(self, data_source, sample_rate=200):
        """
        data_source: 数据源（文件路径或数据列表）
        sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        self.interval = 1.0 / sample_rate  # 采样间隔（秒）
        self.data = []
        self.current_index = 0
        self.start_time = None
        
        # 加载数据
        if isinstance(data_source, str):
            self._load_from_file(data_source)
        else:
            self.data = data_source
    
    def _load_from_file(self, filepath):
        """从文件加载数据"""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip():
                    parts = line.strip().split(",")
                    self.data.append([
                        float(parts[0]),  # timestamp
                        float(parts[1]),  # ax
                        float(parts[2]),  # ay
                        float(parts[3])   # az
                    ])
            print(f"Loaded {len(self.data)} samples from {filepath}")
        except Exception as e:
            print(f"Error loading file: {e}")
            raise
    
    def reset(self):
        """重置数据流"""
        self.current_index = 0
        self.start_time = None
    
    def get_next_sample(self):
        """
        获取下一个样本（模拟实时读取）
        返回: (timestamp, ax, ay, az) 或 None（数据结束）
        """
        if self.current_index >= len(self.data):
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        
        # 使用原始时间戳或生成新的
        if self.start_time is None:
            self.start_time = utime.time()
        
        # 计算实际应该的时间戳
        elapsed = (self.current_index - 1) * self.interval
        timestamp = self.start_time + elapsed
        return timestamp, sample[1], sample[2], sample[3]
    
    def has_more_data(self):
        """检查是否还有数据"""
        return self.current_index < len(self.data)


# 实时测试函数
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


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    mode = "batch"  # 默认实时模式
    speed = 1.0  # 默认速度
    file_path = r"c:\Users\Einzig\Downloads\MicrosoftEdgeDropFiles\Default\test.txt"  # 默认文件
    
    # 简单的参数解析
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            mode = "batch"
        elif sys.argv[1] == "realtime":
            mode = "realtime"
        
        if len(sys.argv) > 2:
            try:
                speed = float(sys.argv[2])
            except:
                pass
        
        if len(sys.argv) > 3:
            file_path = sys.argv[3]
    
    print("\n"*2 + "="*60)
    print("Movement Detector Test Suite")
    print("="*60 +"\n"*2)
    
    if mode == "realtime":
        print("\nMode: Real-time simulation")
        print(f"Speed: {speed}x")
        print("Tip: Use speed=0 for fastest processing without delays")
        print("     Use speed=0.5 for slow motion")
        print("     Use speed=2.0 for double speed")
        test_realtime_detector(file_path, speed)
    else:
        print("\nMode: Batch processing")
        test_batch_detector(file_path)
