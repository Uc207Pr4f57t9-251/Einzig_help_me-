try:
    import utime
except ImportError:
    import time as utime

import math

try:
    from queueManager import QueueManager
    QUEUE_AVAILABLE = True
except ImportError:
    QUEUE_AVAILABLE = False
    print("Warning: QueueManager not available")

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
                 peak_confirmation_delay=0.3,  # 峰值确认延迟（秒）
                 enable_queue=True,         # 启用队列管理
                 queue_filename="queue_status.txt"):  # 队列文件名
        
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
        # self.simulation_start = utime.time()
        self.simulation_start = 0
        
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

        # 队列管理器
        self.queue_manager = None
        if enable_queue and QUEUE_AVAILABLE:
            try:
                self.queue_manager = QueueManager(queue_filename)
                print("Queue Manager initialized")
            except Exception as e:
                print(f"Failed to initialize Queue Manager: {e}")
        elif enable_queue:
            print("Queue management requested but QueueManager not available")
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
        analysis_window = min(int(self.peak_time_window*self.sample_rate*0.8), len(buffer_list))  # 分析最近100个点
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
              f"@{self._format_time(first_peak['time'])} # {first_peak['time'] - self.simulation_start} ->")
        print("|" + " "*20 + f"{second_peak['type']} ({second_peak['value']:.2f}) "
              f" @{self._format_time(second_peak['time'])} # {second_peak['time'] - self.simulation_start}")
        print(f"| Time interval:     {time_interval:.3f}s")
        print(f"| Velocity change:   {velocity_change:.3f} m/s")
        print(f"| Distance:          {distance:.3f} m ({distance * 100:.1f} cm)")
        print(f"| Total movements:   {self.movement_count} "
              f"(Forward: {self.forward_movement_count}, "
              f"Backward: {self.backward_movement_count})")
        print("=" * 50)

        # 更新队列位置
        if self.queue_manager:
            if direction == 'forward':
                success = self.queue_manager.move_forward()
                if success:
                    print("| Queue: Moved forward (position -1)")
                else:
                    print("| Queue: Already at front position")
            else:  # backward
                self.queue_manager.move_backward()
                print("| Queue: Moved backward (position +1)")

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

        # 显示队列状态
        if self.queue_manager:
            self.queue_manager.print_status()

    def set_initial_queue_position(self, position):
        """手动设置队列初始位置（通过按钮或手机终端）"""
        if self.queue_manager:
            return self.queue_manager.set_initial_position(position)
        else:
            print("Queue manager not available")
            return False

    def get_queue_status(self):
        """获取队列状态信息（用于发送到服务器）"""
        if self.queue_manager:
            return self.queue_manager.get_status_dict()
        else:
            return None

    def mark_queue_synced(self):
        """标记队列状态已同步到服务器"""
        if self.queue_manager:
            self.queue_manager.mark_synced()

    def queue_needs_sync(self):
        """检查队列是否需要同步到服务器"""
        if self.queue_manager:
            return self.queue_manager.needs_sync()
        return False

    def reset_queue(self):
        """重置队列状态"""
        if self.queue_manager:
            self.queue_manager.reset_queue()
        else:
            print("Queue manager not available")