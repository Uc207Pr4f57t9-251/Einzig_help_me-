from movementDetector import HorizontalMovementDetector
try: 
    import utime
except ImportError:
    import time as utime
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

