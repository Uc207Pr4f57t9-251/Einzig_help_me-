try:
    import utime
except ImportError:
    import time as utime

class QueueManager:
    def __init__(self, filename="queue_status.txt"):
        """
        排卡队列管理器
        filename: 保存队列状态的文件名
        """
        self.filename = filename
        self.current_position = 1  # 当前位置（1为第一位）
        self.position_start_time = None  # 在当前位置开始的时间
        self.queue_start_time = None     # 开始排队的总时间
        self.sync_status = 0             # 同步状态：0=未同步，1=已同步
        self.last_update_time = None     # 最后更新时间

        # 尝试从文件加载现有状态
        self.load_from_file()

    def load_from_file(self):
        """从文件加载队列状态"""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'current_position':
                        self.current_position = int(value)
                    elif key == 'position_start_time':
                        self.position_start_time = float(value) if value != 'None' else None
                    elif key == 'queue_start_time':
                        self.queue_start_time = float(value) if value != 'None' else None
                    elif key == 'sync_status':
                        self.sync_status = int(value)
                    elif key == 'last_update_time':
                        self.last_update_time = float(value) if value != 'None' else None

            print(f"Queue status loaded: Position {self.current_position}")

        except Exception as e:
            print(f"Could not load queue file, starting fresh: {e}")
            self._initialize_new_queue()

    def _initialize_new_queue(self):
        """初始化新的排队状态"""
        current_time = utime.time()
        self.position_start_time = current_time
        self.queue_start_time = current_time
        self.last_update_time = current_time
        self.sync_status = 0
        self.save_to_file()
        print(f"New queue initialized at position {self.current_position}")

    def save_to_file(self):
        """保存队列状态到文件"""
        try:
            with open(self.filename, 'w') as f:
                f.write(f"current_position={self.current_position}\n")
                f.write(f"position_start_time={self.position_start_time}\n")
                f.write(f"queue_start_time={self.queue_start_time}\n")
                f.write(f"sync_status={self.sync_status}\n")
                f.write(f"last_update_time={self.last_update_time}\n")
            return True
        except Exception as e:
            print(f"Error saving queue file: {e}")
            return False

    def set_initial_position(self, position):
        """
        手动设置初始位置（通过按钮或手机终端）
        position: 当前位置（1为第一位）
        """
        if position < 1:
            position = 1

        current_time = utime.time()

        # 如果是第一次设置或位置发生变化
        if self.queue_start_time is None or self.current_position != position:
            self.current_position = position
            self.position_start_time = current_time

            if self.queue_start_time is None:
                self.queue_start_time = current_time

            self.last_update_time = current_time
            self.sync_status = 0  # 标记为未同步

            if self.save_to_file():
                print(f"Position set to {position}")
                self.print_status()
                return True

        return False

    def move_forward(self):
        """向前移动（位置-1，向左移动）"""
        if self.current_position > 1:
            self._update_position(self.current_position - 1, "forward")
            return True
        else:
            print("Already at position 1 (front of queue)")
            return False

    def move_backward(self):
        """向后移动（位置+1，向右移动）"""
        self._update_position(self.current_position + 1, "backward")
        return True

    def _update_position(self, new_position, direction):
        """内部方法：更新位置"""
        old_position = self.current_position
        current_time = utime.time()

        # 更新位置信息
        self.current_position = new_position
        self.position_start_time = current_time
        self.last_update_time = current_time
        self.sync_status = 0  # 标记为未同步

        # 保存到文件
        if self.save_to_file():
            print(f"\n{'='*50}")
            print(f"| POSITION UPDATE: {direction} movement")
            print(f"| Position changed: {old_position} → {new_position}")
            print(f"| Update time: {self._format_time(current_time)}")
            print(f"| Status: Needs sync to server")
            print(f"{'='*50}\n")

            self.print_status()

    def get_position_wait_time(self):
        """获取在当前位置等待的时间（秒）"""
        if self.position_start_time is None:
            return 0
        return utime.time() - self.position_start_time

    def get_total_wait_time(self):
        """获取总等待时间（秒）"""
        if self.queue_start_time is None:
            return 0
        return utime.time() - self.queue_start_time

    def mark_synced(self):
        """标记为已同步到服务器"""
        self.sync_status = 1
        self.save_to_file()
        print("Queue status marked as synced to server")

    def needs_sync(self):
        """检查是否需要同步到服务器"""
        return self.sync_status == 0

    def get_status_dict(self):
        """获取完整状态信息（用于发送到服务器）"""
        current_time = utime.time()
        return {
            'current_position': self.current_position,
            'position_wait_time': self.get_position_wait_time(),
            'total_wait_time': self.get_total_wait_time(),
            'sync_status': self.sync_status,
            'last_update_time': self.last_update_time,
            'current_time': current_time
        }

    def print_status(self):
        """打印当前队列状态"""
        current_time = utime.time()
        position_wait = self.get_position_wait_time()
        total_wait = self.get_total_wait_time()

        print("\n" + "="*40)
        print("| MAIMAI QUEUE STATUS")
        print("="*40)
        print(f"| Current Position: #{self.current_position}")
        print(f"| Time at position: {self._format_duration(position_wait)}")
        print(f"| Total wait time:  {self._format_duration(total_wait)}")
        print(f"| Server sync:      {'[SYNCED]' if self.sync_status == 1 else '[PENDING]'}")
        print(f"| Last update:      {self._format_time(self.last_update_time) if self.last_update_time else 'Never'}")
        print("="*40 + "\n")

    def _format_time(self, timestamp):
        """格式化时间戳为 HH:MM:SS"""
        if timestamp is None:
            return "N/A"
        try:
            t = utime.localtime(int(timestamp))
            return "{:02d}:{:02d}:{:02d}".format(t[3], t[4], t[5])
        except:
            sec = int(timestamp) % 86400
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            return "{:02d}:{:02d}:{:02d}".format(h, m, s)

    def _format_duration(self, seconds):
        """格式化持续时间"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def reset_queue(self):
        """重置队列状态（清除所有数据）"""
        self.current_position = 1
        self.position_start_time = None
        self.queue_start_time = None
        self.sync_status = 0
        self.last_update_time = None
        try:
            with open(self.filename, 'w') as f:
                f.write("")  # 清空文件
            print("Queue reset successfully")
        except:
            print("Warning: Could not clear queue file")