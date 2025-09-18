"""
NTFY消息发送模块
用于向ntfy.sh平台发送推送通知
"""

import urequests
import json

class NTFY:
    def __init__(self, topic="schwarzerMSG", server="https://ntfy.sh"):
        """
        初始化NTFY客户端
        
        Args:
            topic (str): ntfy主题名称，默认为schwarzerMSG
            server (str): ntfy服务器地址，默认为https://ntfy.sh
        """
        self.topic = topic
        self.server = server
        self.url = f"{server}/{topic}"
        
    def send_simple(self, message):
        """
        发送简单消息
        
        Args:
            message (str): 要发送的消息内容
            
        Returns:
            bool: 发送是否成功
        """
        try:
            print(f"Sending message to {self.url}...")
            response = urequests.post(self.url, data=message)
            
            if response.status_code == 200:
                print("Message sent successfully!")
                response.close()
                return True
            else:
                print(f"Failed to send message. Status code: {response.status_code}")
                response.close()
                return False
                
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def send_advanced(self, message, title=None, priority=None, tags=None):
        """
        发送高级消息（包含标题、优先级、标签等）
        
        Args:
            message (str): 消息内容
            title (str): 消息标题
            priority (str): 优先级 (min, low, default, high, max)
            tags (list): 标签列表
            
        Returns:
            bool: 发送是否成功
        """
        try:
            headers = {"Content-Type": "application/json"}
            
            data = {"message": message}
            
            if title:
                data["title"] = title
            if priority:
                data["priority"] = priority
            if tags:
                data["tags"] = tags
                
            print(f"Sending advanced message to {self.url}...")
            response = urequests.post(
                self.url, 
                data=json.dumps(data),
                headers=headers
            )
            
            if response.status_code == 200:
                print("Advanced message sent successfully!")
                response.close()
                return True
            else:
                print(f"Failed to send advanced message. Status code: {response.status_code}")
                response.close()
                return False
                
        except Exception as e:
            print(f"Error sending advanced message: {e}")
            return False
    
    def send_sensor_data(self, sensor_name, value, unit=""):
        """
        发送传感器数据
        
        Args:
            sensor_name (str): 传感器名称
            value (float): 传感器值
            unit (str): 单位
            
        Returns:
            bool: 发送是否成功
        """
        message = f"{sensor_name}: {value}{unit}"
        title = f"ESP32S3 Sensor Update"
        
        return self.send_advanced(
            message=message,
            title=title,
            priority="default",
            tags=["sensor", "esp32"]
        )
    
    def send_alert(self, alert_message, level="warning"):
        """
        发送警报消息
        
        Args:
            alert_message (str): 警报内容
            level (str): 警报级别 (info, warning, critical)
            
        Returns:
            bool: 发送是否成功
        """
        priority_map = {
            "info": "low",
            "warning": "high", 
            "critical": "max"
        }
        
        tags_map = {
            "info": ["info", "esp32"],
            "warning": ["warning", "esp32"],
            "critical": ["rotating_light", "esp32", "alert"]
        }
        
        return self.send_advanced(
            message=alert_message,
            title=f"ESP32S3 {level.upper()}",
            priority=priority_map.get(level, "default"),
            tags=tags_map.get(level, ["esp32"])
        )


# 便捷函数
def send_message(message, topic="schwarzerMSG"):
    """
    快速发送消息的便捷函数
    
    Args:
        message (str): 消息内容
        topic (str): 主题名称
        
    Returns:
        bool: 发送是否成功
    """
    ntfy_client = NTFY(topic)
    return ntfy_client.send_simple(message)

def send_sensor_alert(sensor_name, value, threshold, unit=""):
    """
    发送传感器超阈值警报
    
    Args:
        sensor_name (str): 传感器名称
        value (float): 当前值
        threshold (float): 阈值
        unit (str): 单位
        
    Returns:
        bool: 发送是否成功
    """
    message = f"{sensor_name} value {value}{unit} exceeds threshold {threshold}{unit}!"
    ntfy_client = NTFY()
    return ntfy_client.send_alert(message, "warning")

# 测试函数
def test_ntfy():
    """
    测试ntfy功能
    """
    print("Testing NTFY functionality...")
    
    ntfy_client = NTFY()
    
    # 测试简单消息
    print("1. Testing simple message...")
    success1 = ntfy_client.send_simple("Hello from ESP32S3!")
    
    # 测试高级消息
    print("2. Testing advanced message...")
    success2 = ntfy_client.send_advanced(
        message="This is a test message with title and tags",
        title="ESP32S3 Test",
        priority="high",
        tags=["test", "esp32"]
    )
    
    # 测试传感器数据
    print("3. Testing sensor data...")
    success3 = ntfy_client.send_sensor_data("Temperature", 25.6, "°C")
    
    # 测试警报
    print("4. Testing alert...")
    success4 = ntfy_client.send_alert("System startup completed", "info")
    
    print(f"Test results: Simple={success1}, Advanced={success2}, Sensor={success3}, Alert={success4}")
    return all([success1, success2, success3, success4])

if __name__ == "__main__":
    # 如果直接运行此文件，执行测试
    test_ntfy()