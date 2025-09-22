from queueManager import QueueManager
from movementDetector import HorizontalMovementDetector

try:
    import utime
except ImportError:
    import time as utime

def test_queue_basic():
    """测试队列管理器基本功能"""
    print("\n" + "="*60)
    print("Testing QueueManager Basic Functions")
    print("="*60)

    # 创建队列管理器（测试文件）
    queue = QueueManager("test_queue.txt")

    # 测试设置初始位置
    print("\n1. Testing initial position setup...")
    queue.set_initial_position(3)
    queue.print_status()

    # 等待一秒模拟时间流逝
    utime.sleep(1.1)

    # 测试移动
    print("\n2. Testing forward movement...")
    queue.move_forward()

    utime.sleep(0.5)

    print("\n3. Testing backward movement...")
    queue.move_backward()

    # 测试边界条件
    print("\n4. Testing boundary conditions...")
    queue.set_initial_position(1)
    result = queue.move_forward()  # 应该失败，已经在第一位

    print("\n5. Testing sync status...")
    print(f"Needs sync: {queue.needs_sync()}")
    queue.mark_synced()
    print(f"After sync - Needs sync: {queue.needs_sync()}")

    # 显示最终状态
    queue.print_status()

    return queue

def test_movement_integration():
    """测试与移动检测器的集成"""
    print("\n" + "="*60)
    print("Testing Movement Detector Integration")
    print("="*60)

    # 创建带队列功能的移动检测器
    detector = HorizontalMovementDetector(
        enable_queue=True,
        queue_filename="test_movement_queue.txt"
    )

    # 设置初始位置
    print("\n1. Setting initial queue position to 4...")
    detector.set_initial_queue_position(4)

    # 显示初始状态
    detector.report_status()

    # 模拟检测到的移动
    print("\n2. Simulating detected movements...")
    current_time = utime.time()

    # 创建模拟峰值数据
    def create_mock_peak(value, time, peak_type):
        return {
            'value': value,
            'time': time,
            'type': peak_type,
            'abs_value': abs(value),
            'velocity_change': value * 0.1,
            'distance': abs(value) * 0.02
        }

    # 模拟向前移动（forward）
    print("\nSimulating FORWARD movement (left move in maimai)...")
    peak1 = create_mock_peak(-2.1, current_time, 'negative')
    peak2 = create_mock_peak(1.8, current_time + 0.5, 'positive')
    detector._register_movement(peak1, peak2, 'forward', 0.3, 0.05)

    utime.sleep(1.0)

    # 模拟向后移动（backward）
    print("\nSimulating BACKWARD movement (right move in maimai)...")
    current_time = utime.time()
    peak3 = create_mock_peak(2.3, current_time, 'positive')
    peak4 = create_mock_peak(-1.9, current_time + 0.4, 'negative')
    detector._register_movement(peak3, peak4, 'backward', -0.4, 0.06)

    # 显示最终状态
    detector.report_status()

    # 测试队列状态获取
    queue_status = detector.get_queue_status()
    if queue_status:
        print(f"\nQueue Status Dict: {queue_status}")

    return detector

def test_maimai_scenario():
    """模拟真实的maimai排队场景"""
    print("\n" + "="*60)
    print("Simulating Real MAIMAI Queue Scenario")
    print("="*60)

    # 创建队列管理器
    queue = QueueManager("maimai_scenario.txt")

    # 场景：玩家刚开始排队，位置为第5位
    print("\n[SCENARIO START] Player joins queue at position 5")
    queue.set_initial_position(5)

    # 等待一段时间
    print("Waiting 3 seconds...")
    utime.sleep(3.1)

    # 前面的玩家游戏结束，所有玩家向前移动一位
    print("\n[GAME END] Player in front finishes, moving forward...")
    queue.move_forward()  # 5 -> 4

    utime.sleep(2.0)

    # 又一个玩家完成
    print("\n[GAME END] Another player finishes, moving forward...")
    queue.move_forward()  # 4 -> 3

    utime.sleep(1.5)

    # 新玩家加入，需要往后让位
    print("\n[NEW PLAYER] Someone cuts in line, moving backward...")
    queue.move_backward()  # 3 -> 4

    utime.sleep(2.5)

    # 继续向前
    print("\n[GAME END] Moving forward again...")
    queue.move_forward()  # 4 -> 3

    utime.sleep(1.0)

    print("\n[GAME END] Moving to position 2...")
    queue.move_forward()  # 3 -> 2

    utime.sleep(1.5)

    print("\n[GAME END] Finally at front of queue!")
    queue.move_forward()  # 2 -> 1

    # 显示完整的排队历程
    final_status = queue.get_status_dict()

    print("\n" + "="*50)
    print("| MAIMAI QUEUE SIMULATION COMPLETE")
    print("="*50)
    print(f"| Final Position:     #{final_status['current_position']}")
    print(f"| Time at position:   {queue._format_duration(final_status['position_wait_time'])}")
    print(f"| Total queue time:   {queue._format_duration(final_status['total_wait_time'])}")
    print(f"| Ready to play:      {'YES!' if final_status['current_position'] == 1 else 'Not yet'}")
    print("="*50)

    return final_status

def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*70)
    print("# MAIMAI SMART QUEUE CARD - COMPREHENSIVE TESTING")
    print("#"*70)

    try:
        # 基础功能测试
        queue_manager = test_queue_basic()

        # 集成测试
        detector = test_movement_integration()

        # 场景测试
        scenario_result = test_maimai_scenario()

        print("\n" + "#"*70)
        print("# ALL TESTS COMPLETED SUCCESSFULLY!")
        print("#"*70)

        # 清理测试文件
        try:
            import os
            test_files = ["test_queue.txt", "test_movement_queue.txt", "maimai_scenario.txt"]
            for file in test_files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Cleaned up test file: {file}")
        except:
            print("Note: Could not clean up test files automatically")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

def manual_queue_test():
    """手动测试队列功能的交互式界面"""
    print("\n" + "="*50)
    print("Manual Queue Testing Interface")
    print("Commands: ")
    print("  pos <n>  - Set position to n")
    print("  left     - Move left (forward)")
    print("  right    - Move right (backward)")
    print("  status   - Show status")
    print("  sync     - Mark as synced")
    print("  reset    - Reset queue")
    print("  quit     - Exit")
    print("="*50)

    queue = QueueManager("manual_test_queue.txt")

    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd:
                continue

            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'pos' and len(cmd) > 1:
                try:
                    pos = int(cmd[1])
                    queue.set_initial_position(pos)
                except ValueError:
                    print("Invalid position number")
            elif cmd[0] == 'left':
                queue.move_forward()
            elif cmd[0] == 'right':
                queue.move_backward()
            elif cmd[0] == 'status':
                queue.print_status()
            elif cmd[0] == 'sync':
                queue.mark_synced()
            elif cmd[0] == 'reset':
                queue.reset_queue()
            else:
                print("Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Manual test session ended.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        manual_queue_test()
    else:
        run_all_tests()