from testFunctions import test_realtime_detector, test_batch_detector

if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    mode = "realtime"  # 默认实时模式
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
