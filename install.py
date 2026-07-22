#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    print("正在安装 Slope Stability Predictor...")
    print(f"当前路径: {os.getcwd()}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "."],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n安装成功！")
        print("运行 'python -m slope_stability --help' 开始使用")
    else:
        print("\n安装失败:")
        print(f"错误信息: {result.stderr}")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()