#!/usr/bin/env python3
# =============================================================================
# 🔥 Ecommerce Analysis Dashboard - SPARK v4.0 Compatible
# =============================================================================

import sys
import subprocess
import os
from datetime import datetime

def check_dependencies():
    """檢查依賴"""
    print("🔍 檢查環境...")
    
    # Spark
    try:
        import pyspark
        print("✅ PySpark 已安裝")
    except ImportError:
        print("⚠️ 安裝 PySpark: pip install pyspark")
    
    # 其他
    required = ['pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"⚠️ 安裝 {pkg}: pip install {pkg}")
    
    print("✅ 環境檢查完成！")

def install_spark():
    """一鍵安裝 Spark"""
    print("🚀 安裝 PySpark...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspark"])
    print("✅ PySpark 安裝完成！")

def main():
    print("\n" + "="*70)
    print("🎯 Ecommerce Analysis Dashboard v4.0 - SPARK READY")
    print("="*70)
    
    check_dependencies()
    
    print("\n🚀 啟動 ModelBuilding...")
    try:
        # 🔥 自動導入 Spark 版 ModelBuilding
        from ModelBuilding import main as model_main
        model_main()
    except ImportError as e:
        print(f"❌ ModelBuilding.py 導入失敗: {e}")
        print("💡 請確保 ModelBuilding.py 在同一目錄")
    
    print("\n🎉 分析完成！查看生成的 PNG/CSV 文件")

if __name__ == "__main__":
    main()