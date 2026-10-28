#!/usr/bin/env python3
"""Запуск Python-проверок диагностики и контракта BD, без компилятора."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    for script in (ROOT/'diagnostics/test_all.py',ROOT/'data_tools/test_contract.py'):
        result=subprocess.run([sys.executable,str(script),'-v'])
        
        if result.returncode:
            sys.exit(result.returncode)
            
    sys.exit(0)
