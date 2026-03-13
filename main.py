# main.py
import sys
import argparse
import ctypes
import glob
import os

# niScope_64.dll은 LabVIEW Runtime(lvrt.dll)을 delay-load로 참조하는데
# 해당 DLL이 시스템 PATH에 없으므로 Qt 및 다른 임포트 전에 미리 로드해야 함.
_kernel32 = ctypes.WinDLL("kernel32")
_ni_scope_dll = r"C:\Program Files\IVI Foundation\IVI\Bin\niScope_64.dll"
if os.path.exists(_ni_scope_dll):
    for _d in sorted(
        glob.glob(r"C:\Program Files\National Instruments\Shared\LabVIEW Run-Time\*"),
        reverse=True,
    ):
        if os.path.isdir(_d):
            _kernel32.AddDllDirectory(_d)
            break
    _kernel32.AddDllDirectory(os.path.dirname(_ni_scope_dll))
    ctypes.CDLL(_ni_scope_dll)

from PyQt5.QtWidgets import QApplication

from usb5133_daq.ui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="NI USB-5133 DAQ Application")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="실제 하드웨어 없이 MockScope로 실행 (개발/테스트용)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(use_mock=args.mock)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
