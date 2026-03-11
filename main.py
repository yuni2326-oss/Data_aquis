# main.py
import sys
import argparse

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
