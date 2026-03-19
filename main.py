import sys
import ctypes

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def _set_windows_app_id() -> None:
    """Set a stable Windows app id so the taskbar uses the app icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("pyside6.ml.workbench")
    except Exception:
        pass


def main() -> None:
    """
    Entry point of the application.

    Initialize the Qt application, create the main window,
    and start the event loop.
    """

    _set_windows_app_id()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
