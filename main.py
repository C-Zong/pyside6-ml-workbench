from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
import sys

def main() -> None:
    """
    Entry point of the application.

    Initialize the Qt application, create the main window,
    and start the event loop.
    """

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()