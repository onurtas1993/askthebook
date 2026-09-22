"""Shared PyInstaller entry point for the Windows and Linux console server."""

from multiprocessing import freeze_support

from src.api.__main__ import main


if __name__ == "__main__":
    freeze_support()
    main()
