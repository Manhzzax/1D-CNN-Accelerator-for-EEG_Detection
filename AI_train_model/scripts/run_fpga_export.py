"""CLI entry point for the selected separable-model FPGA export package."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.fpga_export import export_separable_reference


def main():
    print("=" * 60)
    print("EXPORTING SEPARABLE 1D-CNN FPGA REFERENCE PACKAGE")
    print("=" * 60)
    export_separable_reference()


if __name__ == "__main__":
    main()
