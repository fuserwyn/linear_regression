import argparse
import os


class ArgsParser:
    filename: str = None
    graphics: str = None
    bonus: bool = False

    @classmethod
    def parse_args(cls) -> tuple:
        parser = argparse.ArgumentParser(description='Linear Regression on Car Prices')
        parser.add_argument('filename', type=str, help='The CSV file containing the data')
        parser.add_argument('--bonus', action='store_true', help='Calculate R² score')
        parser.add_argument('--o', action='store_true', help='Original values')
        parser.add_argument('--s', action='store_true', help='Standardized values')
        args = parser.parse_args()
        if not os.path.isfile(args.filename):
            parser.error(f"The file '{args.filename}' does not exist. Please check the filename and try again.")
        if not (args.o or args.s):
            parser.error(
                "You must specify at least one graphics option: --o (original values) or --s (standardized values).")
        cls.bonus = args.bonus
        cls.graphics = "o" if args.o else "s" if args.s else None
        return args.filename, cls.graphics, cls.bonus


args_parser = ArgsParser()