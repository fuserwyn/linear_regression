import csv
import sys
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    data: list = None
    original_values: list = None

    def __init__(self, filename: str):
        self.load_data(filename)

    def load_data(self, filename):
        if not os.path.isfile(filename):
            sys.exit("Error: File does not exist.")

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.data = [row for row in reader]
            self.original_values = self.data.copy()
