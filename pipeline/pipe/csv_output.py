"""
This module defines the CSVOutput class, which is responsible for writing data to a CSV file. 
It initializes the file with a specified path and column names, and provides a method to process 
and write input data as rows in the CSV file. The file is automatically closed when the object 
is deleted, ensuring proper resource management.
"""

from csv import writer

class CSVOutput:
    def __init__(self, path, column_names):
        self.file = open(path, 'w', newline='')
        self.writer = writer(self.file)
        self.writer.writerow(column_names)

    def process(self, inputs):
        self.writer.writerow(inputs)

    def __del__(self):
        self.file.close()
