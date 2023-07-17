#!/usr/bin/env python3
import numpy as np
import cv2
import unittest
# Import the functions we want to test.
from main import process_frame

# This is an example of building a test suite using Python's unittest module.
class MyTestCase(unittest.TestCase):

    def test_stats(self):
        frame = cv2.imread("test.jpg")
        results = process_frame(frame)
        self.assertLessEqual(results["min"][0], results["max"][0])
        self.assertLessEqual(results["min"][1], results["max"][1])
        self.assertLessEqual(results["min"][2], results["max"][2])

    # You can add your own test_something functions below to expand the test!
    def test_something(self):
        self.assertTrue(1 == 1)

# This will run all the tests when this file is executed.
if __name__ == "__main__":
    unittest.main()
