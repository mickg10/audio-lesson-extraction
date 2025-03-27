#!/usr/bin/env python
"""
Test runner script for the audio-lesson-extraction project.

This script provides a simple interface to run the project tests using unittest or pytest.
It also includes code coverage reporting when available.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py --verbose # Run tests with verbose output
    python run_tests.py --coverage # Run tests with coverage report
"""

import sys
import argparse
import unittest
import os

def run_unittest_tests(verbose=False):
    """Run tests using the unittest framework"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    return runner.run(suite).wasSuccessful()

def run_pytest_tests(verbose=False, coverage=False):
    """Run tests using pytest if available"""
    try:
        import pytest
        
        args = ['-xvs' if verbose else '-xs', 'tests']
        
        if coverage:
            try:
                import pytest_cov
                args = ['--cov=translateopenai', '--cov-report=term-missing'] + args
            except ImportError:
                print("Warning: pytest-cov not installed. Skipping coverage report.")
        
        return pytest.main(args) == 0
    except ImportError:
        print("pytest not installed, falling back to unittest")
        return run_unittest_tests(verbose)

def main():
    parser = argparse.ArgumentParser(description="Run tests for the audio-lesson-extraction project")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--use-unittest", action="store_true", help="Use unittest instead of pytest")
    
    args = parser.parse_args()
    
    print("=== Running tests for audio-lesson-extraction ===")
    
    if args.use_unittest:
        success = run_unittest_tests(verbose=args.verbose)
    else:
        success = run_pytest_tests(verbose=args.verbose, coverage=args.coverage)
    
    if success:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
