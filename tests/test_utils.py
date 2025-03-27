import unittest
import logging
import time
import io
import sys
from unittest.mock import patch, mock_open

from translateopenai.utils import timer, read_openai_key, configure_logging, TranslationProgressTracker

class TestTimer(unittest.TestCase):
    @patch('time.time')
    def test_timer(self, mock_time):
        # Mock time.time to return predictable values
        mock_time.side_effect = [100.0, 110.0]  # Start time, end time
        
        log_records = []
        
        # Mock the logging to capture log records
        with patch('logging.log') as mock_log:
            # Use the timer as a context manager
            with timer("Test operation"):
                # The timer should have logged a start message
                self.assertEqual(mock_log.call_count, 1)
                self.assertEqual(
                    mock_log.call_args[0][1],
                    "Test operation starting to execute - measuring timing"
                )
                
                # Reset the mock to start fresh for the end message
                mock_log.reset_mock()
                
            # After exiting the context, the timer should log the execution time
            self.assertEqual(mock_log.call_count, 1)
            self.assertTrue(
                "Test operation executed in 10.000000 seconds" in mock_log.call_args[0][1]
            )

class TestReadOpenAIKey(unittest.TestCase):
    def test_read_openai_key(self):
        # Test with a mock file
        with patch("builtins.open", mock_open(read_data="sk-abc123\n")) as mock_file:
            key = read_openai_key("fake_path")
            self.assertEqual(key, "sk-abc123")
            mock_file.assert_called_once_with("fake_path", "r")

class TestConfigureLogging(unittest.TestCase):
    def test_configure_logging(self):
        # Capture the logging configuration
        with patch('logging.basicConfig') as mock_basic_config:
            configure_logging(level=logging.DEBUG, format_str='%(message)s')
            
            # Verify that basicConfig was called with the right arguments
            mock_basic_config.assert_called_once_with(
                level=logging.DEBUG,
                format='%(message)s'
            )

class TestTranslationProgressTracker(unittest.TestCase):
    @patch('time.time')
    def test_tracker_initialization(self, mock_time):
        mock_time.return_value = 100.0
        
        with patch('logging.info') as mock_info:
            tracker = TranslationProgressTracker(total_items=100, description="Test Progress")
            
            # Verify tracker state
            self.assertEqual(tracker.total, 100)
            self.assertEqual(tracker.current, 0)
            self.assertEqual(tracker.description, "Test Progress")
            self.assertEqual(tracker.start_time, 100.0)
            
            # Verify logging
            mock_info.assert_called_once_with("Test Progress 100 items")
    
    @patch('time.time')
    def test_update(self, mock_time):
        # Setup mock to return fixed values for start time and current time
        mock_time.side_effect = [100.0, 110.0]
        
        with patch('logging.info') as mock_info:
            tracker = TranslationProgressTracker(total_items=100)
            mock_info.reset_mock()  # Clear the initialization log
            
            # Update tracker and test logging
            tracker.update(count=10, additional_info="Test info")
            
            # Verify tracker state
            self.assertEqual(tracker.current, 10)
            
            # Verify logging message contains progress percentage and rate
            log_msg = mock_info.call_args[0][0]
            self.assertIn("10/100 (10.0%)", log_msg)
            self.assertIn("1.00 items/sec", log_msg)
            self.assertIn("Test info", log_msg)
    
    @patch('time.time')
    def test_complete(self, mock_time):
        # Setup mock for time
        mock_time.side_effect = [100.0, 120.0]
        
        with patch('logging.info') as mock_info:
            tracker = TranslationProgressTracker(total_items=100)
            mock_info.reset_mock()  # Clear the initialization log
            
            # Mark as complete and verify logging
            tracker.complete()
            
            # Check that both completion messages were logged
            self.assertEqual(mock_info.call_count, 2)
            first_msg = mock_info.call_args_list[0][0][0]
            second_msg = mock_info.call_args_list[1][0][0]
            
            self.assertIn("completed 100 items in 20.00 seconds", first_msg)
            self.assertIn("Average processing rate: 5.00 items/sec", second_msg)

if __name__ == '__main__':
    unittest.main()
