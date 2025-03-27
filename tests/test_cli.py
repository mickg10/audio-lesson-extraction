import unittest
import argparse
import logging
from unittest.mock import patch, MagicMock

from translateopenai.cli import add_translation_arguments, add_model_argument, create_translation_parser

class TestCLI(unittest.TestCase):
    def test_add_translation_arguments(self):
        # Create a parser
        parser = argparse.ArgumentParser()
        
        # Add translation arguments
        modified_parser = add_translation_arguments(parser, default_api_key="test_key.txt")
        
        # Verify the parser was modified correctly
        self.assertIs(modified_parser, parser)  # Should return the same parser object
        
        # Parse some args and check the defaults
        args = parser.parse_args([])
        self.assertEqual(args.language, "english")
        self.assertEqual(args.context, 40)
        self.assertEqual(args.api_key_file, "test_key.txt")
    
    def test_add_model_argument(self):
        # Create a parser
        parser = argparse.ArgumentParser()
        
        # Mock the get_available_models function within this specific test
        with patch('translateopenai.cli.get_available_models') as mock_get_models:
            mock_get_models.return_value = ["gpt-3.5-turbo", "gpt-4"]
            
            # Add model argument
            modified_parser = add_model_argument(parser, "api_key.txt", default_model="gpt-4")
            
            # Verify the mock was called
            mock_get_models.assert_called_once_with("api_key.txt")
            
            # Verify the parser was modified correctly
            self.assertIs(modified_parser, parser)  # Should return the same parser object
            
            # Parse some args and check the defaults
            args = parser.parse_args([])
            self.assertEqual(args.model, "gpt-4")
    
    def test_create_translation_parser(self):
        # Use patch to avoid actually parsing args from sys.argv
        with patch('argparse.ArgumentParser.parse_known_args') as mock_parse:
            # Set up the mock to return valid args
            mock_args = argparse.Namespace()
            mock_args.log_level = logging.INFO
            mock_args.api_key_file = "test_key.txt"
            mock_parse.return_value = (mock_args, [])
            
            # Create the parser
            parser, temp_args = create_translation_parser(
                description="Test Parser",
                default_api_key="default_key.txt"
            )
            
            # Verify the parser has the right description
            self.assertEqual(parser.description, "Test Parser")
            
            # Verify the temp_args are correct
            self.assertEqual(temp_args.log_level, logging.INFO)
            self.assertEqual(temp_args.api_key_file, "test_key.txt")

if __name__ == '__main__':
    unittest.main()
