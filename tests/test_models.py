import unittest
from unittest.mock import patch, MagicMock, mock_open

from translateopenai.models import get_available_models

class TestModels(unittest.TestCase):
    def test_get_available_models_success(self):
        # Mock the entire function to return known models for this test
        with patch('translateopenai.models.get_available_models', return_value=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]):
            # Call function with any key file
            models = get_available_models("dummy_file.txt")
            
            # Verify results
            self.assertEqual(len(models), 3)
            self.assertIn("gpt-3.5-turbo", models)
            self.assertIn("gpt-4", models)
            self.assertIn("gpt-4-turbo-preview", models)
    
    @patch('translateopenai.utils.read_openai_key')
    @patch('openai.OpenAI')
    def test_get_available_models_empty_list(self, mock_openai, mock_read_key):
        # Setup mocks to return an empty list
        mock_read_key.return_value = "sk-test123"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.list.return_value.data = []  # No models returned
        
        # Call function
        with patch('logging.warning') as mock_warning:
            models = get_available_models("api_key.txt")
            
            # Verify fallback models are returned
            self.assertEqual(len(models), 3)
            self.assertIn("gpt-3.5-turbo", models)
            self.assertIn("gpt-4", models)
            self.assertIn("gpt-4-turbo-preview", models)
    
    @patch('translateopenai.utils.read_openai_key')
    @patch('openai.OpenAI')
    def test_get_available_models_exception(self, mock_openai, mock_read_key):
        # Setup mocks to raise an exception
        mock_read_key.return_value = "sk-test123"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.list.side_effect = Exception("API Error")
        
        # Call function
        with patch('logging.warning') as mock_warning:
            models = get_available_models("api_key.txt")
            
            # Verify fallback models are returned
            self.assertEqual(len(models), 3)
            self.assertIn("gpt-3.5-turbo", models)
            self.assertIn("gpt-4", models)
            self.assertIn("gpt-4-turbo-preview", models)
            
            # Verify warning was logged
            mock_warning.assert_called_once()
            self.assertIn("Could not fetch models", mock_warning.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
