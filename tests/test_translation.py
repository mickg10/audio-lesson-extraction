import unittest
import pandas as pd
import translateopenai
from unittest.mock import patch, MagicMock, call, mock_open

from translateopenai.translation import translate_text, translate_dataframe, translate_file

class TestTranslation(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'speaker': ['SPEAKER_1', 'SPEAKER_2', 'SPEAKER_1'],
            'text': ['Hello world', 'How are you?', 'I am fine.'],
            'index': [0, 1, 2]
        })
        # Set index for dataframe
        self.df.set_index('index', inplace=True)
    
    @patch('openai.OpenAI')
    def test_translate_text(self, mock_openai):
        # Setup mocks
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        
        # Configure the nested mocks
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response
        mock_response.choices = [mock_message]
        mock_message.message.content = "Translated text"
        
        # Call the function
        result = translate_text("Hello", "Spanish", "gpt-3.5-turbo", mock_client)
        
        # Verify the result
        self.assertEqual(result, "Translated text")
        
        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs['model'], "gpt-3.5-turbo")
        self.assertEqual(call_kwargs['temperature'], 0)
        self.assertEqual(len(call_kwargs['messages']), 1)
        self.assertEqual(call_kwargs['messages'][0]['role'], "user")
        self.assertIn("Spanish", call_kwargs['messages'][0]['content'])
        self.assertIn("Hello", call_kwargs['messages'][0]['content'])
    
    def test_translate_dataframe(self):
        # We need to patch at a deeper level - at the open() function itself
        mock_api_key = "sk-test123"
        
        # Create a patched version of the translate_dataframe function that doesn't actually read files
        with patch('builtins.open', mock_open(read_data=mock_api_key)) as mock_file, \
             patch('openai.OpenAI') as mock_openai, \
             patch('translateopenai.translation.translate_text') as mock_translate_text, \
             patch('translateopenai.translation.TranslationProgressTracker'):
             
            # Setup the OpenAI client mock
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # Mock translate_text to return a properly formatted response
            mock_translate_text.return_value = (
                "0|SPEAKER_1|Hola mundo\n"
                "1|SPEAKER_2|u00bfCu00f3mo estu00e1s?\n"
                "2|SPEAKER_1|Estoy bien."
            )
            
            # Create a test DataFrame with known data
            test_df = pd.DataFrame({
                'speaker': ['SPEAKER_1', 'SPEAKER_2', 'SPEAKER_1'],
                'text': ['Hello world', 'How are you?', 'I am fine.']
            })
            test_df.index = [0, 1, 2]  # Ensure specific indexes for testing
            
            # Call the function under test
            result_df = translate_dataframe(
                api_key_file="test_key.txt",  # This file doesn't need to exist now
                df=test_df,
                speaker_col="speaker",
                text_col="text",
                output_col="translated",
                language="Spanish",
                context_lines=3  # Process all at once for testing
            )
            
            # Verify file was opened
            mock_file.assert_called_with("test_key.txt", "r")
            
            # Verify output dataframe has a translated column
            self.assertIn("translated", result_df.columns)
    
    @patch('translateopenai.translation.translate_dataframe')
    @patch('translateopenai.io.read_excel_data')
    @patch('translateopenai.io.write_data')
    def test_translate_file(self, mock_write_data, mock_read_excel, mock_translate_df):
        # Setup mocks
        mock_df = pd.DataFrame({
            'speaker': ['SPEAKER_1', 'SPEAKER_2'],
            'text': ['Hello', 'World']
        })
        mock_read_excel.return_value = mock_df
        mock_translate_df.return_value = mock_df
        
        # Call the function
        translate_file(
            input_file="input.xlsx",
            output_file="output.xlsx",
            api_key_file="key.txt",
            language="French",
            context_lines=10,
            model="gpt-4"
        )
        
        # Verify mocks were called correctly
        mock_read_excel.assert_called_once_with("input.xlsx")
        
        # Verify translate_dataframe was called with correct arguments
        mock_translate_df.assert_called_once()
        call_args = mock_translate_df.call_args[1]
        self.assertEqual(call_args['api_key_file'], "key.txt")
        self.assertIs(call_args['df'], mock_df)
        self.assertEqual(call_args['language'], "French")
        self.assertEqual(call_args['context_lines'], 10)
        self.assertEqual(call_args['model'], "gpt-4")
        
        # Verify final write occurred
        mock_write_data.assert_called_with(mock_df, "output.xlsx")

if __name__ == '__main__':
    unittest.main()
