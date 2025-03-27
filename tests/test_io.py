import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

from translateopenai.io import read_excel_data, write_data, append_data

class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'speaker': ['SPEAKER_1', 'SPEAKER_2', 'SPEAKER_1'],
            'text': ['Hello world', 'How are you?', 'I am fine.'],
        })
    
    @patch('pandas.read_excel')
    def test_read_excel_data(self, mock_read_excel):
        # Set up the mock
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_read_excel.return_value = mock_df
        
        # Call the function
        result = read_excel_data('test.xlsx')
        
        # Verify the result
        self.assertIs(result, mock_df)
        mock_read_excel.assert_called_once_with('test.xlsx', engine='openpyxl')
        
        # Test with additional parameters
        result = read_excel_data('test.xlsx', sheet_name='Sheet2')
        mock_read_excel.assert_called_with('test.xlsx', engine='openpyxl', sheet_name='Sheet2')
    
    def test_write_data_xlsx(self):
        # Create a temp file for testing
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Patch the logging
            with patch('logging.info') as mock_info:
                # Write data
                write_data(self.df, temp_path)
                
                # Verify logging
                mock_info.assert_called_once()
                self.assertIn(f"Writing {len(self.df)} rows to", mock_info.call_args[0][0])
                
                # Read back and verify
                df_read = pd.read_excel(temp_path)
                self.assertEqual(len(df_read), len(self.df))
                self.assertTrue(all(col in df_read.columns for col in self.df.columns))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_data_csv(self):
        # Create a temp file for testing
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write data
            write_data(self.df, temp_path)
            
            # Read back and verify
            df_read = pd.read_csv(temp_path)
            self.assertEqual(len(df_read), len(self.df))
            self.assertTrue(all(col in df_read.columns for col in self.df.columns))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_data_txt(self):
        # Create a temp file for testing
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write data
            write_data(self.df, temp_path)
            
            # Read back and verify
            df_read = pd.read_csv(temp_path, sep='\t')
            self.assertEqual(len(df_read), len(self.df))
            self.assertTrue(all(col in df_read.columns for col in self.df.columns))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_append_data_new_file(self):
        # Test appending to a non-existent file (should create it)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            # Remove the file so it doesn't exist
            os.unlink(temp_path)
            
        try:
            # Patch os.path.exists to simulate file not existing
            with patch('logging.info'):
                append_data(self.df, temp_path)
                
                # Read back and verify
                df_read = pd.read_csv(temp_path)
                self.assertEqual(len(df_read), len(self.df))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_append_data_existing_file(self):
        # Create a temp file with initial data
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write initial data
            initial_df = pd.DataFrame({
                'speaker': ['SPEAKER_3'],
                'text': ['Initial text']
            })
            initial_df.to_excel(temp_path, index=False)
            
            # Append data
            with patch('logging.info'):
                append_data(self.df, temp_path)
                
                # Read back and verify
                df_read = pd.read_excel(temp_path)
                # Should have initial data + appended data
                self.assertEqual(len(df_read), len(initial_df) + len(self.df))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()
