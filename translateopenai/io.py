import pandas as pd
import logging
from typing import Optional, Union, Dict, Any

def read_excel_data(input_file: str, **kwargs) -> pd.DataFrame:
    """
    Read data from Excel file with standard options.
    
    Args:
        input_file: Path to the Excel file
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        DataFrame containing the Excel data
    """
    engine = kwargs.pop('engine', 'openpyxl')
    logging.info(f"Reading data from {input_file}")
    return pd.read_excel(input_file, engine=engine, **kwargs)

def write_data(df: pd.DataFrame, output_file: str, index: bool = False, **kwargs) -> None:
    """
    Write dataframe to appropriate format based on file extension.
    
    Args:
        df: DataFrame to write
        output_file: Path to the output file
        index: Whether to include the index in the output
        **kwargs: Additional arguments to pass to the specific writer function
    """
    logging.info(f"Writing {df.shape[0]} rows to {output_file}")
    
    if output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=index, **kwargs)
    elif output_file.endswith(".txt") or output_file.endswith(".tsv"):
        sep = kwargs.pop('sep', '\t')
        df.to_csv(output_file, index=index, sep=sep, **kwargs)
    elif output_file.endswith(".csv"):
        sep = kwargs.pop('sep', ',')
        df.to_csv(output_file, index=index, sep=sep, **kwargs)
    else:
        logging.warning(f"Unknown file extension for {output_file}, defaulting to CSV format")
        df.to_csv(output_file, index=index, **kwargs)

def append_data(df: pd.DataFrame, output_file: str, **kwargs) -> None:
    """
    Append dataframe to an existing file if it exists.
    
    Args:
        df: DataFrame to append
        output_file: Path to the output file
        **kwargs: Additional arguments to pass to the specific writer function
    """
    import os
    
    if not os.path.exists(output_file):
        write_data(df, output_file, **kwargs)
        return
        
    logging.info(f"Appending {df.shape[0]} rows to {output_file}")
    
    if output_file.endswith(".xlsx"):
        # For Excel, we need to read the existing file, append, and then write back
        existing_df = read_excel_data(output_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        write_data(combined_df, output_file, **kwargs)
    elif output_file.endswith((".txt", ".tsv", ".csv")):
        # For text files, we can use mode='a'
        mode = kwargs.pop('mode', 'a')
        header = kwargs.pop('header', False)  # Don't write header when appending
        
        if output_file.endswith((".txt", ".tsv")):
            sep = kwargs.pop('sep', '\t')
        else:  # CSV
            sep = kwargs.pop('sep', ',')
            
        df.to_csv(output_file, mode=mode, header=header, sep=sep, **kwargs)
    else:
        logging.warning(f"Unknown file extension for {output_file}, defaulting to CSV format")
        df.to_csv(output_file, mode='a', header=False, **kwargs)
