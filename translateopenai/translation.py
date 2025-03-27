import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from openai import OpenAI

from translateopenai.utils import timer, read_openai_key, TranslationProgressTracker

def translate_text(text: str, target_language: str, model: str, client: OpenAI) -> str:
    """
    Translate text to the target language using OpenAI models.
    
    Args:
        text: Text to translate
        target_language: Language to translate to
        model: OpenAI model to use for translation
        client: OpenAI client instance
        
    Returns:
        Translated text
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "user", "content": f"Translate the following text while maintaining format, "
                                       f"prefixing with numbers, and do not merge lines to {target_language}: '{text}'"}
        ]
    )
    return response.choices[0].message.content

def translate_dataframe(
    api_key_file: str,
    df: pd.DataFrame,
    speaker_col: str = "speaker",
    text_col: str = "text",
    output_col: str = "translated",
    language: str = "english",
    context_lines: int = 40,
    model: str = "gpt-3.5-turbo",
    apply_at_step: Optional[Callable[[pd.DataFrame], None]] = None,
    retry_attempts: int = 3,
    retry_delay: float = 2.0
) -> pd.DataFrame:
    """
    Translate text in a dataframe to the target language.
    
    Args:
        api_key_file: Path to OpenAI API key file
        df: DataFrame containing text to translate
        speaker_col: Column name for speaker information
        text_col: Column name for text to translate
        output_col: Column name for translated text
        language: Target language for translation
        context_lines: Number of lines to process together for context
        model: OpenAI model to use for translation
        apply_at_step: Optional callback function to apply after each batch
        retry_attempts: Number of retry attempts for API errors
        retry_delay: Delay between retry attempts in seconds
        
    Returns:
        DataFrame with translated text in the output_col
    """
    logging.debug(f"Translating {df.shape[0]} lines to {language} using {model}")
    client = OpenAI(api_key=read_openai_key(api_key_file))
    step = context_lines
    
    # Initialize progress tracker
    progress = TranslationProgressTracker(df.shape[0], f"Translating to {language}")
    
    # Ensure output column exists
    if output_col not in df.columns:
        df[output_col] = [""] * df.shape[0]
    
    # Process in batches
    for i in range(0, df.shape[0], step):
        batch_end = min(i + step, df.shape[0])
        
        with timer(f"Translating lines {i} to {batch_end}"):
            slice = df.iloc[i:batch_end][[speaker_col, text_col]]
            slice["merged"] = slice.index.astype(str) + "|" + slice[speaker_col] + "|" + slice[text_col]
            line = "\n".join(list(slice["merged"]))
            
            # Implement retry logic
            for attempt in range(retry_attempts):
                try:
                    output = translate_text(f"\n{line}", language, model, client)
                    break
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        logging.warning(f"Translation attempt {attempt+1} failed: {str(e)} - retrying in {retry_delay}s")
                        import time
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"Translation failed after {retry_attempts} attempts: {str(e)}")
                        raise
            
            # Process the translated output
            lines = output.split("\n")
            for l in lines:
                # Skip empty lines
                if not l.strip():
                    continue
                    
                # Parse the output format
                splt = l.split("|")
                if len(splt) != 3:
                    logging.warning(f"Unexpected format in translation output: '{l}' - skipping")
                    continue
                    
                idx, _, text = splt
                try:
                    idx = int(idx.replace("'", "").strip())
                    df.loc[idx, output_col] = text.strip()
                except (ValueError, KeyError) as e:
                    logging.warning(f"Error processing translation result: {str(e)} - line: '{l}'")
            
            # Update progress
            batch_size = batch_end - i
            progress.update(batch_size)
            
            # Call the callback if provided
            if apply_at_step is not None:
                apply_at_step(df)
    
    progress.complete()
    return df

def translate_file(
    input_file: str,
    output_file: str,
    api_key_file: str,
    language: str = "english",
    context_lines: int = 40,
    model: str = "gpt-3.5-turbo",
    speaker_col: str = "speaker",
    text_col: str = "text",
    output_col: str = "translated"
) -> None:
    """
    Translate text in a file to the target language and save to output file.
    
    Args:
        input_file: Path to input file (Excel format)
        output_file: Path to output file
        api_key_file: Path to OpenAI API key file
        language: Target language for translation
        context_lines: Number of lines to process together for context
        model: OpenAI model to use for translation
        speaker_col: Column name for speaker information
        text_col: Column name for text to translate
        output_col: Column name for translated text
    """
    from translateopenai.io import read_excel_data, write_data
    
    df = read_excel_data(input_file)
    
    def write_progress(dataframe):
        """Write progress to output file."""
        write_data(dataframe, output_file)
    
    translate_dataframe(
        api_key_file=api_key_file,
        df=df,
        speaker_col=speaker_col,
        text_col=text_col,
        output_col=output_col,
        language=language,
        context_lines=context_lines,
        model=model,
        apply_at_step=write_progress
    )
    
    # Final write
    write_data(df, output_file)
