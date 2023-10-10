import pandas as pd
import openai
from typing import List
import logging
import time
from contextlib import contextmanager
import logging

@contextmanager
def timer(label="Block of code"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.log(logging.INFO, f"{label} executed in {elapsed_time:.6f} seconds")



def save_to_excel(data: List[dict], file_name: str) -> None:
    df = pd.DataFrame(data, columns=["starttime", 'endtime', 'speaker', 'text'])
    df.to_excel(file_name, index=False)

def translate_to(language, sentence, model):
    # Ensure you have set up your OpenAI API key before calling this
    response = openai.ChatCompletion.create(
        model =  model,
        temperature = 0,
        messages = [
            {"role": "user", "content": f"Translate the following text while maintaining format and do not merge lines to {language}: '{sentence}'"}
        ]
    )
    return response.choices[0]["message"]["content"]

def read_openai_key(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.readline().strip()

def translate_dataframe(keyfile, df: pd.DataFrame, speaker_col: str, text_col: str, output_col:str, language: str, context_lines=40, model="gpt-3.5-turbo", apply_at_step=None):
    openai.api_key = read_openai_key(keyfile)
    logging.log(logging.DEBUG, f"Translating {df.shape[0]} lines")
    step=context_lines
    out=[]
    df[output_col] = ["" for i in range(df.shape[0])]
    for i in range(0, df.shape[0],step):
        with timer(f"Translating lines {i} to {i+step}"):
            slice = df.iloc[i:i+step][[speaker_col,text_col]]
            slice["merged"] = slice.index.astype(str) + "|" + slice[speaker_col] + "|" + slice[text_col]
            line = "\n".join(list(slice["merged"]))
            tries = 0
            while True:
                tries += 1
                try:
                    output = translate_to(language, f"\n{line}", model)
                    break
                except openai.error.APIError as e:
                    logging.error(f"Failed to translate {i} : {e} - retrying")
                except openai.error.ServiceUnavailableError as e:
                    logging.error(f"Failed to translate {i} : {e} - retrying")
            lines = output.split("\n")
            for l in lines:
                idx, _ , text = l.split("|")
                df[output_col][int(idx)] = text.strip()
            logging.log(logging.INFO, f"Translated {i} to {i+step} in {tries} tries: \n {df.iloc[i:i+step]}")
            if apply_at_step is not None:
                apply_at_step(df)
