import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

import pandas as pd
import statistics
from torch.utils.data import Dataset, DataLoader
import torch
import PIL
import numpy as np
import time

import logging
import sys
import time
import argparse
import os
from collections import defaultdict
import sqlite3
#import whisper
import torchaudio
from datetime import timedelta
import pytube
import hashlib
import whisperx
import gc 
import urllib
import multiprocessing
from typing import List
import translateopenai

def download_init(url):
    logging.info(f"Initializing youtube : {url}")
    yt = pytube.YouTube(url=url)

    hash_file = hashlib.md5()
    hash_file.update(yt.title.encode())
    return yt, f'{hash_file.hexdigest()}.mp4'

def download_video(yt: pytube.YouTube, file_name: str) -> None:
    logging.info(f"Downloading from youtube :  {yt.watch_url}")
    yt.streams.first().download("", file_name,skip_existing=True)
    logging.info(f"Downloaded to {file_name}")

def run_one(inputname: str, temp_files: List[str], args: argparse.Namespace) -> None:
    if inputname.startswith("http"):
        yt, audiofile = download_init(inputname)
        title = yt.title
        output_prefix = os.path.join(args.outdir,title.replace("/", "-").replace("'", "_"))
    else:
        audiofile = inputname
        output_prefix = audiofile
    
    if os.path.exists(output_prefix+".xlsx") and not args.force:
        logging.info(f"Skipping {audiofile} as {output_prefix}.xlsx exists")
        return
    
    if inputname.startswith("http"):
        download_video(yt, audiofile)
        temp_files.append(audiofile)

    transcribe_device = "cpu" if args.device == "mps" else args.device 
    align_device = args.device
    diarize_device = args.device
    batch_size = 16 # reduce if low on GPU mem
    transcribe_compute_type = "float32" if transcribe_device=="cpu" else "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    diarize_compute_type = "float32" if diarize_device=="cpu" else "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    align_compute_type = "float32" if align_device=="cpu" else "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    logging.info("Loading model")
    model = whisperx.load_model("large-v3", transcribe_device, compute_type=transcribe_compute_type)

    align_models={}
    def load_align_model(language: str):
        if language not in align_models:
            logging.info(f"Loading model for language {language}") # before alignment
            try:
                model_a, metadata = whisperx.load_align_model(language_code=language, device=align_device)
            except:
                return None, None
            align_models[language] = (model_a, metadata)
            logging.info(f"  loaded model for language {language}") # before alignment
        return align_models[language]

    logging.info(f"Loading audio from {audiofile}")
    audio = whisperx.load_audio(audiofile)
    #audio = whisperx.audio.pad_or_trim(audio)
    #model = whisper.load_model("large")

    # detect the spoken language


    logging.info("Transcribing") # before alignment
    result = model.transcribe(audio, batch_size=batch_size, language = None if args.language == 'AUTO' else args.language)
    logging.debug(result["segments"]) # before alignment
    logging.info(f"Detected language: {result['language']} ")
    logging.info("Aligning") # before alignment
    model_a, metadata = load_align_model(result['language'])
    if model_a is None:
        logging.warning(f"Tried loading a non-existant language : {result['language']}") # before alignment
        return
    result = whisperx.align(result["segments"], model_a, metadata, audio, align_device, return_char_alignments=False)
    logging.debug(result["segments"]) # after alignment
    logging.info("Diarizing") # before alignment
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=diarize_device)
    diarize_segments = diarize_model(audio)

    logging.info(f"Assigning speakers")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logging.debug(diarize_segments)
    logging.debug(result["segments"]) # segments are now assigned speaker IDs

    segments = result['segments']
    logging.info(f"Transcribe keys : {result.keys()}")
    logging.info(f"Writing txt file : {output_prefix}.txt")

    speakers=defaultdict(int)
    for segment in segments:
        if "speaker" in segment:
            speakers[segment['speaker']]+=1
    logging.info(f"Spekaer phrase counts: {speakers}")
    speakerfile={}
    #for s in speakers.keys():
    #    speakerfile[s] = open(f"{output_prefix}.{s}.txt", "w", encoding="utf-8")
    generic_file = open(f"{output_prefix}.ALL.txt", "w", encoding="utf-8")
    res_table = []
    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        #segmentId = segment['id']+1
        #segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
        #srtFilename = f"{output_prefix}.srt"
        #with open(srtFilename, 'a', encoding='utf-8') as srtFile:
        #    srtFile.write(segment)
        speaker = segment.get("speaker", "UNKNOWN_001")
        #if speaker in speakerfile:
        #    speakerfile[speaker].write(f"{text[1:] if text[0] == ' ' else text}\n")
        generic_file.write(f"{startTime} --> {endTime}\t{speaker}\t{text[1:] if text[0] == ' ' else text}\n")
        res_table.append([startTime, endTime, speaker, text])

    if res_table:
        df = pd.DataFrame(res_table, columns=["starttime", 'endtime', 'speaker', 'text'])
        if args.translate_to_language:
            translateopenai.translate_dataframe(args.translate_api_key_file , df, speaker_col="speaker", text_col="text", output_col="translated", language=args.translate_to_language, context_lines=args.translate_context, model=args.translate_model)        
        df.to_excel(f"{output_prefix}.xlsx", index=False)

def mp_run(parallel: bool, fname: str, args: argparse.Namespace, temp_files: List[str] = []) -> None:
    if parallel:
        try:
            run_one(fname, temp_files, args)
        except Exception as e:
            logging.error(f"Failed to download {fname} : {e}")
        finally:
            for f in temp_files:
                os.remove(f)
    else:
        run_one(fname, temp_files, args)
 

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--force', help="force overwrite", action="store_true")
    parser.add_argument('--input', help=f"what file to use")
    parser.add_argument('--device', choices=["cpu", "cuda", "mps"], default="cpu", help=f"what device to use")
    parser.add_argument('--outdir', default='.', help=f"where to write the output files")
    parser.add_argument('--language', default='AUTO', help="What language to use")
    parser.add_argument('--parallel', type=int, default=4, help="How many threads to run in parallel")
    parser.add_argument('--translate_to_language', help="If set, translate to this language by adding a column to the output file")
    parser.add_argument('--translate_context', type=int,default=40,help="How many lines to process")
    parser.add_argument('--translate_model', default="gpt-3.5-turbo",help="What openai model to use - [gpt-3.5-turbo, gpt-4]")
    parser.add_argument('--translate_api_key_file', default="../openai.key",help="OPENAI_API_KEY file")
    parser.add_argument('files', nargs=argparse.ONE_OR_MORE, help="files to process")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s:%(lineno)d %(message)s')
    logger=logging.getLogger(__name__)
    logger.info(f"Args: {args}")
    for f in args.files:
        if args.parallel>1:
            with multiprocessing.Pool(processes=args.parallel) as pool:
                # Launch the worker function asynchronously and pass the input value
                result = pool.apply(mp_run, args=(True, f, args))        
        else:
            mp_run(False, f,args)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

