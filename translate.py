import translateopenai
import argparse
import logging
import pandas as pd
import sys
from openai import OpenAI

def get_available_models(api_key_file):
    """Get available OpenAI models with fallback to common models."""
    try:
        client = OpenAI(api_key=translateopenai.read_openai_key(api_key_file))
        models = [m for m in [x.id for x in client.models.list().data] if 'gpt' in m]
        if models:
            return models
    except Exception as e:
        logging.warning(f"Could not fetch models from OpenAI API: {str(e)}")
    
    # Fallback to common models
    return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]

def main() -> int:
    # First parse just the API key file and log level to initialize logging
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x))
    pre_parser.add_argument('--api_key_file', default="../aikeys/openai.key")
    temp_args, _ = pre_parser.parse_known_args()
    
    # Setup basic logging for model retrieval
    logging.basicConfig(level=temp_args.log_level, format='%(asctime)s:%(lineno)d %(message)s')
    
    # Get available models
    models = get_available_models(temp_args.api_key_file)
    models_str = ", ".join(models)
    
    # Create main parser with model information in the epilog
    parser = argparse.ArgumentParser(description="Translate text using OpenAI models", 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Add all arguments
    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), 
                       help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--input', help=f"what file to use")
    parser.add_argument('--output', help=f"where to write the output files")
    parser.add_argument('--language', default='english', help="What language to use")
    parser.add_argument('--context', type=int, default=40, help="How many lines to process")
    parser.add_argument('--api_key_file', default="../aikeys/openai.key", help="OPENAI_API_KEY file")
    parser.add_argument('--model', default="gpt-3.5-turbo", choices=models,
                       help=f"What OpenAI model to use")
    
    args = parser.parse_args()
    
    # Reset logging with final log level
    logging.basicConfig(level=args.log_level, format='%(asctime)s:%(lineno)d %(message)s')
    
    df = pd.read_excel(args.input, engine='openpyxl')
    def write_df(x):
        logging.info(f"Writing {x.shape[0]} lines to {args.output}")
        if (args.output.endswith(".xlsx")):
            x.to_excel(args.output, index=False)
        elif (args.output.endswith(".txt")): 
            x.to_csv(args.output, index=False, sep="\t")
    logging.info(f"Translating {df.shape[0]} lines")
    translateopenai.translate_dataframe(args.api_key_file, df, speaker_col="speaker", text_col="text", output_col="translated", language=args.language, context_lines=args.context, model=args.model, apply_at_step=write_df)
    logging.info(f"Done translating")
    write_df(df)
    return 0

if __name__ == '__main__':
    sys.exit(main())

