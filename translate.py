
import translateopenai
import argparse
import logging
import pandas as pd
import sys
def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--input', help=f"what file to use")
    parser.add_argument('--output', help=f"where to write the output files")
    parser.add_argument('--language', default='english', help="What language to use")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s:%(lineno)d %(message)s')
    
    df = pd.read_excel(args.input, engine='openpyxl')
    def write_df(x):
        if (args.output.endswith(".xlsx")):
            x.to_excel(args.output, index=False)
        elif (args.output.endswith(".txt")): 
            x.to_csv(args.output, index=False, sep="\t")
    translateopenai.translate_dataframe("../openai.key", df, speaker_col="speaker", text_col="text", output_col="translated", language=args.language, context_lines=2, model="gpt-3.5-turbo", apply_at_step=write_df)
    write_df(df)
    return 0

if __name__ == '__main__':
    sys.exit(main())

