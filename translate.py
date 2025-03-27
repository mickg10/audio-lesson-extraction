import sys
import translateopenai

def main() -> int:
    # Create parser with standard translation options
    parser, temp_args = translateopenai.create_translation_parser(
        description="Translate text using OpenAI models",
        default_api_key="../aikeys/openai.key"
    )
    
    # Setup logging
    translateopenai.configure_logging(temp_args.log_level)
    
    # Add application-specific arguments
    parser.add_argument('--input', required=True, help="Input file to process (Excel format)")
    parser.add_argument('--output', required=True, help="Where to write the output files")
    
    # Add language, context, and model arguments
    translateopenai.add_translation_arguments(parser)
    translateopenai.add_model_argument(parser, temp_args.api_key_file)
    
    args = parser.parse_args()
    
    # Reset logging with final log level
    translateopenai.configure_logging(args.log_level)
    
    # Process the file
    translateopenai.translate_file(
        input_file=args.input,
        output_file=args.output,
        api_key_file=args.api_key_file,
        language=args.language,
        context_lines=args.context,
        model=args.model
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
