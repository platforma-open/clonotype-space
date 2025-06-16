import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='Check if input table is empty.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input file')
    parser.add_argument('-s', '--input-separator', default='\t',
                        help='Input table file separator (default: "\t")')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to save output files (default: current directory)')
    args = parser.parse_args()

    # Open input file
    df_input = pd.read_csv(args.input, sep=args.input_separator, dtype=str)

    # Check if input table is empty
    fileContent = "empty"
    if df_input.empty:
        print("Input table is empty.")
        fileContent = "empty"
    else:
        print("Input table is not empty.")
        fileContent = "notEmpty"

    with open(os.path.join(args.output_dir, 'isFileEmpty.txt'), 'w') as f:
        f.write(fileContent)

if __name__ == '__main__':
    main()