from test_one_line_import import inference
import argparse

parser = argparse.ArgumentParser(description="Run inference on images and save masks.")
parser.add_argument("--input_dir", type=str, help="Path to the input image directory")
parser.add_argument("--output_dir", type=str, help="Path to the input image directory")
args = parser.parse_args()


"""
only need to run "mask = inference(args.input_dir, args.output_dir)"
"""
mask = inference(args.input_dir, args.output_dir)








