"""Run nnU-Net's official ACDC converter (Dataset027_ACDC.py) unchanged, with one workaround.

Bug in nnU-Net v2.8.1: nnunetv2.paths now exposes nnUNet_raw / nnUNet_preprocessed as _EnvPath
objects, but Dataset027_ACDC.make_out_dirs() still calls nnUNet_raw.replace(...), which crashes
with AttributeError. We hand the module plain strings and then run exactly the logic of its
__main__ block: convert, then write the official patient-level 5-fold split (seed 1234).

Usage: python convert_acdc.py -i <ACDC root containing training/ and testing/>
"""
import argparse, os
import nnunetv2.dataset_conversion.Dataset027_ACDC as acdc
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

acdc.nnUNet_raw = os.environ["nnUNet_raw"]
acdc.nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_folder", required=True)
ap.add_argument("-d", "--dataset_id", type=int, default=27)
args = ap.parse_args()

print("Converting...")
acdc.convert_acdc(args.input_folder, args.dataset_id)
name = f"Dataset{args.dataset_id:03d}_ACDC"
pre = join(acdc.nnUNet_preprocessed, name)
maybe_mkdir_p(pre)
split = acdc.create_ACDC_split(join(acdc.nnUNet_raw, name, "labelsTr"))
save_json(split, join(pre, "splits_final.json"), sort_keys=False)
print("Done! splits_final.json written to", pre)
