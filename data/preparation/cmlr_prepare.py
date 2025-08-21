import sys, os, glob, subprocess, shutil, math
# from datetime import timedelta
# import tempfile
# from collections import OrderedDict
# # from pydub import AudioSegment
from tqdm import tqdm
import logging
# import ffmpeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def uncompress_cmlr(corpus_dir, output_dir):
    """
    Uncompress CMLR-CORPUS dataset
    """
    if not os.path.exists(corpus_dir):
        logging.error(f"{corpus_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for dir in ["audio", "video"]:
        dir_path = os.path.join(corpus_dir, dir)
        if not os.path.exists(dir_path):
            logging.error(f"{dir_path} does not exist.")
            return
    
        os.makedirs(f"{output_dir}/{dir}", exist_ok=True)
        for zip_file in glob.glob(os.path.join(dir_path, "*.zip")):
            logging.info(f"Uncompressing {zip_file}...")
            subprocess.run(["unzip", "-o", zip_file, "-d", f"{output_dir}/{dir}"], check=True)
    
    logging.info(f"Uncompressing text.zip...")
    text_zip = os.path.join(corpus_dir, "text.zip")
    if os.path.exists(text_zip):
        subprocess.run(["unzip", "-o", text_zip, "-d", output_dir], check=True)
    else:
        logging.error(f"{text_zip} does not exist.")

    # Copy train.csv, val.csv, test.csv
    for file in ["train.csv", "val.csv", "test.csv"]:
        if file in os.listdir(corpus_dir):
            logging.info(f"Copying {file} to {output_dir} ...")
            shutil.copy(os.path.join(corpus_dir, file), output_dir)
        else:
            logging.error(f"Warning: {file} not found in {corpus_dir}.")


def create_train_val_test_dataset(data_dir):
    if not os.path.exists(data_dir):
        logging.error(f"{data_dir} does not exist.")
        return

    for split in ["train", "val", "test"]:
        csv_file = os.path.join(data_dir, split + ".csv")
        
        with open(csv_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]  
        
        logging.info(f"{csv_file} loaded {len(lines)} samples")

        for line in tqdm(lines, desc=f"Creating {split} subset", unit="file"):
            try:
                #split info from csv
                speaker, sub = line.split("/", 1)
                date, file = sub.split("_", 1)
 
                src_audio = os.path.join(data_dir, "audio", speaker, date, file + ".wav")
                src_video = os.path.join(data_dir, "video", speaker, date, file + ".mp4")
                src_text  = os.path.join(data_dir, "text",  speaker, date, file + ".txt")

                dst_dir = os.path.join(data_dir,"datasets", split, speaker, date)
                os.makedirs(dst_dir, exist_ok=True)

                for src_path in [src_audio, src_video, src_text]:
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_dir)
                    else:
                        logging.error(f"[{split}] Missing file: {src_path}")

            except Exception as e:
                logging.error(f"[{split}] Failed to process line: {line} - {e}")


def check_train_val_test_dataset(data_dir, output_dir):
    print(data_dir, output_dir)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='CMLR preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--step', type=int, help='Steps 0: uncompress CMLR corpus, 2: split datasets, 3: prep audio for trainval/test, 4: get labels and file list)')
    #step 0 [Only for uncompressing]:
    parser.add_argument('--corpus', type=str, help='CMLR corpus directory')
    parser.add_argument('--corpus_output', type=str, help='Output directory for processed data after uncompressing, if you already uncompressed, use --step 2')
    
    #if you already uncompressed datasets, use other steps:
    parser.add_argument('--cmlr', type=str, help='cmlr datasets dir')
    # parser.add_argument('--rank', type=int, help='rank id')
    # parser.add_argument('--nshard', type=int, help='number of shards')
    args = parser.parse_args()

    if args.step == 0:
        logging.info(f"Uncompressing CMLR-CORPUS dataset from {args.corpus} to {args.corpus_output}")
        uncompress_cmlr(args.corpus, args.corpus_output)

    elif args.step == 1:
        logging.info(f"Splitting CMLR datasets into train, val, test")
        create_train_val_test_dataset(args.cmlr)



# /workspace/shuaque/Data/corpus/CMLR-CORPUS1
        
#     elif args.step == 3:
#         print(f"Extracting audio for trainval/test")
#         prep_wav(args.cmlr, args.ffmpeg, args.rank, args.nshard)
#     elif args.step == 4:
#         get_file_label(args.cmlr)

# """
# python3 cmlr_prepare.py --cmlr /workspace/shuaque/Data/datasets/CMLR --step 2

# """