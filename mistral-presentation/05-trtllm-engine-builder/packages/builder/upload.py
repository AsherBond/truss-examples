import argparse
import json
from typing import Dict
import subprocess
import os
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

MAX_TRANSFER_CONCURRENCY = 20

def _multipart_upload_boto3(file_path: str, s3_bucket: str, s3_key: str, credentials: Dict):
    s3_resource = boto3.resource("s3", **credentials)
    filesize = os.stat(file_path).st_size

    with tqdm(
        total=filesize,
        desc="Upload",
        unit="B",
        unit_scale=True,
    ) as pbar:
        s3_resource.Object(s3_bucket, s3_key).upload_file(
            file_path,
            Config=TransferConfig(
                max_concurrency=MAX_TRANSFER_CONCURRENCY,
                use_threads=True,
            ),
            Callback=pbar.update,
        )


def tar_directory(directory_path: str, output_filename: str):
    print(f"Archiving {directory_path}")
    subprocess.Popen(["tar", "-c", "-f", output_filename, directory_path]).communicate()

def upload(
    filename: str,
    s3_upload_info: dict,
):
    _multipart_upload_boto3(filename, s3_upload_info.get("s3_bucket"), s3_upload_info.get("s3_key"), s3_upload_info.get("credentials"))

def main():
    parser = argparse.ArgumentParser(description='Tar a directory and upload it to S3 using a presigned POST URL.')
    parser.add_argument('--directory', type=str, help='Path to the directory to tar and upload.')
    parser.add_argument('--s3_upload_info_json_str', type=str, help='S3 credentials and upload information')
    args = parser.parse_args()

    s3_upload_info = json.loads(args.s3_upload_info_json_str)
    output_filename = os.path.basename(args.directory.rstrip(os.sep)) + '.tar'
    tar_directory(args.directory, output_filename)

    print(f"Uploading {output_filename} to S3...")
    upload(output_filename, s3_upload_info)

if __name__ == "__main__":
    main()