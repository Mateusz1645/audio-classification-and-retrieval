# src/audio_classification/download_data.py
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Switch Cloud URLs for FMA-Small
METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"


def download_progress_hook(block_num, block_size, total_size):
    """Callback to print download progress percentage."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(
            f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)"
        )
        sys.stdout.flush()


def download_and_extract(url: str, dest_dir: Path) -> None:
    """Downloads a ZIP file from URL and extracts it to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split("/")[-1]
    zip_path = dest_dir / file_name

    # Download if not present
    if not zip_path.exists():
        print(f"\nFetching {file_name} from official FMA server...")
        urllib.request.urlretrieve(
            url, zip_path, reporthook=download_progress_hook
        )
        print(f"\nDownload complete: {zip_path}")
    else:
        print(f"\n{file_name} already exists in {dest_dir}. Skipping download.")

    # Extract archive
    print(f"Extracting {file_name} into {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Successfully extracted {file_name}.\n")


def main():
    # Target path relative to project root
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"

    print(f"Target raw data directory: {raw_data_dir}")

    # 1. Download & Extract Metadata
    # download_and_extract(METADATA_URL, raw_data_dir)

    # 2. Download & Extract FMA-Small Audio
    download_and_extract(AUDIO_URL, raw_data_dir)

    print("Data download and extraction complete!")


if __name__ == "__main__":
    main()