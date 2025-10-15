import os
import subprocess
from pathlib import Path

def download_coco(target_dir: str = "data/coco"):
    """
    Downloads COCO 2017 train/val images and annotations.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for name, url in urls.items():
        zip_path = target / name
        if not zip_path.exists():
            print(f"Downloading {name} ...")
            subprocess.run(["wget", "-c", url, "-O", str(zip_path)], check=True)
        else:
            print(f"Already downloaded: {name}")

    # Unzip
    print("Extracting archives...")
    for name in urls.keys():
        subprocess.run(["unzip", "-n", str(target / name), "-d", str(target)], check=True)

    print("\n✅ COCO 2017 dataset ready under:", target.resolve())


if __name__ == "__main__":
    download_coco("data/coco")
