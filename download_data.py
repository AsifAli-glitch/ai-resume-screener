import os
import sys
import urllib.request

def download_from_url(url, dest_path):
    print(f"Attempting to download dataset from: {url}")
    try:
        # Add a User-Agent header to avoid blocks from CDNs
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as out_file:
                chunk_size = 1024 * 64
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
        print(f"Successfully downloaded {downloaded} bytes to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
        return False

def download_via_kaggle(download_dir):
    print("Attempting to download dataset via Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        dataset = "gauravduttakiit/resume-dataset"
        print(f"Downloading Kaggle dataset '{dataset}'...")
        api.dataset_download_files(dataset, path=download_dir, unzip=True)
        print("Kaggle download complete.")
        return True
    except BaseException as e:
        print(f"Kaggle API download failed: {e}")
        return False

def download_dataset():
    download_dir = "./data"
    dest_path = os.path.join(download_dir, "UpdatedResumeDataSet.csv")
    os.makedirs(download_dir, exist_ok=True)
    
    # 1. Try public GitHub URLs first (doesn't require credentials)
    public_urls = [
        "https://raw.githubusercontent.com/mohansaidinesh/Machine-Learning/main/UpdatedResumeDataSet.csv",
        "https://raw.githubusercontent.com/raghavendranhp/Resume_screening/master/UpdatedResumeDataSet.csv"
    ]
    
    for url in public_urls:
        if download_from_url(url, dest_path):
            print(f"Download complete. Files saved in '{download_dir}'.")
            return
            
    # 2. Fallback to Kaggle API (requires ~/.kaggle/kaggle.json)
    if download_via_kaggle(download_dir):
        # Verify if the expected CSV is present
        if os.path.exists(dest_path):
            print(f"Download complete. Files saved in '{download_dir}'.")
            return
            
    # 3. If everything failed, print manual download instructions and exit
    print("\n" + "="*60)
    print("ERROR: All dataset download attempts failed.")
    print("="*60)
    print("To resolve this, please download the dataset manually:")
    print("1. Visit: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset")
    print("2. Download and unzip the file.")
    print("3. Place the CSV file at:")
    print(f"   {os.path.abspath(dest_path)}")
    print("="*60 + "\n")
    sys.exit(1)

if __name__ == "__main__":
    download_dataset()
