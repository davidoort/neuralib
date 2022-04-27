import os
import requests


def download(url: str, file_path: str):
    """Download file from url to file_path."""
    if os.path.exists(file_path):
        print('File already exists: %s' % file_path)
        return
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))