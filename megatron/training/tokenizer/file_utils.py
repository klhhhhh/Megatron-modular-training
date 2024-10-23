import os
import requests

def cached_path(url_or_filename, cache_dir=None):
    """
    Given a URL or local path, this function will:
    - If it's a local path, return the path as is.
    - If it's a URL, download the file and cache it locally.
    """
    if os.path.exists(url_or_filename):
        # If the file exists locally, return the path
        return url_or_filename
    else:
        # It's a URL, so download and cache it
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            local_filename = os.path.join(cache_dir, os.path.basename(url_or_filename))
        else:
            local_filename = os.path.basename(url_or_filename)
        
        # Download file from the URL
        print(f"Downloading {url_or_filename} to {local_filename}")
        response = requests.get(url_or_filename, stream=True)
        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            raise EnvironmentError(f"Failed to download file from {url_or_filename}, status code {response.status_code}")
        
        return local_filename
