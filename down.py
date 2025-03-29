import requests

# URLs of the required files
cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Function to download files
def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Downloaded {filename}")

# Download the files
download_file(cfg_url, "yolov3.cfg")
download_file(weights_url, "yolov3.weights")
download_file(names_url, "coco.names")
