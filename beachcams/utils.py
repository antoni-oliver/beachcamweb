import datetime
import os
import pathlib
import subprocess
from datetime import timedelta


def data_dir():
    path = pathlib.Path(__file__).parent.parent.resolve()
    return f'{path}/media'


def generate_filename(provider_name, service_name, ts_rounded):
    folder = f'{data_dir()}/{provider_name}/{service_name}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{folder}/{ts_rounded:.0f}'
    return filename


def is_multiple(rounded_timestamp: int, desired_frequence: timedelta):
    base_timestamp = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    return (rounded_timestamp - base_timestamp) % desired_frequence.total_seconds() == 0


def download_m3u8(stream_url: str, filepath_without_extension: str, seconds: int = 10) -> list:
    # Download stream
    video_file_path = f'{filepath_without_extension}.mp4'
    image_file_path = f'{filepath_without_extension}.jpg'
    subprocess.run(
        f'ffmpeg -y -i "{stream_url}" -t {seconds} -c copy -f mp4 "{video_file_path}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Create thumbnail
    subprocess.run(
        f'ffmpeg -y -i "{video_file_path}" -vcodec mjpeg -vframes 1 -an -f rawvideo -ss {seconds/2} "{image_file_path}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    return [video_file_path, image_file_path]