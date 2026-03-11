import random
import subprocess
from pathlib import Path
from time import sleep
from io import BytesIO

from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image, ImageChops, ImageDraw
from selenium.common import ElementClickInterceptedException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from seleniumwire import webdriver  # Import from seleniumwire


def m3u8_from_clickable_element(url, clickable_element_xpath):
    """ Downloads first m3u8 found, sleeps 20-35 secs for loading purposes"""
    # Create a new instance of the Chrome driver
    options = Options()
    options.headless = True
    try:
        web_driver = webdriver.Chrome("/usr/bin/chromedriver", options=options)    # Ubuntu
    except (WebDriverException, TypeError):
        web_driver = webdriver.Chrome(options=options)  # MacOS
    web_driver.get(url)
    sleep(random.randint(5, 10))    # Wait until it is fully loaded
    if clickable_element_xpath is not None:
        element = web_driver.find_element(By.XPATH, clickable_element_xpath)
        try:
            element.click()
        except ElementClickInterceptedException:
            web_driver.execute_script("arguments[0].click();", element)
    sleep(random.randint(20, 20))   # Wait until ad is gone
    # Save all api-related requests
    api_requests = [r for r in web_driver.requests
                    if '.m3u8' in r.url and r.response]
    return api_requests


def video_and_image_from_m3u8(stream_url, seconds, video_file_path, image_file_path):
    # Download stream
    subprocess.run(
        f'ffmpeg -y -i "{stream_url}" -t {seconds} -c copy -f mp4 "{video_file_path}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Capture image
    subprocess.run(
        f'ffmpeg -y -i "{video_file_path}" -vcodec mjpeg -vframes 1 -an -f rawvideo -ss {seconds / 2} "{image_file_path}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def _build_polygon_mask_image(size, polygon_points):
    mask_image = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(
        [(point['x'], point['y']) for point in polygon_points],
        fill=255,
    )
    return mask_image


def _save_mask_image_to_field(instance, field_name, image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    filename = (
        f'{instance.camera_slug}_{field_name}_'
        f'{timezone.now().strftime("%Y%m%d%H%M%S")}.png'
    )

    getattr(instance, field_name).save(
        filename,
        ContentFile(buffer.getvalue()),
        save=False,
    )


def save_polygon_mask(webcam, mask_field_name, polygon_points, reference_image_path):
    allowed_mask_fields = {
        'mask_sand',
        'mask_swimming_water',
    }

    if mask_field_name not in allowed_mask_fields:
        raise ValueError(f'Unsupported mask field: {mask_field_name}')

    with Image.open(reference_image_path) as reference_image:
        size = reference_image.size

    mask_image = _build_polygon_mask_image(size, polygon_points)
    _save_mask_image_to_field(webcam, mask_field_name, mask_image)

    merged_mask = Image.new('L', size, 0)

    if webcam.mask_sand and getattr(webcam.mask_sand, 'path', None):
        with Image.open(webcam.mask_sand.path) as sand_image:
            merged_mask = ImageChops.lighter(merged_mask, sand_image.convert('L'))

    if webcam.mask_swimming_water and getattr(webcam.mask_swimming_water, 'path', None):
        with Image.open(webcam.mask_swimming_water.path) as swimming_image:
            merged_mask = ImageChops.lighter(merged_mask, swimming_image.convert('L'))

    _save_mask_image_to_field(webcam, 'mask_sand_and_water', merged_mask)

    webcam.save(update_fields=[
        mask_field_name,
        'mask_sand_and_water',
    ])