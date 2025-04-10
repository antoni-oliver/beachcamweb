import random
import subprocess
from time import sleep

from selenium.common import ElementClickInterceptedException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from seleniumwire import webdriver  # Import from seleniumwire


def m3u8_from_url(url, clicable_element_xpath):
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
    element = web_driver.find_element(By.XPATH, clicable_element_xpath)
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