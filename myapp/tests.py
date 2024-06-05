from django.test import TestCase

# Create your tests here.
# from .views import *
import os
from django.conf import settings

def get_latest_screenshot():
    # Path to the screenshots folder in your Django project
    screenshots_folder = os.path.join(settings.BASE_DIR, 'screenshots')
    
    # List all files in the folder
    files = os.listdir(screenshots_folder)
    
    # Filter out only image files if needed
    image_files = [file for file in files if file.lower().endswith('.png')]
    
    if not image_files:
        print("No image files found in the screenshots folder.")
        return None
    
    # Sort the image files based on their names
    image_files.sort(key=lambda x: int(x.lstrip('screenshot').rstrip('.png')))
    
    if not image_files:
        print("No screenshot files found.")
        return None
    
    # Get the last (latest) file from the sorted list
    latest_screenshot = image_files[-1]
    
    # Get the full path to the latest screenshot
    latest_screenshot_path = os.path.join(screenshots_folder, latest_screenshot)
    
    return latest_screenshot_path

# Call the function to get the path of the latest screenshot
latest_screenshot_path = get_latest_screenshot()

if latest_screenshot_path:
    print("Latest screenshot path:", latest_screenshot_path)
get_latest_screenshot()