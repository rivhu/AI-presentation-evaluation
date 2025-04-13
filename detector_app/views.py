from django.shortcuts import render
from django.http import HttpResponse
import threading
from .confidence_detector import run_confidence_detection


def start_detection(request):
    thread = threading.Thread(target=run_confidence_detection)
    thread.daemon = True  # allows server to shut down even if thread is running
    thread.start()
    return HttpResponse("Confidence detection started. Press 'q' in the window to quit.")
