import sys
import csv

import io
import subprocess
import json
import cv2
import os

from os import listdir
from os.path import isfile, join

# output folder
OUTPUT_DIR = "output"

file_name = "results.csv"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fields = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]

# Get image path
image_path = sys.argv[5]

ref_country = ["us", "eu", "au"]

results = []

for country in ref_country:
    # Execute subprocess command for openAlpr pipeline
    # This is equivalent to the following when run on terminal shell
    # alpr -j -c <country> <image>
    proc = subprocess.Popen(
        ["alpr", "-j", "-c", country, image_path], stdout=subprocess.PIPE)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        try:
            out = json.loads(line)
            if len(out["results"]) > 0:
                # Keep results in an array
                results += out["results"]

        except ValueError:
            continue
    proc.terminate

#load image for extraction
image = cv2.imread(image_path)

# Calculate best results based on confidence level
best_results = {}
for result in results:
    r = best_results.get(result["plate_index"])
    confidence = result["confidence"]
    if r is not None:
        if confidence > r["confidence"]:
            current_confidence = confidence
            coordinates = result["coordinates"]
            plate = result["plate"]
    else:
        best_results[result["plate_index"]] = {
            "confidence": confidence,
            "coordinates": result["coordinates"],
            "plate": result["plate"]
        }

# Generate cropped plate
for key, value in best_results.items():
    coordinates = value["coordinates"]
    plate = value["plate"]

    print("Detected: {}".format(plate))

    # Save image inside OUTPUT_DIR
    plate_snapshot = '{}/{}.jpg'.format(OUTPUT_DIR, plate)
    cv2.imwrite(
        plate_snapshot, image[coordinates[0]["y"]:coordinates[2]["y"],
                              coordinates[0]["x"]:coordinates[2]["x"]])

    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields + [plate_snapshot, plate])

if len(results) == 0:
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields + ["NA", "NA"])
