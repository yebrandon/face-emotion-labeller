import os
import csv
from deepface import DeepFace

IMGS_DIR_PATH = "images/"
results_count = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0,
    "several_faces_found": 0,
    "face_not_found": 0,
    "total_images_successful": 0,
    "total_images_processed": 0,
}

# Write dominant emotion and emotion confidences for each image file to CSV
with open("results.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for filename in os.listdir(IMGS_DIR_PATH):
        try:
            result = DeepFace.analyze(
                img_path=(IMGS_DIR_PATH + filename),
                actions="emotion",
            )
        except ValueError:
            results_count["face_not_found"] += 1
            continue

        if len(result) > 1:
            results_count["several_faces_found"] += 1
            continue

        emotion_confidences = result[0]["emotion"]
        dom_emotion = result[0]["dominant_emotion"]
        dom_emotion_confidence = emotion_confidences[dom_emotion]
        results_count[dom_emotion] += 1
        results_count["total_images_successful"] += 1

        writer.writerow(
            [filename, dom_emotion, dom_emotion_confidence, emotion_confidences]
        )

# Write tally of emotions detected to CSV
with open("results_summary.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for key in results_count.keys():
        writer.writerow([key, results_count[key]])
