from flask import Flask, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Function to process video and handle audio
def process_video(video_path, output_path):
    # Extract audio from the original video
    audio_output_path = "audio.aac"
    subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'aac', audio_output_path])

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = 'temp_video.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    background = np.zeros((height, width, 3), dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmentation.process(frame_rgb)
        mask = result.segmentation_mask

        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(background, background, mask=bg_mask)
        result_frame = cv2.add(fg, bg)

        out.write(result_frame)

    cap.release()
    out.release()

    # Merge audio with the processed video
    subprocess.run(['ffmpeg', '-i', temp_video_path, '-i', audio_output_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_path])

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(audio_output_path)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    input_video_path = os.path.join('uploads', filename)
    file.save(input_video_path)

    output_video_path = os.path.join('uploads', 'processed_' + filename)
    try:
        process_video(input_video_path, output_video_path)
        return send_file(output_video_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)


