from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Serve files from the upload folder
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def create_checkered_background(width, height, square_size=20):
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            color = (255, 255, 255) if (i // square_size + j // square_size) % 2 == 0 else (0, 0, 0)
            background[i:i+square_size, j:j+square_size] = color
    return background

def create_black_background(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def process_video(video_path, output_path, background_type='checkered'):
    # Extract audio using ffmpeg
    audio_output_path = os.path.join(UPLOAD_FOLDER, "audio.aac")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'aac', audio_output_path])

    # Initialize video capture with the uploaded video file
    cap = cv2.VideoCapture(video_path)

    # Check if video capture opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get video properties (width, height, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25  # Default FPS

    # Define video writer to save the output (MP4 format)
    temp_video_path = os.path.join(UPLOAD_FOLDER, 'temp_output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Create the background based on the type
    if background_type == 'checkered':
        background = create_checkered_background(width, height)
    else:
        background = np.zeros((height, width, 3), dtype=np.uint8)  # Black background

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation using MediaPipe
        result = segmentation.process(frame_rgb)

        # Get the segmentation mask
        mask = result.segmentation_mask

        # Threshold the mask to make it binary (foreground-background separation)
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

        # Create the foreground (person) by masking the original frame
        fg = cv2.bitwise_and(frame, frame, mask=mask)

        # Only show the background where the mask is 0 (background area)
        # Use the inverse of the mask for the background
        bg = background.copy()
        bg[mask == 1] = frame[mask == 1]  # Replace the background where the person is

        # Combine the foreground (person) and the background
        result_frame = cv2.add(fg, bg)

        # Write the result frame to the output video
        out.write(result_frame)

    cap.release()
    out.release()

    # Merge the audio with the processed video using ffmpeg
    merged_output_path = os.path.join(UPLOAD_FOLDER, 'merged_output.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_output_path,
        '-c:v', 'copy', '-c:a', 'aac', merged_output_path
    ])

    # Re-encode the merged video for compatibility
    final_output_path = os.path.join(UPLOAD_FOLDER, 'download.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', merged_output_path,
        '-c:v', 'libx264', '-c:a', 'aac', '-movflags', 'faststart', final_output_path
    ])

    # Clean up temporary files
    os.remove(audio_output_path)
    os.remove(temp_video_path)
    os.remove(merged_output_path)

    return final_output_path


@app.route('/')
def index():
    return render_template('fileupload.html')

# Flask route to handle the download request
@app.route('/download', methods=['POST'])
def download():
    if 'video' not in request.files:
        print("No file part found in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        print("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    # Log the file name and size
    print("Received file:", file.filename)
    # Save the file temporarily
    filename = secure_filename(file.filename)
    input_video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_video_path)

    # Output path for processed video
    output_video_path = os.path.join(UPLOAD_FOLDER, filename.replace(".mp4", "_black_bg.mp4"))

    try:
        # Process the video to change background to black
        processed_video_path = process_video(input_video_path, output_video_path, background_type='black')
    except Exception as e:
        return jsonify({"error": f"Error processing video: {e}"}), 500

    # Clean up the original uploaded video
    os.remove(input_video_path)

    # Return the path to the processed video with black background
    return jsonify({
        "message": "Video processed with black background successfully",
        "video_url": f"/uploads/{os.path.basename(processed_video_path)}"
    })

@app.route('/convert', methods=['POST'])
def convert():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    input_video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_video_path)

    output_video_path = os.path.join(UPLOAD_FOLDER, filename.replace(".mp4", "_processed.mp4"))

    try:
        processed_video_path = process_video(input_video_path, output_video_path, background_type='checkered')
    except Exception as e:
        return jsonify({"error": f"Error processing video: {e}"}), 500

    os.remove(input_video_path)

    return jsonify({
        "message": "Video processed successfully",
        "video_url": f"/uploads/{os.path.basename(processed_video_path)}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
