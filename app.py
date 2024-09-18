from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import tempfile
from werkzeug.utils import secure_filename  # Add this import


app = Flask(__name__)

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

def process_video(video_path, output_path):
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

    # Background color (black)
    background = np.zeros((height, width, 3), dtype=np.uint8)

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

        # Create the background (black)
        bg_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(background, background, mask=bg_mask)

        # Combine the foreground and the background
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

    # Now re-encode the merged video to ensure compatibility with browsers
    final_output_path = os.path.join(UPLOAD_FOLDER, 'final_output_with_audio.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', merged_output_path,
        '-c:v', 'libx264', '-c:a', 'aac', '-movflags', 'faststart', final_output_path
    ])

    # Clean up the temporary files
    os.remove(audio_output_path)
    os.remove(temp_video_path)
    os.remove(merged_output_path)

    return final_output_path


@app.route('/')
def index():
    return render_template('fileupload.html')

# Flask route to handle the video conversion
@app.route('/convert', methods=['POST'])
def convert():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a temporary location in uploads folder
    filename = secure_filename(file.filename)
    input_video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_video_path)

    # Output path for processed video
    output_video_path = os.path.join(UPLOAD_FOLDER, filename.replace(".mp4", "_processed.mp4"))

    try:
        # Process the video to remove the background
        processed_video_path = process_video(input_video_path, output_video_path)
    except Exception as e:
        return jsonify({"error": f"Error processing video: {e}"}), 500

    # Optionally clean up the original uploaded video
    os.remove(input_video_path)

    # Return the success message with the path to the processed video
    return jsonify({
        "message": "Video processed successfully",
        "video_url": f"/uploads/{os.path.basename(processed_video_path)}"  # Use the correct uploads path
    })

if __name__ == '__main__':
    app.run(debug=True)
