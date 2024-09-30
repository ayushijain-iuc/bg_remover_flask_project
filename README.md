                                                      Flask Video Background Replacement Application

Overview
This application allows users to upload a video and replace its background with a black background using MediaPipe's Selfie Segmentation. 
It extracts the audio from the original video, processes the frames to isolate the foreground (the person), 
and merges the audio back into the final output video. The processed video can then be downloaded through the web interface.



Features
Video Upload: Users can upload a video file (in MP4 format).
Background Replacement: The application uses MediaPipe's Selfie Segmentation to isolate the foreground and replace the background with a black background.
Audio Extraction: The application extracts audio from the uploaded video and merges it with the processed video.
Download Processed Video: Users can download the processed video with the background changed.


Technologies Used
Flask: A lightweight WSGI web application framework for Python.
OpenCV: A computer vision library for real-time image and video processing.
MediaPipe: A cross-platform framework for building multimodal applied machine learning pipelines.
FFmpeg: A command-line tool for handling multimedia data.
NumPy: A library for numerical computing in Python.


Important Notes
Ensure that the uploaded video is in MP4 format for compatibility.
The application processes videos with a default black background but can be modified for other background types.
Clean up temporary files after processing to free up storage.
