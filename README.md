# Table of contents
- [Pose Detector](#pose-detector)
  - [This code is a Python script that uses the Mediapipe library and machine learning models to classify yoga poses in real-time from a video source. It utilizes pose detection from Mediapipe to extract important landmarks from the human body, and then uses a pre-trained machine learning model to classify the pose based on these landmarks.](#this-code-is-a-python-script-that-uses-the-mediapipe-library-and-machine-learning-models-to-classify-yoga-poses-in-real-time-from-a-video-source-it-utilizes-pose-detection-from-mediapipe-to-extract-important-landmarks-from-the-human-body-and-then-uses-a-pre-trained-machine-learning-model-to-classify-the-pose-based-on-these-landmarks)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Code Overview](#code-overview)
    - [Importing Libraries and Dependencies](#importing-libraries-and-dependencies)
    - [Important Landmarks](#important-landmarks)
    - [Extracting Important Landmarks](#extracting-important-landmarks)
    - [Rescaling the Frame](#rescaling-the-frame)
    - [Loading Machine Learning Models and Scaler](#loading-machine-learning-models-and-scaler)
    - [Recognizing Poses](#recognizing-poses)
  - [Conclusion](#conclusion)
- [Credits](#credits)


## &nbsp;



# Pose Detector

using Mediapipe library and machine learning model, this code classifies yoga poses in real-time from a video source. It utilizes pose detection from Mediapipe to extract important landmarks from the human body, and then uses a pre-trained machine learning model to classify the pose based on these landmarks.
- 
## Dependencies

The following libraries are required to run the code:
- `mediapipe`: Used for pose detection.
- `cv2` (OpenCV): Used for image and video processing.
- `numpy`: Used for numerical operations.
- `pandas`: Used for data manipulation.
- `seaborn`: Used for data visualization.- [Live Pose Documentation]
- `csv`: Used for reading and writing CSV files.
- `os`: Used for file operations.
- `pickle`: Used for serializing and deserializing Python objects.
- `scikit-learn`: Used for machine learning models and utilities.

Make sure these libraries are installed in your Python environment before running the code. You can use the `!pip install` command to install any missing dependencies.

## Usage

To use this code, follow these steps:

1. Install the required dependencies mentioned above.
2. Make sure you have a video file that contains yoga pose sequences. Update the `cap = cv2.VideoCapture("goddess-pose.mp4")` line to specify the path to your video file.
3. Run the code.

The code will open a window showing the video and display the current recognized pose and its probability in the top-left corner. The recognized pose is classified into two categories: "Goddess" and "Warrior". If the recognized pose does not belong to either category or the prediction probability is below a certain threshold, it will be labeled as "unk" (unknown).

## Code Overview

### Importing Libraries and Dependencies

The code starts by importing all the required libraries and dependencies. It includes Mediapipe, OpenCV, numpy, pandas, seaborn, csv, os, pickle, and scikit-learn modules. These libraries are necessary for performing pose detection, image processing, data manipulation, and machine learning operations.

### Important Landmarks

A list of important landmarks is defined. These landmarks correspond to specific body parts detected by Mediapipe, such as nose, shoulders, elbows, wrists, etc. These landmarks will be used to extract landmarks for classification.

### Extracting Important Landmarks

The `extract_important_landmarks(results)` function takes the results of pose detection and extracts the important landmarks for classification. It loops through the list of important landmarks and retrieves the x, y, z coordinates, and visibility of each landmark. The landmarks are returned as a flattened list.

### Rescaling the Frame

The `rescale_frame(frame, percent=50)` function resizes a frame to a specified percentage of its original size. It takes a frame and a percentage value as input and returns the resized frame.

### Loading Machine Learning Models and Scaler

The code loads a pre-trained machine learning model and an input scaler using the `pickle` library. The machine learning model is stored in the file "all_sklearn.pkl", and the input scaler is stored in the file "input_scalerall.pkl". Make sure these files exist in the same directory as the script.

### Recognizing Poses

The code initializes the video capture object `cap` with the path to the video file. It then enters a loop to process each frame of the video.

Inside the loop, it reads the next frame from the video and performs the following operations:

1. Rescales the frame to a smaller size.
2. Converts the color space of the frame from BGR to RGB for pose detection.
3. Processes the frame using Mediapipe pose detection to obtain the pose landmarks.
4. Checks if any pose landmarks are detected in the frame. If not, it continues to the next frame.
5. Converts the color space of the frame back to BGR for display purposes.
6. Draws the landmarks and connections on the frame using Mediapipe.
7. Extracts the important landmarks from the pose detection results.
8. Creates a pandas DataFrame from the landmarks.
9. Makes a prediction using the pre-trained machine learning model.
10. Evaluates the prediction and sets the current pose label based on the prediction probability and threshold.
11. Displays the current pose label and prediction probability on the frame.
12. Shows the frame in a window.
13. Waits for the 'q' key to be pressed to exit the loop and close the window.

After processing all frames, the video capture object is released, and all windows are closed.

## Conclusion

This code demonstrates how to perform real-time yoga pose classification using pose detection and machine learning models. By extracting important landmarks from the detected pose, it uses a pre-trained model to predict and classify the pose into specific categories. You can customize the code to use your own video files and adapt it to different pose classification tasks.

&nbsp;
# Credits
This project was developed by four students from Sapienza University in Applied Computer Science and Artificial Intelligence as part of the AI Lab course.

the credits belongs to:

- `prasad.1968913@studenti.uniroma1.it`
- `kumar.1985864@studenti.uniroma1.it`
- `khabbazian.1981002@studenti.uniroma1.eu`
- `manku.1938352@studenti.uniroma1.it`

# Personal-Yoga-trainer
