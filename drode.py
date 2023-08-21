# Import necessary libraries
from scipy.spatial import distance    # Library for calculating distances between points
from imutils import face_utils       # Utilities for working with facial landmarks
from pygame import mixer             # Library for playing sound (used for the alert)
import imutils                       # Utility functions for working with images
import dlib                          # Library for face detection and shape prediction
import cv2                           # OpenCV library for computer vision tasks

# Initialize the mixer and load music file for the alert sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate eye aspect ratio (ear) from 6 facial landmarks of an eye
def eye_aspect_ratio(eye):
    # Calculate the distances between specific points in the eye region
    A = distance.euclidean(eye[1], eye[5])   # Vertical distance
    B = distance.euclidean(eye[2], eye[4])   # Vertical distance
    C = distance.euclidean(eye[0], eye[3])   # Horizontal distance
    
    # Calculate the eye aspect ratio (ear) as the average of two ratios
    ear = (A + B) / (2.0 * C)
    return ear

# Set the threshold and frame check value for determining eye closure
thresh = 0.25
frame_check = 20

# Initialize face detector and shape predictor using pre-trained models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# Define the indices of left and right eye landmarks from the 68 face landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open a video capture object to capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize a flag to keep track of consecutive frames with low eye aspect ratio
flag = 0

# Main loop for processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Resize the frame to a smaller width for faster processing
    frame = imutils.resize(frame, width=450)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using dlib's face detector
    subjects = detect(gray, 0)
    
    # Process each detected face
    for subject in subjects:
        # Predict facial landmarks for the current face
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye regions from the facial landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate the eye aspect ratio (ear) for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Average the eye aspect ratios for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw convex hulls around the eyes for visualization
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check if the eye aspect ratio is below the threshold (indicating closed eyes)
        if ear < thresh:
            flag += 1
            print(flag)
            
            # If consecutive frames have low eye aspect ratio, trigger an alert
            if flag >= frame_check:
                cv2.putText(frame, "*****ALERT!*****", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*****ALERT!*****", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()  # Play the alert sound
                
        else:
            flag = 0  # Reset the flag if eyes are open
        
    # Show the processed frame with any alerts and eye regions drawn
    cv2.imshow("Frame", frame)
    
    # Check for user input to exit the loop (press 'q' to quit)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close the video capture and destroy any open windows
cv2.destroyAllWindows()
cap.release()