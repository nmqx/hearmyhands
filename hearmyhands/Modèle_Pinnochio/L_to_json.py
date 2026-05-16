import cv2
import mediapipe as mp
import time
import mod_json as htm
import json
'''
Permet de créer un Json de squelette en se filmant, 
Enregistre le Json sous "L.json"


Appuyer sur r pour commencer/finir l'enregistrement
Appuyer sur q pour fermer la fenetre
Ce programme sert a enregistrer la position d'UNE MAIN (21 points/42 coordonnées utiles) 
Ca sera la position que l'on veut reconnaitre plus tard.

'''
# Global list to store data from all frames during a recording session
recorded_frames = []

# Use consistent snake_case variable names
previous_time = 0
current_time = 0

# IMPORTANT: Camera index is now correctly set to 0 (Internal Camera)
cap = cv2.VideoCapture(0)

# Set low resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("FATAL ERROR: Camera failed to open at index 0. Check if the device is connected or in use.")
    exit()

detector = htm.HandDetector()

# Define the output file name
OUTPUT_JSON_FILE = "L.json"

# Recording state variable, starts stopped
is_recording = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame during runtime. Exiting.")
        break

    img = detector.find_hands(img)
    # Use the new function to get data for ALL hands and their handedness
    all_hand_info = detector.find_all_positions(img)

    # FPS Calculation
    current_time = time.time()
    if current_time != previous_time:
        fps = 1 / (current_time - previous_time)
    else:
        fps = 0
    previous_time = current_time

    # ----------------------------------------------------
    # Conditional JSON recording (APPENDING to list)
    # ----------------------------------------------------
    if is_recording:
        # Check if any hand is detected
        if len(all_hand_info) > 0:

            # 1. Convert ALL detected hands to the required JSON format for this FRAME
            frame_data = detector.landmarks_to_json(all_hand_info)

            # 2. Add frame number and append to the list
            frame_entry = {
                "frame": len(recorded_frames),
                "timestamp": current_time,
                "hands": frame_data
            }
            recorded_frames.append(frame_entry)

            # Visual cue that recording is active and successful
            text = f"REC: {len(recorded_frames)} frames"
            cv2.putText(img, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Show visual cue that recording is active but no hand is seen
            cv2.putText(img, "WAITING...", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    # Show recording status (REC or STOPPED)
    status_text = "REC" if is_recording else "STOPPED"
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(img, status_text, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    cv2.imshow("image", img)

    # Keypress handling
    key = cv2.waitKey(1) & 0xFF

    # Quit button ('q')
    if key == ord('q'):
        break

    # Toggle recording button ('r')
    if key == ord('r'):
        is_recording = not is_recording

        if is_recording:
            # START RECORDING: Clear previous session data
            recorded_frames.clear()
            print("--- Recording Started (Press 'r' to Stop) ---")
        else:
            # STOP RECORDING: Save all collected frames to the final JSON file
            print(f"--- Recording Stopped. Saving {len(recorded_frames)} frames to {OUTPUT_JSON_FILE} ---")
            if len(recorded_frames) > 0:
                with open(OUTPUT_JSON_FILE, 'w') as f:
                    # Write the entire list of frames, which will generate a large, sequential JSON file.
                    json.dump(recorded_frames, f, indent=2)

cap.release()
cv2.destroyAllWindows()