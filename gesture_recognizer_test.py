import datetime

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.models import load_model
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import pyautogui


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('./hand-gesture-recognition-code/mp_hand_gesture')

# Load class names
f = open('./hand-gesture-recognition-code/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

# get audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()

# volume ranges
volume_levels = [-96,-34.75468063354492,-24.35443115234375,-18.236774444580078,-13.886737823486328,-10.508596420288086,-7.746397495269775,-5.409796714782715,-3.384982109069824,-1.5984597206115723,0]

attention_flag = False
start_attention = None

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)


    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''
    #post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms,
                                  mpHands.HAND_CONNECTIONS)

        # Predict gesture in Hand Gesture Recognition project
        prediction = model.predict([landmarks], verbose=0)
        classID = np.argmax(prediction)
        className = classNames[classID]

        #print(className)

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 4, cv2.LINE_AA)

        if className == "okay":
            print(f"{attention_flag=}")
            attention_flag = True
            start_attention = datetime.datetime.now()

        if attention_flag:
            now = datetime.datetime.now()

            # check if we wait for a command longer than 2 seconds
            # TODO this doesnt work yet, set check outside of if
            if (now - start_attention).total_seconds() < 2:

                # get the current level so we can add or subtract from it
                current_volume_level = volume.GetMasterVolumeLevel()
                closest_value = min(volume_levels, key=lambda x:abs(x-current_volume_level))
                ci = volume_levels.index(closest_value)

                # increase volume if thumbs up
                if className == "thumbs up":
                    volume.SetMasterVolumeLevel(min(volume_levels[min(ci+1, 10)], volume_range[1]), None)
                    print("Up",current_volume_level, "->",volume.GetMasterVolumeLevel())
                    attention_flag = False

                # decrease volume if thumbs down
                elif className == "thumbs down":
                    volume.SetMasterVolumeLevel(max(volume_levels[max(0,ci-1)], volume_range[0]), None)
                    print("down",current_volume_level, "->",volume.GetMasterVolumeLevel())
                    attention_flag = False

                # pass
                elif className == "peace":
                    pyautogui.press("playpause")
                    print("pause")
                    attention_flag = False

            # Reset flags if we wait longer than 2 seconds
            else:
                attention_flag = False
                start_attention = None
                now = None

    # print flag
    cv2.putText(frame, "att " + str(attention_flag), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)
    
    # Show the final output
    cv2.imshow("Output", frame)
             
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

