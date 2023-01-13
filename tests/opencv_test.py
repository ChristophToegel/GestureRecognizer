import cv2

def open_cam():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        x,y,c = frame.shape
        print(x,y,c)
        frame = cv2.flip(frame, 1)



        # Show webcam
        cv2.imshow("Output", frame)

        # destroy frame on 'q' press
        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_cam()
