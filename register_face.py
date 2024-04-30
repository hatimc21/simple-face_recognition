import cv2
import os

def register_face():
    person_name = input("Enter the person's name: ")
    folder_path = 'authorized_faces'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            face_image_path = os.path.join(folder_path, f'{person_name}.jpg')
            cv2.imwrite(face_image_path, frame)
            print(f"Face registered for {person_name}!")
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_face()
