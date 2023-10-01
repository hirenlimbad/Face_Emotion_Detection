
import cv2
import numpy as np
from tensorflow import keras

class EmotionDetector:

    def __init__(self):
        self.model = keras.models.load_model("ML_end_to_end/artifacts/saved_model")
        self.EMOTION = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def getEmotion(self, frame):
        face_frame = self.detect_faces(frame)
        if len(face_frame) > 0:
            x, y, w, h = face_frame[0]  # Coordinates and dimensions of the first detected face

            # Crop the image based on the detected face
            cropped = frame[y:y+h, x:x+w]

        emotion = self.detect_emotion(frame)
        return emotion

    def allowed_file(self, filename):
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    def detect_faces(self,frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.17, minNeighbors=6, minSize=(30, 50))
        return faces
    

    def detect_emotion(self,face_frame):
        # # Resize the face frame to 48x48 and convert to grayscale
        face_frame = cv2.resize(face_frame, (48, 48))
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        face_frame = face_frame.reshape(-1, 48, 48, 1)
        face_frame = face_frame / 255.0


        # Use the loaded model to make predictions
        predictions = self.model.predict(face_frame)
        emotion_index = np.argmax(predictions)
        emotion = self.EMOTION[emotion_index]
        return emotion


    def generate_frames(self):
        camera = cv2.VideoCapture(0)

        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                faces = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_frame = frame[y:y + h, x:x + w]
                    emotion = self.detect_emotion(face_frame)
                    cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def videoHandler(self, video = r"videos/temp.mp4"):
        try:
            vid = cv2.VideoCapture(r"ML_end_to_end/src/components/videos/temp.mp4")
            while True:
                success, frame = vid.read()
                if not success:
                    break
                else:
                    faces = self.detect_faces(frame)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        face_frame = frame[y:y + h, x:x + w]
                        emotion = self.detect_emotion(face_frame)
                        cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error processing {video.filename}: {e}")

                
    def masterImageHandler(self, image):
        try:
            if image.filename != '':
                # Load the image and detect faces
                img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
                faces = self.detect_faces(img)

                # Draw emotion borders around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_frame = img[y:y + h, x:x + w]
                    emotion = self.detect_emotion(face_frame)
                    cv2.putText(img, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Encode the image and return it
                ret, buffer = cv2.imencode('.jpg', img)
                img_bytes = buffer.tobytes()
                return img_bytes

            else:
                return "Invalid file type." + image.filename

        except Exception as e:
            print(f"Error processing {image.filename}: {e}")