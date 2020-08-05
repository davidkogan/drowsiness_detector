from scipy.spatial import distance
import numpy as np
import dlib
import cv2
from threading import Thread
import playsound

def get_faces(img, detector):
	faces = detector(img, 1)
	return faces

def get_landmarks(img, face, predictor):
	landmarks = predictor(img, face)
	landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
	return landmarks

def get_eyes(img, landmarks):

	return [landmarks[36:42], landmarks[42:48]]

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])

	ratio = (A + B) / (2.0 * C)

	return ratio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
ALARM_PATH = 'alarm.wav'

def sound_alarm(path):
	playsound.playsound(path)

def main():

	counter = 0

	cap = cv2.VideoCapture(0)
	cv2.namedWindow('Image')
	while True:

		_, frame = cap.read()
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = get_faces(gray_frame, detector)

		for face in faces:
			landmarks = get_landmarks(gray_frame, face, predictor)
			eyes = get_eyes(gray_frame, landmarks)
			left_eye, right_eye = eyes
			ratio = sum([eye_aspect_ratio(eye) for eye in eyes]) / 2.0

			if ratio < EYE_AR_THRESH:
				counter += 1
				if counter >= EYE_AR_CONSEC_FRAMES:
					t = Thread(target = sound_alarm, args = (ALARM_PATH,))
					t.daemon = True
					t.start()
			else:
				counter = 0

		cv2.imshow('Image', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()