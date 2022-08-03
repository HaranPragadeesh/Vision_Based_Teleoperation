import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import rospy
import time
from tensorflow.keras.models import load_model
from std_msgs.msg import Float64MultiArray


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data = Float64MultiArray()

pub = rospy.Publisher('newdata', Float64MultiArray, queue_size=10)
rospy.init_node('publisher')
time.sleep(0.1) 


def calculate_angle(a,b,c):
	a = np.array(a) # First
	b = np.array(b) # Mid
	c = np.array(c) # End

	radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
	angle = np.abs(radians*180.0/np.pi)

	if angle >180.0:
		angle = 360-angle

	return angle 

a=360
b=300
c=0
wristpitch = 0
wristroll = 0
ee = 1	
	
draw = mp.solutions.drawing_utils
pose = mp.solutions.pose
holistic = mp.solutions.holistic
hol = holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands = mp.solutions.hands

model = load_model('mp_hand_gesture')

classNames= ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

cap = cv2.VideoCapture(3)

while not rospy.is_shutdown() and cap.isOpened():
	

	ret, frame = cap.read()
	x , y, c = frame.shape


	frame = cv2.flip(frame, 1)

	framecol = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	result = hol.process(framecol)
		
	landmark_pose = result.pose_landmarks.landmark	
	
	gestureL = ''
	gestureR = ''

	# post process the result
	if result.right_hand_landmarks:
		landmarks = []

		for id, ld in enumerate(result.right_hand_landmarks.landmark):
			# print(id, lm)
			ldx = int(ld.x * x)
			ldy = int(ld.y * y)

			landmarks.append([ldx, ldy])

		# Drawing landmarks on frames
		draw.draw_landmarks(frame, result.right_hand_landmarks,holistic.HAND_CONNECTIONS)
		# Predict gesture
		prediction = model.predict([landmarks])
		# print(prediction)
		classID = np.argmax(prediction)
		gestureL = classNames[classID]
	

	# post process the result
	if result.left_hand_landmarks:
		landmarks = []
		lmlist = []
		for id, lm in enumerate(result.left_hand_landmarks.landmark):
			# print(id, lm)
			lmx = int(lm.x * x)
			lmy = int(lm.y * y)

			landmarks.append([lmx, lmy])
			
			#lmlist[id].append(lmx,lmy)
	
			"""
			x1, y1 = landmarks[5]
			x2, y2 = landmarks[17]
			distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
			A, B, C = coff
			distanceCM = A * distance ** 2 + B * distance + C
	 		
			print(distanceCM, distance)
			
			"""
		#print(landmarks2)
		# Drawing landmarks on frames
		
		draw.draw_landmarks(frame, result.left_hand_landmarks,holistic.HAND_CONNECTIONS)
		# Predict gesture
		prediction = model.predict([landmarks])
		# print(prediction)
		classID = np.argmax(prediction)
		gestureR = classNames[classID]
	

	# Get coordinates for left shoulder, right shoulder, elbow and wrist
	L_Elbow = [landmark_pose[pose.PoseLandmark.RIGHT_ELBOW.value].x,landmark_pose[pose.PoseLandmark.RIGHT_ELBOW.value].y]
	L_Shoulder = [landmark_pose[pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmark_pose[pose.PoseLandmark.RIGHT_SHOULDER.value].y]
	R_Shoulder = [landmark_pose[pose.PoseLandmark.LEFT_SHOULDER.value].x,landmark_pose[pose.PoseLandmark.LEFT_SHOULDER.value].y]
	R_Elbow = [landmark_pose[pose.PoseLandmark.LEFT_ELBOW.value].x,landmark_pose[pose.PoseLandmark.LEFT_ELBOW.value].y]
	R_Wrist = [landmark_pose[pose.PoseLandmark.LEFT_WRIST.value].x,landmark_pose[pose.PoseLandmark.LEFT_WRIST.value].y]
	R_Index =  [landmark_pose[pose.PoseLandmark.LEFT_INDEX.value].x,landmark_pose[pose.PoseLandmark.LEFT_INDEX.value].y]
	


	# Calculate angle of the shoulder
	angle = calculate_angle(L_Elbow, L_Shoulder, R_Shoulder)
	angleS = calculate_angle(L_Shoulder, R_Shoulder, R_Elbow)
	angleE = calculate_angle(R_Shoulder, R_Elbow, R_Wrist)
	angleW = calculate_angle(R_Elbow, R_Elbow , R_Index)
	
	"""
	cv2.putText(frame, str(angleS), tuple(np.multiply(R_Shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, str(angleE), tuple(np.multiply(R_Elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, str(angleW), tuple(np.multiply(R_Wrist, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	"""
	#print(landmark_pose[pose.PoseLandmark.LEFT_WRIST])



	if gestureL == 'live long':
		motion = 'Forward'
		m = 1
		xvel = 0.5
		yvel = 0.5
	elif gestureL == 'fist':
		motion = 'Reverse'
		m = 2
		xvel = -0.5
		yvel = -0.5
	elif gestureL == 'thumbs up':
		stage = 'Spin Left'
		m = 5
		xvel = -0.5
		yvel = 0.5
		
	elif gestureL == 'thumbs down':
		stage = 'Spin Right'
		m = 6
		xvel = 0.5
		yvel = -0.5
	else:
		
		if 115 < angle < 160 :
			motion = "Left"
			m = 3
			xvel = -0.5
			yvel = 0.2
		
		elif 20 < angle < 90:
			motion = "Right"
			m = 4
			xvel = 0.2
			yvel = -0.5
		
		else:	
			motion = 'Stop'
			m = 5
			xvel = 0
			yvel = 0	
	
	
		
				
	if gestureR == 'fist' :
		arm = 'Close'
		ee = 1
	else:
		arm = 'Open'
		ee = 0
	
	if gestureR == 'thumbs up' :
		arm = "X axis"
		
		a = (angleS-30)*5
		
	
	elif gestureR == 'rock' :
		arm = "Y axis"
		
		b = (angleS-30)*5
		
	elif gestureR == 'okay':
		arm = 'Z axis'
		
		if (angleS<80):
			
			c = (angleS-120)*5
			
		else:
		
			c = (angleS-70)*5
	
	
	elif gestureR == 'peace':
	
		a=360
		b=300
		c=0
		wristpitch = 0
		wristroll = 0
		ee = 1
		arm = ""
	
	
	

	print(a)
	
	data.data = [a,b,c,wristpitch,wristroll,ee,xvel,yvel]

	pub.publish(data)
	
	# show the motion commands
	cv2.putText(frame, motion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


	# show the arm commands
	cv2.putText(frame, arm, (530, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

	
	

	draw.draw_landmarks(frame, result.right_hand_landmarks,holistic.HAND_CONNECTIONS)
	draw.draw_landmarks(frame, result.pose_landmarks,holistic.POSE_CONNECTIONS)
	draw.draw_landmarks(frame, result.left_hand_landmarks,holistic.HAND_CONNECTIONS)

	cv2.imshow("Output", frame)
	if cv2.waitKey(1) == ord('q'):
		break


# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
