#!/usr/bin/env python3
import numpy
import pybullet as p
import time
import pybullet_data
import os
import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

def pybulletsim():
	
	global x
	global y
	global z
	global wristpitch
	global wristroll
	global ee
	global xvel
	global yvel

	x=360
	y=300
	z=0
	wristpitch = 0
	wristroll = 0
	ee = 1
	xvel = 0
	yvel = 0 #right

	urdf_path = os.path.dirname(os.path.realpath(__file__))+"/husky.urdf"

	print("Loading urdf from " + urdf_path)

	p.connect(p.GUI)
	p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

	plane = p.loadURDF("plane.urdf")
	#box = p.loadURDF("box.urdf")
	p.setGravity(0,0,-9.8)
	#p.setTimeStep(1./500)
	p.setTimeStep(1./500)
	#p.setDefaultContactERP(0)
	#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
	urdfFlags = p.URDF_USE_SELF_COLLISION
	quadruped = p.loadURDF(urdf_path, [0,0,.6], [0,0,0,1], flags=urdfFlags, useFixedBase=False)

	a = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
	#enable collision between lower legs

	for j in range (p.getNumJoints(quadruped)):
			print(p.getJointInfo(quadruped,j))
	Position, Ori = p.getBasePositionAndOrientation(quadruped)
	print (Position)

	#2,5,8 and 11 are the lower legs
	lower_legs = [2,5,8,11]
	hip = [0,3,6,9]
	thigh = [1,4,7,10]
	calf = [2,5,8,11]
	arm = [13,14,15,16,17]
	end_effector = [22,23]
	fixed = [12,18,19,20,21,24]

	arms = [0,0,0,0,0,0,0,0]
	eelinks = [0,0,0]

	for i in range(15,23):
		enableCollision = 0
		p.setCollisionFilterPair(quadruped, quadruped, 16,i, enableCollision)
		p.setCollisionFilterPair(quadruped, quadruped, 17,i, enableCollision)
		p.setCollisionFilterPair(quadruped, quadruped, 20,i, enableCollision)
		p.setCollisionFilterPair(quadruped, quadruped, 21,i, enableCollision)

	jointIds=[]
	paramIds=[]	
	jointOffsets=[]
	jointDirections=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

	for i in range (12):
		jointOffsets.append(0)

	maxForceId = p.addUserDebugParameter("maxForce",0,100,100)

	leftwheels = [2,4]
	rightwheels = [3,5]
	arm = [11,12,13]
	wrist = [14,15]
	end_effector = [20,21]

	arms = [0,0,0,0,0,0,0,0]
	eelinks = [0,0,0]

	p.setRealTimeSimulation(0)

	joints=[]


	p.setRealTimeSimulation(1)


	while not rospy.is_shutdown():	

		maxForce = p.readUserDebugParameter(maxForceId)


	############ARM

		h = 126.75
		l01 = 300
		l02 = 60
		l1 = 305.941170816
		l1theta = 11.309932474
		l2 = 300
		l3 = 70
		

		a1 = 0
		a2 = 0
		a3 = 0

		if(x**2 + y**2 + z**2 <= 600**2 and not (x==0 and y==0 and z==0)):

			if(x==0):
				x = 0.00001
			#base motor
			try:	
				a1 = np.arctan(z/x)
			except:
				pass
			
			if(x>0 and z>0): #top right
				a1 = a1
			if(x<0 and z>0): #bottom left
				a1 = np.radians(90)+np.radians(90)+a1
			if(x>0 and z<0): #top left
				a1 = a1
			if(x<0 and z<0): #bottom right
				a1 = np.radians(90)+np.radians(90)-a1
				a1 = -a1

			if(x<0 and z==0):
				a1 = np.radians(180)

			if abs(z)>abs(x):
				x = z
			if(x<0):
				y = -y

			#second motor and first motor
			try:
				a3 = np.arccos((x**2+y**2-l1**2-l2**2)/(2*l1*l2))
				a2 = -np.arctan(y/x)-np.arctan((l2*np.sin(a3))/(l1+(l2*np.cos(a3))))
				a3 = -a3
				a2 = a2 + 1.37
				a3 = a3 + 1.37
			except:
				pass
			angles = [a1,a2,a3]

			i=0
			for a in arm:
				p.setJointMotorControl2(quadruped,a,controlMode=p.POSITION_CONTROL,targetPosition=angles[i],force = maxForce)

				i += 1

		else:
			pass
		


	############WHEELS

		xvelms = xvel/0.33

		left = xvelms
		right = xvelms

		if yvel > 0:
			left = xvelms-abs(yvel)/10
			right = xvelms+abs(yvel)/10

		if yvel < 0:
			left = xvelms+abs(yvel)/10
			right = xvelms-abs(yvel)/10


		for w in leftwheels:
				p.setJointMotorControl2(quadruped,w,controlMode=p.VELOCITY_CONTROL,targetVelocity=left,force = maxForce)

		for w in rightwheels:
				p.setJointMotorControl2(quadruped,w,controlMode=p.VELOCITY_CONTROL,targetVelocity=right,force = maxForce)

		
	############ENDEFFECTOR
		print(ee)
		ee1 = 0.021 + ((ee)*(0.057-0.021)) #0 close 1 open

		p.setJointMotorControl2(quadruped,20,controlMode=p.POSITION_CONTROL,targetPosition=ee1,force = maxForce)
		p.setJointMotorControl2(quadruped,21,controlMode=p.POSITION_CONTROL,targetPosition=-ee1,force = maxForce)


	############WRIST

		wristpitch1 = wristpitch #-2.27 down 1.87 up #0 default
		wristroll1 = -3.14 + ((wristroll+1)*(3.14+3.14)/2) #-1 ccw 1 cw 0 default

		p.setJointMotorControl2(quadruped,14,controlMode=p.POSITION_CONTROL,targetPosition=wristpitch1,force = maxForce)
		p.setJointMotorControl2(quadruped,15,controlMode=p.POSITION_CONTROL,targetPosition=wristroll1,force = maxForce)

	############END

		p.stepSimulation()

	p.disconnect()

def callback(data):

	global x
	global y
	global z
	global wristpitch
	global wristroll
	global ee
	global xvel
	global yvel

	print(data.data)

	x = data.data[0]
	y = data.data[1]
	z = data.data[2]
	wristpitch = data.data[3]
	wristroll = data.data[4]
	ee = data.data[5]
	xvel = data.data[6]
	yvel = data.data[7]
    
def listener():
	rospy.init_node('robotdatanode')
	rospy.Subscriber("robotdata", Float64MultiArray, callback)
	pybulletsim()
	rospy.spin()

listener()

