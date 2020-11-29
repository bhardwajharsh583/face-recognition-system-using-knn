import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data=[]
dataset_path ='./data/'

while True:
	ret,frame = cap.read()

	if ret == False:
		continue
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#sorting faces in one frame

	faces = face_cascade.detectMultiScale(frame,1.3,5) # it is a list of touple of faces
	#print(faces)

	faces = sorted(faces,key = lambda f:f[2]*f[3])

	# picks the last face after sorting(largest face in frame) !!

	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# extract region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] # by convention frame has axis as y,x
		face_section = cv2.resize(face_section,(100,100))

		#Store 10th data 
		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
		cv2.imshow("Face section",face_section)
	cv2.imshow("Frame",frame)
	
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# convert our face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# save into file system
file_name = input("Enter the person name :")
np.save(dataset_path+file_name+'.npy',face_data)

cap.release()
cv2.destroyAllWindows()