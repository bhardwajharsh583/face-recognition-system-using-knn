import cv2
import numpy as np 
import os


################## KNN CODE #####################
def distance(v1,v2):
	# eucladian dist
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	dist = []
	for i in range(train.shape[0]):
		#get vector and label
		ix = train[i,:-1]
		iy = train[i,-1]
		# compute the dist from test pt
		d = distance(test,ix)
		dist.append([d,iy])

	#sort based on dist and get top k
	dk = sorted(dist, key=lambda f:f[0])[:k]

	#retrieve only the labels
	labels = np.array(dk)[:,-1]

	#get frequencies of each other
	output = np.unique(labels ,return_counts = True)
	#Find max freq and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

################################################

#Init Camera
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data=[]
dataset_path ='./data/'
class_id = 0 # labels for given file
labels = []
names = {}

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4] # take name upto . (in .npy)
		#print('Loaded '+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# create labels for class
		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))

# print(face_dataset.shape)
# print(face_labels.shape)

traindataset = np.concatenate((face_dataset,face_labels),axis = 1) # last col for featues and see what for axis?
print(traindataset.shape)

# testing

while True:
	ret,frame=cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		#get the face roi
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Predicted label
		out = knn(traindataset,face_section.flatten())

		# display name and rect
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()






