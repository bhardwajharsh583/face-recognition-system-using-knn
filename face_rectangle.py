import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	ret,frame = cap.read()  
	gray_fr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(gray_fr,1.3,5)

	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)

	cv2.imshow("faces detected",frame)

    # USER ENTERS A KEY TO END READING
	key_pressed = cv2.waitKey(1) & 0xFF  
	if key_pressed== ord('q'):
		break
    


cap.release()
cv2.destroyAllWindows()

