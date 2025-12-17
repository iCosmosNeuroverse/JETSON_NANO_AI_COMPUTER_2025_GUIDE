import cv2

url = "http://192.168.100.192:8080/video"  

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        continue

    cv2.imshow("IP Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

