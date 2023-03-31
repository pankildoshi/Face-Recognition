from flask import Flask, request, redirect, url_for, render_template, Response
import os
import zipfile
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = './upload/'
PREDICTION_FOLDER = './predictions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

def extract_zipfile(filepath, extractpath):
    """
    Extracts a ZIP file at the specified filepath to the specified extractpath.

    Args:
        filepath (str): Path to the ZIP file to extract.
        extractpath (str): Path to the directory where the contents of the ZIP file should be extracted.
    """
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extractpath)

@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    if request.method == 'POST':
        file = request.files['zipfile']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            extractpath = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
            extract_zipfile(filepath, extractpath)
    print('zip uploaded successfully')
    return render_template('index.html', uploaded="Dataset uploaded successfully")
        
def generate_embeddings():
    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("face_detection_model/nn4.small2.v1.t7")

    # grab the paths to the input images in our dataset
    print("Quantifying Faces...")
    imagePaths = list(paths.list_images("upload/extracted/dataset"))

    # initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        if (i%50 == 0):
            print("Processing image {}/{}".format(i, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    
    # Return a message indicating success
    print('embeddings extracted successfully')

@app.route('/train_model', methods=['POST'])
def train_model():
    # generate the face embeddings
    generate_embeddings()
    
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("output/recognizer", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()
    return render_template('index.html', trained="Model has been trained successfully")

def make_prediction(image):
    # load our serialized face detector from disk
    print("Loading Face Detector...")
    protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
    modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch('face_detection_model/nn4.small2.v1.t7')

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open('output/recognizer', "rb").read())
    le = pickle.loads(open('output/le.pickle', "rb").read())

    # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(image)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    retval, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        if image:
            filename = image.filename
            filepath = os.path.join(app.config['PREDICTION_FOLDER'], filename)
            image.save(os.path.join(app.config['PREDICTION_FOLDER'], filename))

        # Make prediction on the uploaded image
        img_str = make_prediction(filepath)

    # Render the HTML template for uploading an image
    return render_template('index.html', image=img_str)

def generate_frames():
    cap = cv2.VideoCapture(0)

    # load our serialized face detector from disk
    print("Loading Face Detector...")
    protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
    modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch('face_detection_model/nn4.small2.v1.t7')

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open('output/recognizer', "rb").read())
    le = pickle.loads(open('output/le.pickle', "rb").read())

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert the image to base64 encoding
        jpeg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
