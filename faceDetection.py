
# import cv2
# import numpy as np
# from os import listdir
# from os.path import isfile, join

# # ----------- Step 1: Face Detection and Sample Collection -----------

# def collect_samples(data_path):
#     # face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#     if face_classifier.empty():
#         print("Haar Cascade not loaded. Check the path.")
#         return

#     def face_extractor(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#         if len(faces) == 0:
#             return None
#         for (x, y, w, h) in faces:
#             return img[y:y+h, x:x+w]

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Camera not opened. Check the connection.")
#         return

#     count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image")
#             break

#         face = face_extractor(frame)
#         if face is not None:
#             count += 1
#             face = cv2.resize(face, (200, 200))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             file_name_path = f'{data_path}/{count}.jpg'
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow('Face Cropper', face)
#         else:
#             print("Face not found")

#         if cv2.waitKey(1) == 13 or count == 100:  # Press Enter or collect 100 samples
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print('Sample collection completed')


# # ----------- Step 2: Train the Model -----------

# def train_model(data_path):
#     onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

#     Training_Data, Labels = [], []

#     for i, file in enumerate(onlyfiles):
#         image_path = join(data_path, file)
#         images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if images is not None:
#             Training_Data.append(np.asarray(images, dtype=np.uint8))
#             Labels.append(i)
#         else:
#             print(f"Warning: Unable to read image {image_path}")

#     Labels = np.asarray(Labels, dtype=np.int32)

#     model = cv2.face.LBPHFaceRecognizer_create()
#     model.train(np.asarray(Training_Data), Labels)
#     print("Dataset model training completed")
#     return model


# # ----------- Step 3: Recognize Face -----------

# def recognize_faces(model):
#     # face_classifier = cv2.CascadeClassifier('C:/Users/adity/Downloads/haarcascade_frontalface_default.xml')
#     face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#     def face_detector(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#         if len(faces) == 0:
#             return img, None
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             roi = img[y:y + h, x:x + w]
#             roi = cv2.resize(roi, (200, 200))
#             return img, roi

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Camera not opened.")
#         return

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture image")
#                 break

#             frame = cv2.resize(frame, (640, 480))
#             image, face = face_detector(frame)

#             if face is not None:
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                 result = model.predict(face)
#                 confidence = int(100 * (1 - (result[1] / 300)))

#                 if confidence > 82:
#                     cv2.putText(image, "Recognized", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#                 else:
#                     cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#             cv2.imshow('Face Recognition', image)

#             if cv2.waitKey(1) == 13:  # Enter key
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     collect_samples("faces")











import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# ----------- Step 1: Face Detection and Sample Collection -----------

def collect_samples(data_path):
    # Load Haar cascade
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_classifier.empty():
        print("Haar Cascade not loaded. Check the path.")
        return

    # Create folder if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        print("Faces detected:", len(faces))  # Debug info

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return img[y:y+h, x:x+w]

        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened. Check the connection.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Live Camera", frame)  # Debug window

        face = face_extractor(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f'{data_path}/{count}.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q') or key == 27 or count == 100:  # Enter, 'q', ESC, or 100 samples
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Sample collection completed')


# ----------- Step 2: Train the Model -----------

def train_model(data_path):
    """
    Train the model with images organized in folders (one folder per person)
    or with files named as 'PersonName_number.jpg'
    Returns: (model, label_to_name_dict)
    """
    Training_Data, Labels = [], []
    label_to_name = {}
    current_label = 0
    
    # Check if data_path contains folders (person-based organization)
    subdirs = [d for d in listdir(data_path) if os.path.isdir(join(data_path, d))]
    
    if subdirs:
        # Folder-based organization: each folder is a person
        print("Detected folder-based organization")
        for person_name in sorted(subdirs):
            person_path = join(data_path, person_name)
            files = [f for f in listdir(person_path) if isfile(join(person_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in files:
                image_path = join(person_path, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    Training_Data.append(np.asarray(image, dtype=np.uint8))
                    Labels.append(current_label)
                    label_to_name[current_label] = person_name
                else:
                    print(f"Warning: Unable to read image {image_path}")
            
            if files:  # Only increment if we found images for this person
                current_label += 1
    else:
        # File-based organization: files named as 'PersonName_number.jpg' or just numbered
        print("Detected file-based organization")
        files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Try to extract person names from filenames
        person_names = {}
        for file in sorted(files):
            # Try to extract name from filename (e.g., "John_1.jpg" -> "John")
            name = file.split('_')[0] if '_' in file else file.split('.')[0]
            # Remove numbers from name if it's just a number
            if name.isdigit():
                name = f"Person_{name}"
            
            if name not in person_names:
                person_names[name] = current_label
                label_to_name[current_label] = name
                current_label += 1
            
            image_path = join(data_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                Training_Data.append(np.asarray(image, dtype=np.uint8))
                Labels.append(person_names[name])
            else:
                print(f"Warning: Unable to read image {image_path}")
    
    if len(Training_Data) == 0:
        print("No training images found!")
        return None, None
    
    Labels = np.asarray(Labels, dtype=np.int32)
    
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), Labels)
    print(f"Dataset model training completed with {len(label_to_name)} person(s)")
    print(f"Labels: {label_to_name}")
    return model, label_to_name


# ----------- Step 3: Recognize Face -----------

def recognize_faces(model, label_to_name):
    """
    Recognize faces and display the person's name/label
    """
    if model is None or label_to_name is None:
        print("Model not trained. Please train the model first.")
        return
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def face_detector(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            return img, None, None
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
            return img, roi, (x, y, w, h)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            frame = cv2.resize(frame, (640, 480))
            image, face, face_coords = face_detector(frame)

            if face is not None and face_coords is not None:
                x, y, w, h = face_coords
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)
                label_id = result[0]
                confidence = int(100 * (1 - (result[1] / 300)))

                if confidence > 82 and label_id in label_to_name:
                    person_name = label_to_name[label_id]
                    # Display name and confidence
                    text = f"{person_name} ({confidence}%)"
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == ord('q') or key == 27:  # Enter, 'q', or ESC key
                break
          

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------- MAIN EXECUTION FLOW -----------

if __name__ == "__main__":
    # Option 1: Collect samples from camera (uncomment to use)
    # collect_samples("faces")
    
    # Option 2: Train model from uploaded photos
    # Organize photos in folders: faces/Person1/, faces/Person2/, etc.
    # OR name files as: PersonName_1.jpg, PersonName_2.jpg, etc.
    model, label_to_name = train_model("faces")
    
    if model is not None:
        # Step 3: Recognize faces and display labels
        recognize_faces(model, label_to_name)
    else:
        print("Training failed. Please check your image files.")

