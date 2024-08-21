import cv2
import os


# Function to read all images from the folder
def load_images_from_folder(folder_path):
    images = []
    image_names = []
    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            print(f"Skipping {image_file}: Unable to load.")
                            continue
                        images.append(image)
                        image_names.append(person_folder)
                        image = None  # Release memory
                    except cv2.error as e:
                        print(f"Error loading image {image_file}: {e}")
                        continue
    return images, image_names


# Specify the root folder path where all images are stored
root_folder = 'C:\\Users\\Admin\\Downloads\\lfw-deepfunneled\\lfw-deepfunneled'
# root_folder = os.path.join(os.getcwd(), 'images')
# Load all images from the folder
images, image_names = load_images_from_folder(root_folder)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Assume the first face detected is the person
        person_name = "Unknown"
        for i, image in enumerate(images):
            # Compare the detected face with the images in the folder
            # This is a simple comparison, you may want to use a more robust method
            face_gray = gray[y:y + h, x:x + w]  # Extract the face region from the grayscale frame
            if cv2.matchTemplate(face_gray, image, cv2.TM_CCOEFF_NORMED)[0][0] > 0.5:
                person_name = image_names[i]
                break
        cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()



# import cv2
#
# # Open the default camera (index 0)
# cap = cv2.VideoCapture(0)
#
# # Check if the camera is opened
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Cannot receive frame")
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Frame', frame)
#
#     # Press 'q' to quit
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import os
#
# # Function to read all images from the folder
# def load_images_from_folder(folder_path):
#     images = []
#     image_names = []
#     for person_folder in os.listdir(folder_path):
#         person_path = os.path.join(folder_path, person_folder)
#         if os.path.isdir(person_path):
#             for image_file in os.listdir(person_path):
#                 image_path = os.path.join(person_path, image_file)
#                 if image_path.endswith('.jpg') or image_path.endswith('.png'):
#                     try:
#                         image = cv2.imread(image_path)
#                         if image is None:
#                             print(f"Skipping {image_file}: Unable to load.")
#                             continue
#                         # Optionally resize the image if it's too large
#                         # image = cv2.resize(image, (500, 500))  # Example resize to 500x500
#                         images.append(image)
#                         image_names.append((image_file, person_folder))  # Store folder name with image name
#                     except cv2.error as e:
#                         print(f"Error loading image {image_file}: {e}")
#                         continue
#     return images, image_names
#
# # Specify the root folder path where all images are stored
# root_folder = 'C:\\Users\\Admin\\Downloads\\lfw-deepfunneled\\lfw-deepfunneled'
#
# # Load all images from the folder
# images, image_names = load_images_from_folder(root_folder)
#
# # Specify the index of the image you want to display
# image_index_to_display = 1000  # Change this to the desired index
#
# # Display the image if the index is within range
# if 0 <= image_index_to_display < len(images):
#     print(f"Displaying image: {image_names[image_index_to_display][0]}")
#     image = images[image_index_to_display].copy()  # Create a copy to draw on
#     folder_name = image_names[image_index_to_display][1]
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(image, folder_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print(f"Index {image_index_to_display} is out of range. There are {len(images)} images available.")

# # import cv2
# # import os
# # import face_recognition
# #
# # # Load the face cascade
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #
# # # Create a dictionary to store face encodings and folder names
# # face_encodings_dict = {}
# #
# # # Populate the dictionary
# # root_folder = 'C:\\Users\\Admin\\Downloads\\lfw-deepfunneled\\lfw-deepfunneled'
# # for person_folder in os.listdir(root_folder):
# #     person_path = os.path.join(root_folder, person_folder)
# #     if os.path.isdir(person_path):
# #         face_encodings_list = []
# #         for image_file in os.listdir(person_path):
# #             image_path = os.path.join(person_path, image_file)
# #             if image_path.endswith('.jpg') or image_path.endswith('.png'):
# #                 image = face_recognition.load_image_file(image_path)
# #                 encodings = face_recognition.face_encodings(image)
# #                 if encodings:
# #                     face_encodings_list.append(encodings[0])
# #         face_encodings_dict[person_folder] = face_encodings_list
# #
# # # Open the default camera
# # cap = cv2.VideoCapture(0)
# #
# # while True:
# #     # Capture frame-by-frame
# #     ret, frame = cap.read()
# #
# #     # Convert to grayscale
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     # Detect faces
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# #
# #     # Draw rectangles around detected faces with labels
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #
# #         # Get the face encoding of the detected face
# #         face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
# #
# #         # Find the folder name from the dictionary
# #         folder_name = "Unknown"
# #         for person_folder, encodings_list in face_encodings_dict.items():
# #             for encodings in encodings_list:
# #                 if face_recognition.compare_faces([encodings], face_encoding)[0]:
# #                     folder_name = person_folder
# #                     break
# #
# #         # Display the folder name outside the rectangle
# #         cv2.putText(frame, folder_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# #
# #     # Display the resulting frame
# #     cv2.imshow('Face Detection', frame)
# #
# #     # Press 'q' to quit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # # Release the camera
# # cap.release()
# # cv2.destroyAllWindows()