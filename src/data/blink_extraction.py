import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import os

EAR_LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
EAR_RIGHT_EYE_IDXS = [33,  160, 158, 133, 153, 144]

def image_to_array(image_file_path, showImagePlot=False): 
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
    image = np.ascontiguousarray(image)
    
    if showImagePlot: 
        plt.imshow(image)
    
    return image

def extract_face_mesh(image): 
    mp_facemesh = mp.solutions.face_mesh
    
    with mp_facemesh.FaceMesh(
    static_image_mode=True,         # Default=False
    max_num_faces=1,                # Default=1
    refine_landmarks=False,         # Default=False
    min_detection_confidence=0.5,   # Default=0.5
    min_tracking_confidence= 0.5,   # Default=0.5
) as face_mesh:
        return face_mesh.process(image)

def get_pixel_coords(results, image, eyeType):
    imgH, imgW, _ = image.shape         
    
    chosen_landmark_idxs = []
    if eyeType == "left":
        chosen_landmark_idxs = EAR_LEFT_EYE_IDXS
    if eyeType == "right":
        chosen_landmark_idxs = EAR_RIGHT_EYE_IDXS

    mp_drawing  = mp.solutions.drawing_utils
    denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
    
    # if no face was detected 
    if results.multi_face_landmarks is None: 
        return {}
    else: 
        pred_coords = {}
        for landmark_idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            # For plotting chosen eye landmarks
            if landmark_idx in chosen_landmark_idxs:
                pred_coord = denormalize_coordinates(landmark.x, 
                                                    landmark.y, 
                                                    imgW, imgH)
                pred_coords[landmark_idx] = pred_coord
        return pred_coords

def l2_norm(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def calculate_ear(coords_points, eyeType):
    if len(coords_points) == 0: #no face detected
        return -1.0 
    
    eye_landmark_idxs = []
    if eyeType == "left":
        eye_landmark_idxs = EAR_LEFT_EYE_IDXS
    if eyeType == "right":
        eye_landmark_idxs = EAR_RIGHT_EYE_IDXS

    P2_P6 = l2_norm(coords_points[eye_landmark_idxs[1]], coords_points[eye_landmark_idxs[5]])
    P3_P5 = l2_norm(coords_points[eye_landmark_idxs[2]], coords_points[eye_landmark_idxs[4]])
    P1_P4 = l2_norm(coords_points[eye_landmark_idxs[0]], coords_points[eye_landmark_idxs[3]])
 
    # Compute the eye aspect ratio
    ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    
    return ear


def extract_blink(img_file): 
    blinks_by_frame = []
    image = image_to_array(img_file)
    mp_extraction = extract_face_mesh(image)
    left_eye_pixel_coords = get_pixel_coords(mp_extraction, image, "left")
    right_eye_pixel_coords = get_pixel_coords(mp_extraction, image, "right")
    left_ear = calculate_ear(left_eye_pixel_coords, "left")
    right_ear = calculate_ear(right_eye_pixel_coords, "right")
    avg_ear = (left_ear+right_ear)/2
    
    if avg_ear == -1: # face was not detected in the frame
        return 0.5
    if avg_ear > 0.20: # there was no blink 
        return 0.
    if avg_ear <= 0.20: # indicates a blink
        return 1.
    
