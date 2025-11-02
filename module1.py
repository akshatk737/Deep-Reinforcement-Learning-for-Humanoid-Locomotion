import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read the image from {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized / 255.0
    return np.asarray(image_normalized, dtype=np.float32)

# Load images
images_dir = Path(r"./images")
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

if not image_files:
    print(f"No image files found in {images_dir}")
else:
    image_arrays = {}
    for img_path in image_files:
        try:
            image_arrays[img_path.stem] = load_and_preprocess_image(img_path)
            print(f"Loaded: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    for name, arr in image_arrays.items():
        print(f"{name}: shape = {arr.shape}")

    print(f"\nTotal images loaded: {len(image_arrays)}")

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Mapping and connections 
BODY_25_MAPPING = [
    0, 12, 12, 14, 16, 11, 13, 15,
    24, 24, 26, 28, 23, 25, 27,
    5, 2, 8, 7, 31, 31, 29, 32, 32, 30
]

BODY_25_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8),
    (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24)
]

def mediapipe_to_openpose_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    skeleton = np.zeros((25, 3), dtype=np.float32)
    for i, mp_idx in enumerate(BODY_25_MAPPING):
        if mp_idx == 0 and i > 0:
            skeleton[i] = [0, 0, 0]
        else:
            lm = landmarks[mp_idx]
            skeleton[i] = [lm.x * w, lm.y * h, lm.visibility]
    return skeleton

def select_main_skeleton(skeletons):
    if len(skeletons) == 0:
        return None
    max_area, selected_skeleton = 0, skeletons[0]
    for skeleton in skeletons:
        valid_points = skeleton[skeleton[:, 2] > 0]
        if valid_points.shape[0] == 0:
            continue
        x_min, y_min = valid_points[:, 0].min(), valid_points[:, 1].min()
        x_max, y_max = valid_points[:, 0].max(), valid_points[:, 1].max()
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area, selected_skeleton = area, skeleton
    return selected_skeleton

# New: Kinematic conversion to joint angles (simplified mapping to 9 humanoid joints)
def skeleton_to_joint_angles(skeleton):
    # Keypoint indices (OpenPose BODY_25): 0=Nose, 1=Neck, 8=RHip, 11=LHip, 9=RKnee, 12=LKnee, 10=RAnkle, 13=LAnkle, 2=RShoulder, 5=LShoulder, 3=RElbow, 6=LElbow, etc.
    # Humanoid joints: [neck_yaw, left_shoulder_pitch, left_elbow, right_shoulder_pitch, right_elbow, left_hip, left_knee, right_hip, right_knee]
    angles = np.zeros(9, dtype=np.float32)
    
    # Neck yaw (from nose to neck, but simplified as angle between head and torso)
    if skeleton[0, 2] > 0 and skeleton[1, 2] > 0:
        v_neck = skeleton[1, :2] - skeleton[0, :2]  # Neck to nose vector
        angles[0] = np.arctan2(v_neck[1], v_neck[0])  # Yaw approximation
    
    # Left shoulder pitch (angle at shoulder)
    if skeleton[5, 2] > 0 and skeleton[6, 2] > 0:
        v_shoulder_elbow = skeleton[6, :2] - skeleton[5, :2]
        angles[1] = np.arctan2(v_shoulder_elbow[1], v_shoulder_elbow[0])
    
    # Left elbow (angle at elbow)
    if skeleton[6, 2] > 0 and skeleton[7, 2] > 0:
        v_elbow_wrist = skeleton[7, :2] - skeleton[6, :2]
        angles[2] = np.arctan2(v_elbow_wrist[1], v_elbow_wrist[0])
    
    # Right shoulder pitch (mirror)
    if skeleton[2, 2] > 0 and skeleton[3, 2] > 0:
        v_shoulder_elbow = skeleton[3, :2] - skeleton[2, :2]
        angles[3] = np.arctan2(v_shoulder_elbow[1], v_shoulder_elbow[0])
    
    # Right elbow
    if skeleton[3, 2] > 0 and skeleton[4, 2] > 0:
        v_elbow_wrist = skeleton[4, :2] - skeleton[3, :2]
        angles[4] = np.arctan2(v_elbow_wrist[1], v_elbow_wrist[0])
    
    # Left hip (angle at hip)
    if skeleton[11, 2] > 0 and skeleton[12, 2] > 0:
        v_hip_knee = skeleton[12, :2] - skeleton[11, :2]
        angles[5] = np.arctan2(v_hip_knee[1], v_hip_knee[0])
    
    # Left knee
    if skeleton[12, 2] > 0 and skeleton[13, 2] > 0:
        v_knee_ankle = skeleton[13, :2] - skeleton[12, :2]
        angles[6] = np.arctan2(v_knee_ankle[1], v_knee_ankle[0])
    
    # Right hip
    if skeleton[8, 2] > 0 and skeleton[9, 2] > 0:
        v_hip_knee = skeleton[9, :2] - skeleton[8, :2]
        angles[7] = np.arctan2(v_hip_knee[1], v_hip_knee[0])
    
    # Right knee
    if skeleton[9, 2] > 0 and skeleton[10, 2] > 0:
        v_knee_ankle = skeleton[10, :2] - skeleton[9, :2]
        angles[8] = np.arctan2(v_knee_ankle[1], v_knee_ankle[0])
    
    return angles

# Process each image and collect theta_init (joint angles)
all_theta_init = {}
for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable file: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    skeletons = []
    if results.pose_landmarks:
        skeleton = mediapipe_to_openpose_landmarks(results.pose_landmarks.landmark, image.shape)
        skeletons.append(skeleton)

    skeletons = np.array(skeletons)
    main_skeleton = select_main_skeleton(skeletons)

    if main_skeleton is None:
        print(f"No skeleton detected in {image_path.name}")
        continue

    # Compute joint angles
    theta_init = skeleton_to_joint_angles(main_skeleton)
    all_theta_init[image_path.stem] = theta_init
    print(f"\nOutput: Initial Pose Vector (theta_init) for {image_path.name}")
    print(theta_init)

    # Visualize skeleton (unchanged)
    vis_img = image.copy()
    for i, (x, y, conf) in enumerate(main_skeleton):
        if conf > 0:
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for (i, j) in BODY_25_CONNECTIONS:
        if main_skeleton[i, 2] > 0 and main_skeleton[j, 2] > 0:
            x1, y1 = int(main_skeleton[i, 0]), int(main_skeleton[i, 1])
            x2, y2 = int(main_skeleton[j, 0]), int(main_skeleton[j, 1])
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(f"\nDisplaying skeleton for: {image_path.name}")
    cv2.imshow("Skeleton", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Export all_theta_init for Module 2
print("\nAll theta_init (joint angles) collected:")
for name, angles in all_theta_init.items():
    print(f"{name}: {angles}")