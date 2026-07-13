import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def parse_ntu_skeleton(file_path):
    """
    Parses an NTU RGB+D .skeleton file to extract 3D joint coordinates.
    Input: Path to .skeleton file
    Output: Numpy array of shape (Frames, 25, 3) and (Frames, 25, 4)
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    num_frames = int(lines[0].strip())
    frames_data = []
    kinect_quats = [] # Added to store built-in quaternions
    current_line = 1
    
    for f_idx in range(num_frames):
        num_bodies = int(lines[current_line].strip())
        current_line += 1
        current_line += 1 # Skip body metadata (ID, flags, etc.)
        
        num_joints = int(lines[current_line].strip())
        current_line += 1
        
        joints = []
        k_quats = []
        for j_idx in range(num_joints):
            joint_data = list(map(float, lines[current_line].strip().split()))
            # NTU format: x, y, z are the first three floats
            joints.append(joint_data[:3])
            
            # Kinect orientations: W=index 7, X=8, Y=9, Z=10
            # Scipy uses [x, y, z, w] order
            k_quats.append([joint_data[8], joint_data[9], joint_data[10], joint_data[7]])
            current_line += 1
            
        frames_data.append(np.array(joints))
        kinect_quats.append(np.array(k_quats))
        
    return np.array(frames_data), np.array(kinect_quats)

def get_bone_quaternion(p_start, p_end, ref_vec):
    """
    Computes shortest arc rotation from a reference N-pose vector 
    to the actual observed bone vector.
    """
    bone_vec = p_end - p_start
    norm = np.linalg.norm(bone_vec)
    
    if norm < 1e-6:
        return np.array([0, 0, 0, 1])
        
    bone_vec /= norm
    
    axis = np.cross(ref_vec, bone_vec)
    axis_norm = np.linalg.norm(axis)
    dot = np.clip(np.dot(ref_vec, bone_vec), -1.0, 1.0)
    angle = np.arccos(dot)
    
    if axis_norm < 1e-6:
        if dot > 0: 
            return np.array([0, 0, 0, 1]) 
        else:
            temp_axis = np.array([1, 0, 0]) if abs(ref_vec[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(ref_vec, temp_axis)
            axis /= np.linalg.norm(axis)
            return R.from_rotvec(axis * np.pi).as_quat()

    axis /= axis_norm
    return R.from_rotvec(axis * angle).as_quat()

def process_to_local_quats(frames):
    """
    Transforms global 3D joints into local relative quaternions for 17 segments.
    Standardizes NTU data to align with Xsens N-pose identity.
    """
    UP = np.array([0, 1, 0])
    DOWN = np.array([0, -1, 0])
    FORWARD = np.array([0, 0, 1])
    RIGHT = np.array([1, 0, 0])
    LEFT = np.array([-1, 0, 0])

    segments_config = [
        (2, 1, UP),        # 0: Pelvis (1->2)
        (3, 2, UP),        # 1: T8 (2->3)
        (4, 3, UP),        # 2: Head (3->4)
        (9, 21, RIGHT),    # 3: R Shoulder (21->9)
        (10, 9, DOWN),     # 4: R Upper Arm (9->10)
        (11, 10, DOWN),    # 5: R Forearm (10->11)
        (12, 11, DOWN),    # 6: R Hand (11->12)
        (5, 21, LEFT),     # 7: L Shoulder (21->5)
        (6, 5, DOWN),      # 8: L Upper Arm (5->6)
        (7, 6, DOWN),      # 9: L Forearm (6->7)
        (8, 7, DOWN),      # 10: L Hand (7->8)
        (18, 17, DOWN),    # 11: R Upper Leg (17->18)
        (19, 18, DOWN),    # 12: R Lower Leg (18->19)
        (20, 19, FORWARD), # 13: R Foot (19->20)
        (14, 13, DOWN),    # 14: L Upper Leg (13->14)
        (15, 14, DOWN),    # 15: L Lower Leg (14->15)
        (16, 15, FORWARD)  # 16: L Foot (15->16)
    ]
    
    # Kinematic hierarchy: {Segment_Index: Parent_Segment_Index}
    # q_local = inverse(q_parent) * q_child
    hierarchy = {
        0: None, 1: 0, 2: 1, 3: 1, 4: 3, 5: 4, 6: 5, 7: 1, 8: 7, 9: 8, 10: 9, 11: 0, 12: 11, 13: 12, 14: 0, 15: 14, 16: 15
    }

    processed_seq = []
    # Store global quats for comparison debugging
    debug_globals = []

    for f in frames:
        global_quats = []
        for child, parent, ref in segments_config:
            global_quats.append(get_bone_quaternion(f[parent-1], f[child-1], ref))
        
        debug_globals.append(global_quats)

        local_quats = []
        for i in range(len(global_quats)):
            parent_idx = hierarchy[i]
            q_child = R.from_quat(global_quats[i])
            
            if parent_idx is None:
                local_quats.append(q_child.as_quat())
            else:
                q_parent = R.from_quat(global_quats[parent_idx])
                q_local = q_parent.inv() * q_child
                local_quats.append(q_local.as_quat())
        
        processed_seq.append(np.array(local_quats))
        
    return np.array(processed_seq), np.array(debug_globals)

# --- Main Execution ---
if __name__ == "__main__":
    test_file = "NTU-SkeletalData/s001_to_s017/S001C001P001R001A001.skeleton"
    
    # Updated to receive kinect orientations
    ntu_frames, kinect_raw_quats = parse_ntu_skeleton(test_file)

    if ntu_frames is not None:
        local_quat_sequence, global_quat_sequence = process_to_local_quats(ntu_frames)

        print(f"Successfully processed: {os.path.basename(test_file)}")
        print(f"Final Tensor Shape: {local_quat_sequence.shape}")

        print("\n=== KINECT VS. OUR CALCULATED GLOBAL QUATS (Frame 0) ===")
        # We compare GLOBAL quats because Kinect quats are in camera-space
        
        # 1. Pelvis: Segment 0 (Joint 1->2) vs Kinect Joint 1
        print(f"Pelvis (Seg 0):")
        print(f"  Ours:   {global_quat_sequence[0, 0]}")
        print(f"  Kinect: {kinect_raw_quats[0, 0]}")

        # 2. R Upper Arm: Segment 4 (Joint 9->10) vs Kinect Joint 9
        print(f"R Upper Arm (Seg 4):")
        print(f"  Ours:   {global_quat_sequence[0, 4]}")
        print(f"  Kinect: {kinect_raw_quats[0, 8]}") # Index 8 is Joint 9

        # 3. R Upper Leg: Segment 11 (Joint 17->18) vs Kinect Joint 17
        print(f"R Upper Leg (Seg 11):")
        print(f"  Ours:   {global_quat_sequence[0, 11]}")
        print(f"  Kinect: {kinect_raw_quats[0, 16]}") # Index 16 is Joint 17