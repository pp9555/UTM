# import cv2
# import numpy as np
# import websocket
# import json
# import time
# import os

# # --- Configuration ---
# WEBSOCKET_URL = "ws://localhost:8080"
# VIDEO_PATH = "test_video.mp4" 
# SPECIMEN_WIDTH_MM = 10.0 # From the paper's Ncorr calibration standard
# FAKE_STRESS_MULTIPLIER = 5.0 

# def generate_ncorr_subsets(roi, spacing=15):
#     """Generates a uniform grid of subsets inside the ROI, exactly like MATLAB Ncorr."""
#     x, y, w, h = roi
#     points = []
#     # Create an evenly spaced grid of points
#     for i in range(x + spacing, x + w - spacing, spacing):
#         for j in range(y + spacing, y + h - spacing, spacing):
#             points.append([[float(i), float(j)]])
#     return np.array(points, dtype=np.float32)

# def main():
#     if not os.path.exists(VIDEO_PATH):
#         print(f"ERROR: Could not find '{VIDEO_PATH}'.")
#         return

#     try:
#         ws = websocket.WebSocket()
#         ws.connect(WEBSOCKET_URL)
#         print("Connected to Dashboard.")
#     except Exception as e:
#         print(f"WebSocket Error: {e}")
#         return

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     ret, frame = cap.read()
#     if not ret:
#         return

#     print("--- NCORR CALIBRATION ---")
#     print(f"Draw ONE box over the middle gauge section.")
#     print(f"Ensure the left and right edges of the box align with the edges of the specimen.")
#     print("Press SPACE or ENTER when done.")
    
#     roi = cv2.selectROI("Ncorr Setup", frame, fromCenter=False, showCrosshair=True)
#     cv2.destroyWindow("Ncorr Setup")

#     if roi[2] == 0 or roi[3] == 0:
#         print("Invalid ROI selected.")
#         return

#     # 1. Calibration and Scaling
#     pixels_per_mm = roi[2] / SPECIMEN_WIDTH_MM
#     print(f"Calibration Complete: {pixels_per_mm:.2f} pixels per mm.")

#     # 2. Subset Generation
#     old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     p0 = generate_ncorr_subsets(roi, spacing=12) # 12-pixel spacing for dense mesh

#     if len(p0) < 10:
#         print("ROI too small to generate sufficient Ncorr subsets.")
#         return

#     # Sort initial points to find the top and bottom rows of the grid
#     initial_y_coords = p0[:, 0, 1]
#     avg_y_top_initial = np.mean(np.sort(initial_y_coords)[:10])
#     avg_y_bottom_initial = np.mean(np.sort(initial_y_coords)[-10:])
#     initial_length_pixels = abs(avg_y_bottom_initial - avg_y_top_initial)

#     lk_params = dict(winSize=(31, 31), maxLevel=4,
#                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#     last_send_time = time.time()
#     cv2.namedWindow('Ncorr MATLAB Style DIC',cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('window_name', 50, 50)
#     last_known_strain = 0 
#     is_paused = False

#     while True:
#         if not is_paused:
#             ret, frame = cap.read()
            
            
#             if not ret:
#                 print("--- END OF VIDEO REACHED ---")
#                 print("Sending fracture signal to database...")
#                 payload = json.dumps({"strain": float(round(last_known_strain, 5)), "stress": 0.0})
#                 ws.send(payload)
#                 time.sleep(0.5)
#                 break
                
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # 3. DIC Processing Analysis
#             p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#             # Keep only subsets that are actively tracking
#             good_new = p1[st == 1]

#             for pt in good_new:
#                 a, b = pt.ravel()
#                 cv2.circle(frame, (int(a), int(b)), 2, (0, 255, 255), -1)

#             # 4. Strain Computation Method
#             if len(good_new) > 10:
#                 current_y_coords = good_new[:, 1]
                
#                 # Dynamic virtual extensometer: track the average of the highest and lowest surviving points
#                 avg_y_top = np.mean(np.sort(current_y_coords)[:10])
#                 avg_y_bottom = np.mean(np.sort(current_y_coords)[-10:])
                
#                 current_length_pixels = abs(avg_y_bottom - avg_y_top)
                
#                 # Calculate strain
#                 delta_l = current_length_pixels - initial_length_pixels
#                 strain = delta_l / initial_length_pixels if initial_length_pixels > 0 else 0
                
#                 # Prevent negative noise
#                 if strain > last_known_strain:
#                     last_known_strain = strain 
                
#                 fake_stress = last_known_strain * FAKE_STRESS_MULTIPLIER

#                 cv2.putText(frame, f"Strain: {last_known_strain:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 cv2.putText(frame, f"Active Subsets: {len(good_new)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
#                 if time.time() - last_send_time > 0.1:
#                     payload = json.dumps({"strain": float(round(last_known_strain, 5)), "stress": float(round(fake_stress, 3))})
#                     ws.send(payload)
#                     last_send_time = time.time()

#                 old_gray = frame_gray.copy()
#                 p0 = good_new.reshape(-1, 1, 2)
#             else:
#                 cv2.putText(frame, "SPECIMEN FRACTURED / TRACKING LOST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         cv2.imshow('Ncorr MATLAB Style DIC', frame)

#         key = cv2.waitKey(30) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             is_paused = not is_paused 

#     ws.close()
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import websocket
import json
import time
import os

# --- Configuration ---
WEBSOCKET_URL = "ws://localhost:8080"
VIDEO_PATH = "test_video.mp4" 
SPECIMEN_WIDTH_MM = 10.0 # From the paper's Ncorr calibration standard

def generate_ncorr_subsets(roi, spacing=15):
    """Generates a uniform grid of subsets inside the ROI, exactly like MATLAB Ncorr."""
    x, y, w, h = roi
    points = []
    # Create an evenly spaced grid of points
    for i in range(x + spacing, x + w - spacing, spacing):
        for j in range(y + spacing, y + h - spacing, spacing):
            points.append([[float(i), float(j)]])
    return np.array(points, dtype=np.float32)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Could not find '{VIDEO_PATH}'.")
        return

    try:
        ws = websocket.WebSocket()
        ws.connect(WEBSOCKET_URL)
        print("Connected to Dashboard.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        return

    print("--- NCORR CALIBRATION ---")
    print(f"Draw ONE box over the middle gauge section.")
    print(f"Ensure the left and right edges of the box align with the edges of the specimen.")
    print("Press SPACE or ENTER when done.")
    
    roi = cv2.selectROI("Ncorr Setup", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Ncorr Setup")

    if roi[2] == 0 or roi[3] == 0:
        print("Invalid ROI selected.")
        return

    # 1. Calibration and Scaling
    pixels_per_mm = roi[2] / SPECIMEN_WIDTH_MM
    print(f"Calibration Complete: {pixels_per_mm:.2f} pixels per mm.")

    # 2. Subset Generation
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = generate_ncorr_subsets(roi, spacing=12) # 12-pixel spacing for dense mesh

    if len(p0) < 10:
        print("ROI too small to generate sufficient Ncorr subsets.")
        return

    # Save an absolute copy of the initial points to calculate total heatmap strain
    absolute_p0 = p0.copy()

    # Sort initial points to find the top and bottom rows of the grid
    initial_y_coords = p0[:, 0, 1]
    avg_y_top_initial = np.mean(np.sort(initial_y_coords)[:10])
    avg_y_bottom_initial = np.mean(np.sort(initial_y_coords)[-10:])
    initial_length_pixels = abs(avg_y_bottom_initial - avg_y_top_initial)

    lk_params = dict(winSize=(31, 31), maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    last_send_time = time.time()
    cv2.namedWindow('Ncorr MATLAB Style DIC', cv2.WINDOW_NORMAL)
    last_known_strain = 0 
    is_paused = False

    # Send command to clear the dashboard before starting
    try:
        ws.send(json.dumps({"command": "CLEAR_CHART"}))
        time.sleep(0.1)
    except:
        pass

    while True:
        if not is_paused:
            ret, frame = cap.read()
            
            if not ret:
                print("--- END OF VIDEO REACHED ---")
                print("Sending fracture signal to database...")
                payload = json.dumps({"strain": float(round(last_known_strain, 5)), "stress": 0.0})
                ws.send(payload)
                time.sleep(0.5)
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 3. DIC Processing Analysis
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Keep only subsets that are actively tracking (st == 1)
            # Flatten the status array to filter both the current points AND the absolute anchors
            st_flat = st.flatten() == 1
            good_new = p1[st_flat]
            absolute_p0 = absolute_p0[st_flat]

            # 4. Strain Computation Method & Heatmap Generation
            if len(good_new) > 10:
                current_y_coords = good_new[:,0, 1]
                
                # --- DIGITAL TWIN HEATMAP ---
                # Calculate the displacement of every individual point from its absolute origin
                original_y_coords = absolute_p0[:, 0, 1]
                displacements = np.abs(current_y_coords - original_y_coords)
                
                min_disp = np.min(displacements)
                max_disp = np.max(displacements)
                range_disp = max_disp - min_disp if max_disp - min_disp > 0 else 1
                
                # Draw the heatmap mesh
                for i, pt in enumerate(good_new):
                    a, b = pt.ravel()
                    norm_disp = (displacements[i] - min_disp) / range_disp
                    
                    # 120 = Blue (Low Strain), 0 = Red (High Strain)
                    hue = int((1.0 - norm_disp) * 120)
                    
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                    
                    # Draw filled circles for the mesh
                    cv2.circle(frame, (int(a), int(b)), 4, color, -1)

                # --- VIRTUAL EXTENSOMETER (For Dashboard Chart) ---
                avg_y_top = np.mean(np.sort(current_y_coords)[:10])
                avg_y_bottom = np.mean(np.sort(current_y_coords)[-10:])
                
                current_length_pixels = abs(avg_y_bottom - avg_y_top)
                delta_l = current_length_pixels - initial_length_pixels
                strain = delta_l / initial_length_pixels if initial_length_pixels > 0 else 0
                
                if strain > last_known_strain:
                    last_known_strain = strain 
                
                # --- REALISTIC STRESS SIMULATION ---
                yield_strain = 0.05
                elastic_modulus = 25.0 
                
                if last_known_strain < yield_strain:
                    fake_stress = last_known_strain * elastic_modulus
                else:
                    yield_stress = yield_strain * elastic_modulus
                    fake_stress = yield_stress + (2.0 * np.log(1 + 30 * (last_known_strain - yield_strain)))

                # UI Overlay
                cv2.putText(frame, f"Strain: {last_known_strain:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Active Subsets: {len(good_new)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Stream to Dashboard
                if time.time() - last_send_time > 0.1:
                    payload = json.dumps({"strain": float(round(last_known_strain, 5)), "stress": float(round(fake_stress, 3))})
                    ws.send(payload)
                    last_send_time = time.time()

                # Update old tracking points for the next frame's optical flow calculation
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                cv2.putText(frame, "SPECIMEN FRACTURED / TRACKING LOST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Ncorr MATLAB Style DIC', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_paused = not is_paused 

    ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()