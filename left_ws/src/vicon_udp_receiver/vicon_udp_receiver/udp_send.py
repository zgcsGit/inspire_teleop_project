import time
import json
import socket
from vicon_dssdk import ViconDataStream

# -----------------------------
# Vicon connect
# -----------------------------
client = ViconDataStream.Client()
client.Connect('localhost:801')
client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
client.EnableMarkerData()

# -----------------------------
# UDP setup
# -----------------------------
UDP_IP = "192.168.10.3"  # linux ip zhicheng
# UDP_IP = "192.168.10.4"  # linux ip tailai 
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# Subject Filter
# -----------------------------
USE_FILTER = True   # false enable all subjects
FILTER_SUBJECTS = ["lefthand"]  # subject name
# FILTER_SUBJECTS = ["umi_test(UMI)"]  #tailai subject


# -----------------------------
# UDP trans loop
# -----------------------------
print("Start sending Vicon markers via UDP...")
while True:
    # get frame
    if not client.GetFrame():
        continue

    # get subject name
    subjects = client.GetSubjectNames()
    #print(f"subject name is {subjects}")

    if USE_FILTER:
        subjects_to_use = [s for s in subjects if s in FILTER_SUBJECTS]

        # check subject list. at least one 
        if not subjects_to_use:
            raise RuntimeError(
                f"there are no subjects"
            )
    else:
        # send all subjects
        subjects_to_use = subjects
    #print(f"subject name is {subjects_to_use}") 

    frame_data = {}
    for subject in subjects_to_use:
        # get marker name
        markers = client.GetMarkerNames(subject)

        for markerName, parentSeg in markers:
            # get marker position
            (x, y, z), occluded = client.GetMarkerGlobalTranslation(subject, markerName)

            # save as frame, json format. markername:[x,y,z]
            frame_data[markerName] = [x, y, z]

    # send msg-
    try:
        msg = json.dumps(frame_data).encode('utf-8')
        sock.sendto(msg, (UDP_IP, UDP_PORT))
    except Exception as e:
        print("UDP send error:", e)

    time.sleep(0.01)  
