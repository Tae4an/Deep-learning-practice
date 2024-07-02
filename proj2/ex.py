#STEP 1. import modules
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__>='0.3'

# STEP 2. create inference instance
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3. load infence data
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

# STEP 4. inference
faces = app.get(img1)
assert len(faces)==1

faces2 = app.get(img2)
assert len(faces2)==1

# STEP 5. post processing
face1_feat = np.array(faces[0].normed_embedding, dtype=np.float32)
face2_feat = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(face1_feat, face2_feat.T)
print(sims)