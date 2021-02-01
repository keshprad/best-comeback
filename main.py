import dlib
from PIL import Image
import argparse
from imutils import face_utils
import numpy as np
import moviepy.editor as mpy

# Make a arg parser to get image path
terminal_parser = argparse.ArgumentParser(description="Receive input image")
terminal_parser.add_argument("-i", required=True, help="input path")
path = terminal_parser.parse_args().i

# Open the base image, and convert to RGBA
img = Image.open(path)
# Open assets
sunglass = Image.open('assets/sunglasses.png')
smoke = Image.open('assets/smoke.png')
text = Image.open('assets/text.png')

# Resize image if too big
max_width = 750
height, width = img.size
if width > max_width:
    scaled_height = int(max_width * height / width)
    img.thumbnail((max_width, scaled_height))

# Convert to grayscale and represent as numpy array
grayscale = np.array(img.convert('L'))

# Create dlib's frontal face detector and shape predictor
# Frontal Face detector finds the faces in the image
face_detector = dlib.get_frontal_face_detector()
# Shape predictor 68 finds the landmarks (eyes, nose, mouth...)
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Find faces. Exit if no faces found
rects = face_detector(grayscale, 1)
if len(rects) == 0:
    print("No faces found. Goodbye!")
    exit()
print("{} faces found. Processing...".format(len(rects)))

# Loop over faces
faces = []
for rect in rects:
    face = {}
    sunglass_width = rect.right() - rect.left()

    # detector used to find facial landmarks
    shape = shape_predictor(grayscale, rect)
    shape = face_utils.shape_to_np(shape)

    # landmarks needed
    leftEye = shape[36:42]
    rightEye = shape[42:48]
    mouth = shape[48:68]

    # Summarize points: Center of each eyes, and edge of mouth
    leftEyeCenter = leftEye.mean(axis=0).astype("int")
    rightEyeCenter = rightEye.mean(axis=0).astype("int")
    mouthCenter, mouthEdges = rightEye.mean(axis=0).astype("int"), (shape[48], shape[54])

    # Find angle between eye center points
    angle = np.rad2deg(np.arctan2(leftEyeCenter[1] - rightEyeCenter[1], leftEyeCenter[0] - rightEyeCenter[0]))

    # Edit sunglasses to fit face
    # Resizing and downsampling with LANCZOS
    current_sunglass = sunglass.resize((sunglass_width, int(sunglass_width * sunglass.size[1] / sunglass.size[0])),
                                       resample=Image.LANCZOS)
    # Rotate to angle of eyes
    current_sunglass = current_sunglass.rotate(angle, expand=True)
    current_sunglass = current_sunglass.transpose(Image.FLIP_TOP_BOTTOM)

    # Line up props with image and add to faces
    face['sunglasses'] = current_sunglass
    leftEye_x = leftEye[0, 0] - sunglass_width // 4
    leftEye_y = leftEye[0, 1] - sunglass_width // 6
    face['sunglass_position'] = (leftEye_x, leftEye_y)
    faces.append(face)

# if __name__ == "__main__":
