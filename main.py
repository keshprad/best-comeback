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
cig = Image.open('assets/cig.png')
text = Image.open('assets/text.png')

# Resize image if too big
max_width = 750
if img.width > max_width:
    scaled_height = int(max_width * img.height / img.width)
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

    # Edit props to fit
    # Resizing and downsampling with LANCZOS
    current_sunglass = sunglass.resize((sunglass_width, int(sunglass_width * sunglass.height / sunglass.width)),
                                       resample=Image.LANCZOS)
    # current_cig = cig.resize((cig))

    # Rotate to angle of eyes
    current_sunglass = current_sunglass.rotate(angle, expand=True)
    current_sunglass = current_sunglass.transpose(Image.FLIP_TOP_BOTTOM)

    # Line up props with image and add to faces
    face['sunglasses'] = current_sunglass
    leftEye_x = int(leftEye[0, 0] - sunglass_width // 4)
    leftEye_y = int(leftEye[0, 1] - sunglass_width // 6)
    face['sunglass_pos'] = (leftEye_x, leftEye_y)
    faces.append(face)

duration, text_duration = 5, 2
if img.width > img.height:
    text_width = int(img.width / 2)
else:
    text_width = int(3 * img.width / 4)
text_height = int(text_width * text.height / text.width)
text = text.resize((text_width, text_height), resample=Image.LANCZOS).convert('RGBA')


def make_frame(t):
    out_img = img.convert('RGBA')

    if t == 0:  # no glasses
        return np.asarray(out_img)

    for face in faces:
        if t <= duration - text_duration:  # last secs for text
            curr_x = face['sunglass_pos'][0]  # x is constant
            curr_y = int(face['sunglass_pos'][1] * t / (duration - text_duration))  # y moves from top -> down
            out_img.paste(face['sunglasses'], (curr_x, curr_y), face['sunglasses'])
            out_img.show()
        else:  # draw the text
            out_img.paste(face['sunglasses'], face['sunglass_pos'], face['sunglasses'])
            text_pos = (out_img.width // 2 - text.width // 2, out_img.height - text.height)
            out_img.paste(text, text_pos, text)
    return np.asarray(out_img)


animation = mpy.VideoClip(make_frame, duration=duration)
animation.write_gif("DealWithIt.gif", fps=4)
