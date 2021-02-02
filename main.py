import os
import dlib
from PIL import Image
import argparse
from imutils import face_utils
import numpy as np
import moviepy.editor as mpy


def deal_with_it(img_path, name):
    print(img_path[6:])
    img, sunglasses, cig, text = open_assets(img_path)
    resize(img)

    # Convert to grayscale and represent as numpy array
    grayscale = np.array(img.convert('L'))

    rects = find_faces(img, grayscale)
    if len(rects) == 0:
        print("No faces found. Goodbye!")
        return
    print("{} faces found. Processing...".format(len(rects)))

    faces = calculate_prop_positions(rects, grayscale, sunglasses)
    text = resize_text(img, text)

    def make_frame(t):
        out_img = img.convert('RGBA')

        if t == 0:  # no glasses
            return np.asarray(out_img)

        for face in faces:
            if t <= duration - text_duration:  # last secs for text
                curr_x = face['sunglasses_pos'][0]  # x is constant
                curr_y = int(face['sunglasses_pos'][1] * t / (duration - text_duration))  # y moves from top -> down
                out_img.paste(face['sunglasses'], (curr_x, curr_y), face['sunglasses'])
            else:  # draw the text
                out_img.paste(face['sunglasses'], face['sunglasses_pos'], face['sunglasses'])
                text_pos = (out_img.width // 2 - text.width // 2, out_img.height - text.height)
                out_img.paste(text, text_pos, text)
        return np.asarray(out_img)

    duration, text_duration = 5, 2
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif("output/{}.gif".format(name), fps=4)


def open_assets(img_path):
    # Open the base image
    img = Image.open(img_path)
    # Open assets
    sunglasses = Image.open('assets/sunglasses.png')
    cig = Image.open('assets/cig.png')
    text = Image.open('assets/text.png')
    return img, sunglasses, cig, text


def resize(img, max_width=750):
    """
    Resize image if too big
    """
    if img.width > max_width:
        scaled_height = int(max_width * img.height / img.width)
        img.thumbnail((max_width, scaled_height))


def find_faces(img, grayscale):
    # Find faces. Exit if no faces found
    face_detector = dlib.get_frontal_face_detector()
    rects = face_detector(grayscale, 1)
    return rects


def calculate_prop_positions(rects, grayscale, sunglasses):
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = []

    for rect in rects:
        face = {}
        sunglasses_width = rect.right() - rect.left()

        # Shape predictor 68 finds the landmarks (eyes, nose, mouth...)
        shape = shape_predictor(grayscale, rect)
        shape = face_utils.shape_to_np(shape)

        # Landmarks needed
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        mouth = shape[48:68]

        # Summarize points: Center of each eyes, and edge of mouth
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")
        mouthCenter = mouth.mean(axis=0).astype("int")

        # Edit props to fit
        # Resizing and downsampling with LANCZOS
        curr_sunglasses = sunglasses.resize(
            (sunglasses_width, int(sunglasses_width * sunglasses.height / sunglasses.width)),
            resample=Image.LANCZOS)
        # current_cig = cig.resize((cig))

        # Find angle between eye center points
        angle = np.rad2deg(np.arctan2(leftEyeCenter[1] - rightEyeCenter[1], leftEyeCenter[0] - rightEyeCenter[0]))
        # Rotate to angle of eyes
        curr_sunglasses = curr_sunglasses.rotate(angle, expand=True)
        curr_sunglasses = curr_sunglasses.transpose(Image.FLIP_TOP_BOTTOM)

        # Line up props with image and add to array
        face['sunglasses'] = curr_sunglasses
        leftEye_x = int(leftEye[0, 0] - sunglasses_width // 4)
        leftEye_y = int(leftEye[0, 1] - sunglasses_width // 6)
        face['sunglasses_pos'] = (leftEye_x, leftEye_y)
        faces.append(face)
    return faces


def resize_text(img, text):
    if img.width > img.height:
        text_width = int(img.width / 2)
    else:
        text_width = int(3 * img.width / 4)
    text_height = int(text_width * text.height / text.width)
    return text.resize((text_width, text_height), resample=Image.LANCZOS).convert('RGBA')


if __name__ == "__main__":
    # for gif in os.listdir('output/'):
    #     os.remove(os.path.join('output/', gif))

    for root, dirs, files in os.walk('input/'):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = file_path[6:].split('.')[0]
                deal_with_it(file_path, name)
                print("\n")     # new line in between images
