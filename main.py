import os
import dlib
from PIL import Image
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

    # Create final_img to avoid creating it multiple times in make_frame method
    final_img = img.convert('RGBA')
    text_pos = (final_img.width // 2 - text.width // 2, final_img.height - text.height)
    final_img.paste(text, text_pos, text)
    for face in faces:
        final_img.paste(face['sunglasses'], face['sunglasses_pos'], face['sunglasses'])


    def make_frame(t):
        out_img = img.convert('RGBA')

        if t == 0:  # no glasses
            return np.asarray(out_img)

        if t <= duration - text_duration:  # last secs for text
            for face in faces:
                curr_x = face['sunglasses_pos'][0]  # x is constant
                curr_y = int(face['sunglasses_pos'][1] * t / (duration - text_duration))  # y moves from top -> down
                out_img.paste(face['sunglasses'], (curr_x, curr_y), face['sunglasses'])
            return np.asarray(out_img)
        else:
            # If I create the image here, I create the same image (text_duration * fps) times
            # So instead, I can create the final_img before this method to only create the image only once
            return np.asarray(final_img)

    duration, text_duration = 4, 2
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif("output/{}.gif".format(name), fps=16)


def open_assets(img_path):
    # Open the base image
    img = Image.open(img_path)
    # Open assets
    sunglasses = Image.open('assets/sunglasses.png')
    cig = Image.open('assets/cig.png')
    text = Image.open('assets/text.png')
    return img, sunglasses, cig, text


def resize(img, max_width=1200):
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
        glasses_center = np.array([leftEyeCenter, rightEyeCenter]).mean(axis=0).astype('int')
        sunglasses_x = int(glasses_center[0] - curr_sunglasses.width // 2)
        sunglasses_y = int(glasses_center[1] - curr_sunglasses.height // 2)
        face['sunglasses_pos'] = (sunglasses_x, sunglasses_y)
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
                print("\n")  # new line in between images
