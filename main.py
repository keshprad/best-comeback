import dlib
from PIL import Image
from imutils import face_utils
import numpy as np
import moviepy.editor as mpy
import typer
import pathlib
import os


def deal_with_it(img_path, path):
    img, sunglasses, cig, text = open_assets(img_path)
    resize(img)

    # Convert to grayscale and represent as numpy array
    grayscale = np.array(img.convert('L'))

    rects = find_faces(img, grayscale)
    if len(rects) == 0:
        typer.secho('No faces found. Goodbye!', fg=typer.colors.RED)
        return
    typer.secho(f'{len(rects)} faces found. Processing...',
                fg=typer.colors.GREEN)

    faces = calculate_prop_positions(rects, grayscale, (sunglasses, cig))
    text = resize_text(img, text)

    # Create final_img to avoid creating it multiple times in make_frame method
    final_img = img.convert('RGBA')
    for face in faces:
        final_img.paste(face['sunglasses'],
                        face['sunglasses_pos'], face['sunglasses'])
        final_img.paste(face['cig'], face['cig_pos'], face['cig'])
    text_pos = (final_img.width // 2 - text.width //
                2, final_img.height - text.height)
    final_img.paste(text, text_pos, text)

    def make_frame(t):
        out_img = img.convert('RGBA')

        if t == 0:  # no glasses
            return np.asarray(out_img)

        if t <= duration - text_duration:  # last secs for text
            for face in faces:
                sg_currPos = face['sunglasses_pos'][0], int(
                    face['sunglasses_pos'][1] * t / (duration - text_duration))  # y moves from top -> down
                cig_currPos = face['cig_pos'][0], int(
                    face['cig_pos'][1] * t / (duration - text_duration))
                out_img.paste(face['sunglasses'],
                              sg_currPos, face['sunglasses'])
                out_img.paste(face['cig'], cig_currPos, face['cig'])
            return np.asarray(out_img)
        else:
            # If I create the image here, I create the same image (text_duration * fps) times
            # So instead, I can create the final_img before this method to only create the image only once
            return np.asarray(final_img)

    duration, text_duration = 4, 2
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif(f"{path}.gif", fps=16)


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


def calculate_prop_positions(rects, grayscale, props):
    shape_predictor = dlib.shape_predictor(
        'shape_predictor_68_face_landmarks.dat')
    faces = []
    sunglasses, cig = props

    for rect in rects:
        face = {}

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

        # Edit size of props
        # Resizing and downsampling with LANCZOS
        def pythagorean(p1, p2): return (
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        sunglasses_width = int(1.625 * pythagorean(shape[45], shape[36]))
        curr_sunglasses = sunglasses.resize(
            (sunglasses_width, int(sunglasses_width *
             sunglasses.height / sunglasses.width)),
            resample=Image.LANCZOS)

        cig_width = rect.right() - rect.left()
        curr_cig = cig.resize(
            (cig_width, int(cig_width * cig.height / cig.width)), resample=Image.LANCZOS)

        # Find angles and position of props
        # Sunglasses angle and position
        sunglasses_angle = np.rad2deg(np.arctan2(
            leftEyeCenter[1] - rightEyeCenter[1], leftEyeCenter[0] - rightEyeCenter[0]))
        curr_sunglasses = curr_sunglasses.rotate(
            sunglasses_angle, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
        glasses_center = np.array(
            [leftEyeCenter, rightEyeCenter]).mean(axis=0).astype('int')
        sunglasses_x, sunglasses_y = int(
            glasses_center[0] - curr_sunglasses.width // 2), int(glasses_center[1] - curr_sunglasses.height // 2)
        # Cig angle and position
        cig_angle = np.rad2deg(np.arctan2(
            shape[54][1] - shape[48][1], shape[54][0] - shape[48][0]))

        if pythagorean(mouthCenter, shape[3]) < pythagorean(mouthCenter, shape[13]):
            curr_cig = curr_cig.rotate(
                cig_angle, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            cig_x, cig_y = int(
                mouthCenter[0] - curr_cig.width // 2), int(mouthCenter[1] - curr_cig.height // 2)
        else:
            curr_cig = curr_cig.rotate(-1*cig_angle, expand=True)
        cig_x, cig_y = int(
            mouthCenter[0] - curr_cig.width // 2), int(mouthCenter[1] - curr_cig.height // 2)

        # Add props to array
        face['sunglasses'], face['cig'] = curr_sunglasses, curr_cig
        face['sunglasses_pos'], face['cig_pos'] = (
            sunglasses_x, sunglasses_y), (cig_x, cig_y)
        faces.append(face)
    return faces


def resize_text(img, text):
    if img.width > img.height:
        text_width = int(img.width / 2)
    else:
        text_width = int(3 * img.width / 4)
    text_height = int(text_width * text.height / text.width)
    return text.resize((text_width, text_height), resample=Image.LANCZOS).convert('RGBA')


def main(path: pathlib.Path = typer.Argument(..., help="The path to your image file or directory with multiple images")):
    if path.is_dir():
        for child in path.iterdir():
            if not child.is_dir():
                main(child)
    elif path.is_file() and is_image(path):
        typer.secho(str(path))
        without_ext = os.path.splitext(path)[0]
        deal_with_it(path, without_ext)
    else:
        typer.secho(str(path))
        typer.secho("Path is not a valid image. Skipping...",
                    fg=typer.colors.RED)


def is_image(path):
    try:
        img = Image.open(path)  # make sure we can open as Image
        try:
            img.seek(1)  # make sure not a gif
        except EOFError:
            return True
        else:
            return False
    except IOError:
        return False


if __name__ == "__main__":
    typer.run(main)
