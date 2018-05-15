# system
import io
import time

# lib
import shutil
from PIL import Image
from flask import Flask, render_template, request
from keras.preprocessing.image import array_to_img, img_to_array
import cv2
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc


# self
from models.steganography.d_2 import Steganography2D
from models.steganography.d_2_evaluator import *

# ===== flask init ==============================
app = Flask(__name__)

# ===== self init ===============================
model = Steganography2D(dir='./bin/2018-05-03_13:02__61857339')
model.load()
helper = SteganographyImageCoverWrapper(model)

# bug fix (call encode and decode methods before starting web-app)
hidden_secret, cover, secret = helper.hide_image_in_image(return_cover=True)
revealed_secret = array_to_img(helper.decode_image_in_cover(hidden_secret))

# ===== global vars =============================
request_number = 0

secret_image = None
cover_image = None
secret_video = []
cover_image = []

secret = ""
cover = ""
hidden = ""
revealed = ""
combined = ""

image_res = (32, 32)

data_dir = './web/static/'
image_dir = "data/image_output/"
video_dir = "data/video_output/"
frame_dir = "data/scaled_frames/"
secret_frames_directory = ""
cover_frames_directory = ""


@app.route('/')
def hello_world():
    folder1 = './web/static/data/video_output/'
    folder2 = './web/static/data/image_output/'
    folder3 = './web/static/data/scaled_frames/'
    for folder in [folder1, folder2, folder3]:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    return render_template("viv.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed, combined=combined)

@app.route('/get_viv', methods=['GET'])
def get_viv():
    secret = ""
    cover = ""
    hidden = ""
    revealed = ""
    combined = ""

    return render_template("viv.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed, combined=combined)

@app.route('/get_pip', methods=['GET'])
def get_pip():
    secret = ""
    cover = ""
    hidden = ""
    revealed = ""

    return render_template("pip.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed)

@app.route('/get_tip', methods=['GET'])
def get_tip():
    secret = ""
    cover = ""
    hidden = ""
    revealed = ""

    return render_template("tip.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed)

@app.route('/select_video', methods=['POST'])
def select_video():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global secret_num
    global cover_num
    global secret_frames_directory
    global cover_frames_directory

    hidden = ""
    revealed = ""
    combined = ""

    secret_frames_directory = data_dir + frame_dir + str(request_number) + "/secret/"
    cover_frames_directory = data_dir + frame_dir + str(request_number) + "/cover/"

    if request.files.get("secret"):
        file = request.files["secret"]
        if file:
            if os.path.exists(secret_frames_directory):
                shutil.rmtree(secret_frames_directory)
            os.makedirs(secret_frames_directory)

            secret = video_dir + "secret" + str(request_number) + ".mp4"
            file.save(data_dir + secret)

            secret_capture = cv2.VideoCapture(data_dir + secret)

            count = 0
            while (secret_capture.isOpened()):
                ret, frame = secret_capture.read()
                if not ret:
                    break
                frame = prepare_image(array_to_img(frame[:,:,::-1]))
                frame.save(secret_frames_directory + "{:05}".format(count) + ".png", 'PNG')
                count += 1

    if request.files.get("cover"):
        file = request.files["cover"]
        if file:
            if os.path.exists(cover_frames_directory):
                shutil.rmtree(cover_frames_directory)
            os.makedirs(cover_frames_directory)

            cover = video_dir + "cover" + str(request_number) + ".mp4"
            file.save(data_dir + cover)

            cover_capture = cv2.VideoCapture(data_dir + cover)

            count = 0
            while (cover_capture.isOpened()):
                ret, frame = cover_capture.read()
                if not ret:
                    break
                frame = prepare_image(array_to_img(frame[:,:,::-1]))
                frame.save(cover_frames_directory + "{:05}".format(count) + ".png", 'PNG')
                count += 1

    return render_template("viv.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed, combined=combined)

@app.route('/select_image', methods=['POST'])
def select_image():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global secret_image
    global cover_image
    global request_number

    hidden = ""
    revealed = ""
    combined = ""

    if request.files.get("secret"):
        # read the image in PIL format
        secret_image = request.files["secret"].read()
        secret_image = Image.open(io.BytesIO(secret_image))

        # preprocess the image and save itn
        secret_image = prepare_image(secret_image, target=image_res)
        secret = image_dir + "secret" + str(request_number) + ".png"
        secret_image.save(data_dir + secret, 'PNG')

    if request.files.get("cover"):
        # read the image in PIL format
        cover_image = request.files["cover"].read()
        cover_image = Image.open(io.BytesIO(cover_image))

        # preprocess the image and save it
        cover_image = prepare_image(cover_image, target=image_res)
        cover = image_dir + "cover" + str(request_number) + ".png"
        cover_image.save(data_dir + cover, 'PNG')

    return render_template("pip.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed)

@app.route('/select_image_tip', methods=['POST'])
def select_image_tip():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global secret_image
    global cover_image
    global request_number

    hidden = ""
    revealed = ""
    combined = ""

    if request.files.get("secret"):
        # read the image in PIL format
        secret_image = request.files["secret"].read()
        secret_image = Image.open(io.BytesIO(secret_image))

        # preprocess the image and save itn
        secret_image = prepare_image(secret_image, target=image_res)
        secret = image_dir + "secret" + str(request_number) + ".png"
        secret_image.save(data_dir + secret, 'PNG')

    if request.files.get("cover"):
        # read the image in PIL format
        cover_image = request.files["cover"].read()
        cover_image = Image.open(io.BytesIO(cover_image))

        # preprocess the image and save it
        cover_image = prepare_image(cover_image, target=image_res)
        cover = image_dir + "cover" + str(request_number) + ".png"
        cover_image.save(data_dir + cover, 'PNG')

    return render_template("tip.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed)

@app.route('/select_text', methods=['POST'])
def select_text():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global secret_image
    global cover_image
    global request_number

    hidden = ""
    revealed = ""
    combined = ""

    secret_text = request.form["secret"]
    if secret_text:
        secret = secret_text

    return render_template("tip.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed)

def prepare_image(image, target=image_res):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image
    image = image.resize(target)

    # return the processed image
    return image

@app.route('/compute_video', methods=['POST'])
def compute_video():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global helper
    global request_number
    global secret_num
    global cover_num

    if secret and cover:
        video_in_video(helper, secret_frames_directory,
                       cover_frames_directory, request_number)
        path = "./data/video_output/"
        hidden = path + "hidden" + str(request_number) + ".mp4"
        revealed = path + "revealed" + str(request_number) + ".mp4"
        combined = path + "combined" + str(request_number) + ".mp4"
        request_number += 1

    secret_image = None
    cover_image = None

    return render_template("viv.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed, combined=combined)

@app.route('/compute_image', methods=['POST'])
def compute_image():
    global cover
    global secret
    global hidden
    global revealed
    global helper
    global request_number
    global secret_image
    global cover_image

    if secret_image and cover_image:
        picture_in_picture(helper, secret_image, cover_image, request_number)
        path = "./data/image_output/"
        hidden = path + "hidden" + str(request_number) + ".png"
        revealed = path + "revealed" + str(request_number) + ".png"
        request_number += 1

    secret_image, cover_image = None, None

    return render_template("pip.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed)

@app.route('/compute_text', methods=['POST'])
def compute_text():
    global cover
    global secret
    global hidden
    global revealed
    global helper
    global request_number
    global secret_image
    global cover_image

    if secret and cover_image:
        revealed = text_in_picture(helper, secret, cover_image, request_number)
        path = "./data/image_output/"
        hidden = path + "hidden" + str(request_number) + ".png"
        request_number += 1

    cover_image = None

    return render_template("tip.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed)

if __name__ == '__main__':
    app.run(debug=True)
