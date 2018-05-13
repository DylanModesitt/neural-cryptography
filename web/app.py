from models.steganography.d_2 import Steganography2D
from models.steganography.d_2_evaluator import *
from flask import Flask, render_template, request
app = Flask(__name__)

request_number = 0

secret_num = 0
cover_num = 0

secret = ""
cover = ""
hidden = ""
revealed = ""
combined = ""


model = Steganography2D(dir='./bin/2018-05-03_13:02__61857339')
model.load()

helper = SteganographyImageCoverWrapper(model)
hidden_secret, cover, secret = helper.hide_image_in_image(return_cover=True)
revealed_secret = array_to_img(helper.decode_image_in_cover(hidden_secret))

@app.route('/')
def hello_world():
    folder1 = './web/static/data/video_output/'
    folder2 = './web/static/data/image_output/'
    for folder in [folder1, folder2]:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
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
    secret_num = 0
    cover_num = 0

    return render_template("viv.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed, combined=combined)

@app.route('/get_pip', methods=['GET'])
def get_pip():
    secret = ""
    cover = ""
    hidden = ""
    revealed = ""
    secret_num = 0
    cover_num = 0

    return render_template("pip.html", secret=secret, cover=cover,
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

    hidden = ""
    revealed = ""
    combined = ""

    if (request.form.get('secret', type=int) != None):
        secret_num = request.form.get('secret', type=int)
        secret = "scaled_videos/" + str(secret_num) + ".mp4"
    if (request.form.get('cover', type=int) != None):
        cover_num = request.form.get('cover', type=int)
        cover = "scaled_videos/" + str(cover_num) + ".mp4"

    return render_template("viv.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed, combined=combined)

@app.route('/select_image', methods=['POST'])
def select_image():
    global cover
    global secret
    global hidden
    global revealed
    global combined
    global secret_num
    global cover_num

    hidden = ""
    revealed = ""
    combined = ""

    if (request.form.get('secret', type=int) != None):
        secret_num = request.form.get('secret', type=int)
        secret = "images/" + "{:05}".format(secret_num) + ".png"
    if (request.form.get('cover', type=int) != None):
        cover_num = request.form.get('cover', type=int)
        cover = "images/" + "{:05}".format(cover_num) + ".png"

    return render_template("pip.html", secret=secret, cover=cover,
                               hidden=hidden, revealed=revealed)


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

    if secret_num != 0 and cover_num != 0:
        video_in_video(helper, secret_num, cover_num, request_number)
        path = "./data/video_output/"
        hidden = path + "hidden" + str(request_number) + ".mp4"
        revealed = path + "revealed" + str(request_number) + ".mp4"
        combined = path + "combined" + str(request_number) + ".mp4"
        request_number += 1

    secret_num = 0
    cover_num = 0

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
    global secret_num
    global cover_num

    if secret_num != 0 and cover_num != 0:
        picture_in_picture(helper, secret_num, cover_num, request_number)
        path = "./data/image_output/"
        hidden = path + "hidden" + str(request_number) + ".png"
        revealed = path + "revealed" + str(request_number) + ".png"
        request_number += 1

    secret_num = 0
    cover_num = 0

    return render_template("pip.html", secret=secret, cover=cover,
                           hidden=hidden, revealed=revealed)


if __name__ == '__main__':
    app.run(debug=True)
