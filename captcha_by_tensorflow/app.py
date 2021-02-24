# -*- coding: utf-8 -*-
# @Time : 2021/1/25 15:17
# @Author : xiaojie
# @File : app.py
# @Software: PyCharm

import cv2
import numpy as np
from flask import Flask, request

from model import verification_code
model_api = verification_code()

app = Flask(__name__)


@app.route('/crack_captcha', methods=["POST"])
def crack_captcha():
    print("request.args: {}".format(request.args.to_dict()))
    print("request.form: {}".format(request.form.to_dict()))
    photo = request.files['file']
    print("filename:", photo.filename)
    img = photo.read()
    im2 = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    if im2 is not None:
        label = model_api.detect(im2)
        return {"label": label, "message": "成功", "status": "sucess"}
    return {"label": "", "message": "出错", "status": "error"}


if __name__ == "__main__":
    # model_api.detect(r"wait_train_images\EK111V.jpg")
    app.run(debug=False)
