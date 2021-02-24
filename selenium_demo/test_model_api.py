# -*- coding: utf-8 -*-
# @Time : 2021/1/26 16:11
# @Author : xiaojie
# @File : test_model_api.py
# @Software: PyCharm

import time
import shutil
import requests
import Browser


browser = Browser.Browser(headless=False, chromedriverPath="./chromedriver.exe")
tmp_img_path = "./tmp/temp.png"
if __name__ == '__main__':
    try:
        browser.browser.maximize_window()
        pass_count = 0
        failed_count = 0
        for i in range(0, 100):

            browser.navigateToUrl("https://www.jq22.com/yanshi19634")

            browser.switchToDefaultContent()
            browser.switchToFrame("//iframe[@id='iframe']")
            # 等待验证码元素出现
            browser.waitForElementVisible("//span[@class='code']")
            # 截取验证密码元素图片
            browser.browser_ele_screenshot(xpath="//span[@class='code']", save_path=tmp_img_path)

            # 调用验证码识别api接口
            files = {"file": open(tmp_img_path, "rb")}
            result = requests.post("http://127.0.0.1:5000/crack_captcha", files=files)
            res = result.json()
            print(res)

            # 填入api返回的验证码结果
            browser.setText("//input[@class='input-code']", res["label"])

            browser.click("//div[1]")
            time.sleep(1)
            alert_text = browser.getAlertText()
            if alert_text == "验证通过":
                pass_count += 1
            else:
                failed_count += 1
                true_label = browser.getAttribute("//span[@class='code']/input", "value")
                shutil.copy(tmp_img_path, "./tmp/error_img/{}.jpg".format(true_label))
                print("正确内容:", true_label, "识别结果:", res["label"])
            browser.acceptAlert()

        print("pass:", pass_count, "failed:", failed_count)

    except Exception as e:
        print(e)
    finally:
        browser.acceptAlert()
        browser.closeBrowser()
