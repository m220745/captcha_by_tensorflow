# -*- coding: utf-8 -*-
# @Time : 2020/5/11 10:24
# @Author : xiaojie
# @File : Browser.py
# @Software: PyCharm

import os
import re
import uuid
import threading
import _thread
import time
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.alert import Alert

alert_white_list = [
    {"value": "验证码不正确", "a_or_d": ""},
    {"value": "验证通过", "a_or_d": ""},
]


class Browser(object):
    """
    浏览器对象操作基本功能封装
    """
    _instance_lock = threading.Lock()

    def __new__(cls, headless=True, remote=False, chromedriverPath="", *args, **kwargs):
        if not hasattr(Browser, "_instance"):
            with Browser._instance_lock:
                if not hasattr(Browser, "_instance"):

                    Browser._instance = object.__new__(cls)
                    option = webdriver.ChromeOptions()
                    option.add_argument('disable-infobars')
                    option.add_experimental_option('useAutomationExtension', False)
                    option.add_experimental_option('excludeSwitches', ['enable-automation'])
                    prefs = {
                        "profile.content_settings.exceptions.plugins.*,*.per_resource.adobe-flash-player": 1,
                    }
                    option.add_experimental_option('prefs', prefs)
                    option.add_argument('disable-infobars')
                    option.add_argument('--start-maximized')
                    if headless:
                        # logging.debug("使用后台模式")
                        option.add_argument('--headless')
                        option.add_argument('--no-sandbox')
                        option.add_argument('--disable-dev-shm-usage')
                    if chromedriverPath != "":
                        Browser.browser = webdriver.Chrome(chrome_options=option, executable_path=chromedriverPath)
                    else:
                        Browser.browser = webdriver.Chrome(chrome_options=option)
                    Browser.errorText = ""
                    Browser.alertText = ""
                    Browser.errorInfoText = ""
                    Browser.failedArr = []
                    Browser.stopFlag = False
                    Browser.Continuous_failure = 0
                    _thread.start_new_thread(Browser.checkAlert, (Browser._instance,))
                    _thread.start_new_thread(Browser.checkfailclickArr, (Browser._instance,))
        return Browser._instance

    def checkAlert(self):
        # print("Alert弹窗监控...")
        while True:
            alert_flag = False
            try:
                alertText = self.alertText = self.getAlertText()
                if alertText != '':
                    print("检测到Alert弹窗 -- {}".format(alertText))
                    for _w_item in alert_white_list:
                        if _w_item["value"] in alertText:
                            print("检测到白名单Alert -- {}".format(_w_item["value"]))
                            alert_flag = True
                            self.errorText = ""
                            self.stopFlag = False
                            if _w_item["a_or_d"] == "a":
                                self.acceptAlert()
                            elif _w_item["a_or_d"] == "d":
                                self.dismissAlert()
                    if not alert_flag:
                        print("检测到不在白名单Alert -- {}".format(alertText))
                        self.errorText = "{}".format(alertText)
                        self.stopFlag = True
            except Exception as ex:
                print(ex)
            time.sleep(1)

    def checkfailclickArr(self):
        # print("监控连续点击失败")
        while True:
            if len(self.failedArr) >= 10:
                self.stopFlag = True
                self.errorText = "连续点击失败!"
            time.sleep(3)

    def resetErrorFlag(self):
        self.stopFlag = False
        self.errorText = ""
        self.Continuous_failure = 0
        self.failedArr = []

    def navigateToUrl(self, URL):
        self.browser.get(URL)

    def setBrowserSize(self, width=1920, height=1080):
        self.browser.set_window_size(width, height)
        # self.browser.set_window_position(0, 0)

    def maximize_window(self):
        self.browser.maximize_window()
        # self.browser.set_window_position(0, 0)

    def switchToFrame(self, xpath):
        ret = self.waitForElementVisible(xpath, retry=20)
        if ret:
            self.browser.switch_to.frame(self.browser.find_element_by_xpath(xpath))
            return True
        else:
            return False

    def switchToDefaultContent(self):
        self.browser.switch_to_default_content()

    def click(self, xpath, retry=20, interval=0.5, ignoreFlag=False):
        print("{} : 正在点击--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag and ignoreFlag is False:
                raise Exception(self.errorText)
            try:
                self.browser.find_element_by_xpath(xpath).click()
                self.Continuous_failure = 0
                self.failedArr = []
                return True
            except Exception as e:
                # print(e)
                time.sleep(interval)
                continue
        print("点击{}失败".format(xpath))
        self.failedArr.append(xpath)
        self.Continuous_failure += 1
        return False

    def doubleClick(self, xpath, retry=20, interval=0.5, ignoreFlag=False):
        print("{} : 正在双击--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag and ignoreFlag is False:
                # print("errorText = {}".format(self.errorText))
                raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                action = ActionChains(self.browser)
                action.double_click(ele).perform()
                self.Continuous_failure = 0
                self.failedArr = []
                return True
            except Exception as e:
                # print(e)
                time.sleep(interval)
                continue
        print("双击{}失败".format(xpath))
        self.failedArr.append(xpath)
        self.Continuous_failure += 1
        return False

    def check(self, xpath, retry=20, interval=0.5):
        print("{} : 正在勾选--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                if "input[" in xpath:
                    if not self.browser.find_element_by_xpath(xpath).is_selected():
                        self.browser.find_element_by_xpath(xpath).click()
                        self.Continuous_failure = 0
                    return True

                else:
                    if self.getAttribute(xpath, "class") != "":
                        while "checked" not in self.getAttribute(xpath, "class", retry=6):
                            self.browser.find_element_by_xpath(xpath).click()
                            self.Continuous_failure = 0
                        return True
                    return False
            except Exception as e:
                time.sleep(interval)
                continue
        print("勾选{}失败".format(xpath))
        self.Continuous_failure += 1
        return False

    def uncheck(self, xpath, retry=20, interval=0.5):
        print("{} : 正在取消勾选--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                if "input[" in xpath:
                    if self.browser.find_element_by_xpath(xpath).is_selected():
                        self.browser.find_element_by_xpath(xpath).click()
                        self.Continuous_failure = 0
                    return True

                else:
                    _class = self.getAttribute(xpath, "class")
                    if "checked" in _class:
                        self.browser.find_element_by_xpath(xpath).click()
                        self.Continuous_failure = 0
                    elif _class == "":
                        return False
                    return True
            except Exception as e:
                time.sleep(interval)
                continue
        print("取消勾选{}失败".format(xpath))
        self.Continuous_failure += 1
        return False

    def mouseOver(self, xpath, retry=20, interval=0.5):
        print("{} : 正在移动鼠标到--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                action = ActionChains(self.browser)
                action.move_to_element(ele).perform()

                return True
            except Exception as e:
                time.sleep(interval)
                continue
        # print("移动鼠标到{}失败".format(xpath))
        return False

    def waitForElementVisible(self, xpath, retry=20, interval=0.5):
        # print("{} : 正在等待元素可见--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ret = self.browser.find_element_by_xpath(xpath)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        # print("等待元素{}失败".format(xpath))
        return False

    def waitForElementNoVisible(self, xpath, retry=20, interval=0.5, waitTime=0.1):
        print("{} : 正在等待元素不可见--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        time.sleep(waitTime)
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ret = self.browser.find_element_by_xpath(xpath)
                time.sleep(interval)
                continue
            except Exception as e:
                return True
        # print("等待元素{}失败".format(xpath))
        return False

    def selectOptionByLabel(self, xpath, label, retry=20, interval=0.5):
        print("{} : 正在设置选择下拉选项 -- {} -- 为 -- {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath,
                                                        label))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                Select(self.browser.find_element_by_xpath(xpath)).select_by_visible_text(label)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        # print("选择选项失败!")
        return False

    # 获取Cookie
    def getCookie(self, name=""):
        return self.browser.get_cookie(name=name)

    def closeBrowser(self):
        self.browser.close()
        self.browser.quit()

    def setText(self, xpath, _text, retry=20, interval=0.5):
        print(
            "{} : 正在输入文本框 -- {} -- 为 -- {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath, _text))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                ele.clear()
                time.sleep(0.5)
                ele.send_keys(_text)
                time.sleep(0.5)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        print("输入失败!")
        return False

    def getText(self, xpath, retry=20, interval=0.5):
        # print("{} : 正在获取文本值 -- {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            # if self.stopFlag:
            #     raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                return ele.text
            except Exception as e:
                time.sleep(interval)
                continue
        # print("获取失败!")
        return ""

    def executeJS(self, js):
        if self.stopFlag:
            raise Exception(self.errorText)
        return self.browser.execute_script(js)

    def getAttribute(self, xpath, attributeName, retry=20, interval=0.5):
        print("{} : 正在获取属性值 -- {} -- 属性名 -- {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath,
                                                       attributeName))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                self.scrollToElement(xpath, retry=2)
                ele = self.browser.find_element_by_xpath(xpath)
                ret = ele.get_attribute(attributeName)
                return ret
            except Exception as e:
                time.sleep(interval)
                pass
        print("获取失败！")
        return ""

    def getAlertText(self, retry=20, interval=0.5):
        # print("{} : 正在获取Alert值...".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in range(0, retry):
            try:
                alter = Alert(self.browser)
                return alter.text
            except Exception as e:
                time.sleep(interval)
                continue
        # print("获取Alert值失败！")
        return ''

    def dismissAlert(self):
        if self.stopFlag:
            raise Exception(self.errorText)
        try:
            Alert(self.browser).dismiss()
            return True
        except Exception as e:
            return False

    def acceptAlert(self):
        try:
            try:
                # print("正在关闭Alert对话框...")
                self.browser.switch_to.alert.accept()
            except Exception as ex:
                # print("关闭Alert对话框异常...")
                # print(ex)
                Alert(self.browser).accept()
            return True
        except Exception as e:

            return False

    def refreshPage(self):
        self.browser.refresh()
        self.resetErrorFlag()

    def scrollToElement(self, xpath, retry=20, interval=0.5):
        print("{} : 正在滑动页面到 -- {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                self.browser.execute_script("arguments[0].scrollIntoView(true);", ele)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        print("页面滑动失败！")
        return False

    def takeSnapshot(self, save_path=""):
        date = time.strftime("%Y-%m-%d", time.localtime())
        frontPath = "tmp/{}/".format(date)
        try:
            alertText = self.getAlertText(retry=4)
            if alertText != "":
                self.acceptAlert()
            if save_path == "":
                save_path = frontPath

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imagePath = save_path + "{}.png".format(str(uuid.uuid4()).replace("-", ""))
            self.browser.get_screenshot_as_file(imagePath)
            return imagePath
        except Exception as e:
            print(e)
            return "创建图片失败"

    def takeSnapshotToBase64(self):
        try:
            alertText = self.getAlertText(retry=4)
            if alertText != "":
                self.acceptAlert()
            return self.browser.get_screenshot_as_base64()
        except Exception as e:
            print(e)
            return "截图失败"

    def uploadFile(self, xpath, _file, retry=20, interval=0.5):
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ele = self.browser.find_element_by_xpath(xpath)
                ele.clear()
                time.sleep(0.5)
                ele.send_keys(_file)
                time.sleep(0.5)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        return False

    def getErrorInfoTextForTime(self, retry=10, interval=0.5, contains_str=[]):
        print("获取ErrorInfo内容...")
        returnText = ""
        for i in range(0, retry):
            try:
                for _str in contains_str:
                    if _str in self.errorInfoText:
                        returnText = self.errorInfoText
                        return returnText
                    else:
                        time.sleep(interval)
                        continue
            except Exception as e:
                time.sleep(interval)
                continue
        return returnText

    def getElementByJs(self, xpath, attr="text"):
        fun_js = "function _x(STR_XPATH){var xresult=document.evaluate(STR_XPATH,document,null,XPathResult.ANY_TYPE,null);var xnodes=[];var xres;while(xres=xresult.iterateNext()){xnodes.push(xres)}return xnodes}" + 'return _x("{}");'.format(
            xpath)
        # print(fun_js)
        webElements = self.executeJS(fun_js)
        retusnlist = []
        for el in webElements:
            if attr == "text":
                retusnlist.append(el.text)
            else:
                retusnlist.append(el.get_attribute(attr))
        return retusnlist

    def clickElementByJs(self, xpath, action_type="click", retry=20, interval=0.5):
        print("{} : 正在点击clickElementByJs--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                # print("errorText = {}".format(self.errorText))
                raise Exception(self.errorText)
            try:

                fun_js = "function _x(STR_XPATH,action_type){var xresult=document.evaluate(STR_XPATH,document,null,XPathResult.ANY_TYPE,null);var xnodes=[];var xres;while(xres=xresult.iterateNext()){xnodes.push(xres)}if(action_type==\"click\") xnodes.forEach(function(v){v.click()}); else{} return xnodes}" + 'return _x("{}","{}");'.format(
                    xpath, action_type)
                # print(fun_js)
                webElements = self.executeJS(fun_js)
                if len(webElements) == 0:
                    continue
                else:
                    self.Continuous_failure = 0
                    self.failedArr = []
                    return True
            except Exception as e:
                # print(e)
                time.sleep(interval)
                continue
        print("点击{}失败".format(xpath))
        self.failedArr.append(xpath)
        self.Continuous_failure += 1
        return False

    def removeElementByJs(self, xpath, retry=20, interval=0.5):
        print("{} : 正在点击--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))
        for i in range(0, retry):
            if self.stopFlag:
                # print("errorText = {}".format(self.errorText))
                raise Exception(self.errorText)
            try:

                fun_js = "function _x(STR_XPATH){var xresult=document.evaluate(STR_XPATH,document,null,XPathResult.ANY_TYPE,null);var xnodes=[];var xres;while(xres=xresult.iterateNext()){xnodes.push(xres)}if(action_type==\"click\") xnodes.forEach(function(v){v.remove()}); else{} return xnodes}" + 'return _x("{}");'.format(
                    xpath)
                # print(fun_js)
                webElements = self.executeJS(fun_js)
                if len(webElements) == 0:
                    continue
                else:
                    self.Continuous_failure = 0
                    self.failedArr = []
                    return True
            except Exception as e:
                # print(e)
                time.sleep(interval)
                continue
        # print("点击{}失败".format(xpath))
        self.failedArr.append(xpath)
        self.Continuous_failure += 1
        return False

    def getMatch(self, match, str, returnType="bool"):
        returnB = False
        returnS = ""
        pattern = re.search(r'{}'.format(match), str)
        if pattern:
            returnS = pattern.group()
            returnB = True
            # print("search --> pattern.group() : ", returnS)
        else:
            # print("No match!!")
            returnS = ""
            returnB = False
        if returnType == "bool":
            return returnB
        elif returnType == "str":
            return returnS

    def element_is_displayed(self, xpath, retry=20, interval=0.5):
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                ret = self.browser.find_element_by_xpath(xpath)
                return ret.is_displayed()
            except Exception as e:
                time.sleep(interval)
                continue
        return False

    def getWindowHandle(self, getThis=True, retry=4, interval=0.5):
        handle = None
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                if getThis is False:
                    handle = self.browser.window_handles
                else:
                    handle = self.browser.current_window_handle
                if handle is None:
                    continue
            except Exception as e:
                time.sleep(interval)
                continue
        return handle

    def switchToWindow(self, handle, retry=4, interval=0.5):
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                self.browser.switch_to.window(handle)
                return True
            except Exception as e:
                time.sleep(interval)
                continue
        return False

    def JSESSIONID登录系统(self, JSESSIONID, name="JSESSIONID", path="", expiry=0, navigateTo=""):
        cookie = {'name': name, 'value': JSESSIONID}
        self.browser.delete_all_cookies()
        if path != "":
            cookie["path"] = path
        if expiry != 0:
            cookie["expiry"] = expiry
        self.browser.add_cookie(cookie)

        time.sleep(1)
        if navigateTo != "":
            self.browser.get(navigateTo)
        else:
            self.browser.refresh()

    def 多Cookie登录(self, cookies, expiry=0, navigateTo=""):
        self.browser.delete_all_cookies()

        for c in cookies:
            if expiry != 0:
                c["expiry"] = expiry
            try:
                self.browser.add_cookie(c)
            except Exception as ex:
                print(ex)

        time.sleep(1)
        if navigateTo != "":
            self.browser.get(navigateTo)
        else:
            self.browser.refresh()

    def click_at_relative_point(self, xpath, x_point=0, y_point=0, retry=20, interval=0.5):
        for i in range(0, retry):
            if self.stopFlag:
                # print("errorText = {}".format(self.errorText))
                raise Exception(self.errorText)
            try:
                el = self.browser.find_element_by_xpath(xpath)
                actions = ActionChains(self.browser)
                actions.move_to_element_with_offset(el, x_point, y_point).click().perform()
                self.Continuous_failure = 0
                self.failedArr = []
                return True
            except Exception as e:
                # print(e)
                time.sleep(interval)
                continue
        # print("点击{}失败".format(xpath))
        self.failedArr.append(xpath)
        self.Continuous_failure += 1
        return False

    def browser_ele_screenshot(self, xpath, save_path=None, retry=10, interval=0.5):
        print("{} : 正在截图--{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), xpath))

        element = None
        for i in range(0, retry):
            if self.stopFlag:
                raise Exception(self.errorText)
            try:
                element = self.browser.find_element_by_xpath(xpath)
            except Exception as e:
                time.sleep(interval)
                continue
        if element:
            if save_path:
                element.screenshot(save_path)
            print("截图成功")
            return True
        else:
            print("截图失败")
            return False
