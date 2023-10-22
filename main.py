import io
import json
import os
import socket
import sys
import threading
import winreg
import time
import base64
import requests
import urllib
import cv2
import numpy as np
from PIL import Image
from aip import AipFace
from PySide6.QtCore import QTimer, QThread, QSettings
from PySide6.QtGui import QIcon, Qt, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, \
    QLineEdit, QComboBox, QMessageBox, QFileDialog
from concurrent.futures import ThreadPoolExecutor
import queue
from urllib.parse import quote
import paho.mqtt.client as mqtt
import time
import urllib.request

# 创建一个全局队列
results_queue = queue.Queue()
CAPTURE_IMAGE_DATA = None
SAVE_VIDEO_PATH = ""
face_probability = None
results_lock = threading.Lock()
is_client_thread_running = False
is_face_thread_running = False
DetectedStatus = False
# 设置默认值
user_group_id = "N/A"
user_id = "N/A"
user_info = "N/A"
user_score = "N/A"

# 百度相关
APP_ID = "41225147"
API_KEY = "YGrOXTbCyyeZMQtoLu248lGE"
SECRET_KEY = "aa1gqbx7b9eSQ6B62ZoAFhzxEGDQQXT4"
passwd = '4321'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# Onenet相关
ClientId = "1073624288"  # 设备ID
Username = "596314"  # 产品ID
accesskey = "f81pJg/oRenty5L1qosL8Z51Db5tZngw/iF33vwCCWc="
APIKey = "WdhG5yLD1sp49nzaN9gmtpUDb8o="

# 创建一个最大可容纳n个工作线程的线程池
executor = ThreadPoolExecutor(max_workers=1)
# 加载预训练好的模型
net = cv2.dnn.readNetFromTensorflow('./model/opencv_face_detector_uint8.pb', './model/opencv_face_detector.pbtxt')


# 获取token
def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


# 图片转码base64
def picToBase64(path, urlencoded=False):
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read())
    image = str(data, 'utf-8')
    return image


# 判断图片是否为BASE64
def is_base64(string):
    try:
        # 尝试将字符串解码为 base64
        base64.b64decode(string)
        return True
    except:
        return False


# 人脸对比
def face_match(filepath1, filepath2):
    image1 = picToBase64(filepath1)
    image2 = picToBase64(filepath2)
    result = client.match([
        {
            'image': image1,
            'image_type': 'BASE64',
        },
        {
            'image': image2,
            'image_type': 'BASE64',
        }])
    print(result)  # 打印出所有的信息


# 人脸检测
def face_detect(filepath):
    image = picToBase64(filepath)
    imageType = "BASE64"
    result = client.detect(image, imageType)
    print(result)
    return result


# 人脸搜索MN
def searchMN(filepath):
    image = picToBase64(filepath)
    imageType = "BASE64"
    result = client.multiSearch(image, imageType, "2023_personal_488")
    print(result)
    return result


# 人脸搜索1N
def face_search(filepath, groupIdList):  # 人脸库搜索  222207 groupIdList="你的用户组名称"
    image = picToBase64(filepath)
    imageType = "BASE64"
    result = client.search(image, imageType, groupIdList)
    print(result)  # 打印出所有信息


# 人脸库增加 地址 组 用户
def face_add(filepath, groupid, userid):
    image = picToBase64(filepath)
    imageType = "BASE64"
    result = client.addUser(image, imageType, groupid, userid)
    if result['error_code'] == 0:
        print("增加人脸成功")
    else:
        print("增加人脸失败")


# 框选人脸模型 功能：识别人脸，框选出人脸
def boxSelectFace(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    conf_threshold = 0.7

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * image.shape[1])
            y1 = int(detections[0, 0, i, 4] * image.shape[0])
            x2 = int(detections[0, 0, i, 5] * image.shape[1])
            y2 = int(detections[0, 0, i, 6] * image.shape[0])
            bboxes.append([x1, y1, x2, y2])

    return bboxes


# 人脸识别综合
def face_search_ALL(filepath):
    while 1:
        picToBase64(filepath)
        result = face_detect(filepath)
        print(f"\nresult{result}\n")
        if result['result'] is None:
            time.sleep(2)
        else:
            try:
                face_probability = result['result']['face_list'][0]['face_probability']
                if face_probability == 1:
                    result = searchMN(filepath)
                    user_group_id = result['result']['face_list'][0]['user_list'][0]['group_id']
                    user_id = result['result']['face_list'][0]['user_list'][0]['user_id']
                    user_info = result['result']['face_list'][0]['user_list'][0]['user_info']
                    user_score = result['result']['face_list'][0]['user_list'][0]['score']
                    if user_score != "":
                        if int(user_score) > 80:
                            print(
                                f"\n已查找到匹配度为:{user_score} 的用户,用户组: {user_group_id} 用户id {user_id} 用户信息:{user_info}\n")
                            # 保护进程，进行加锁操作
                            results_lock.acquire()
                            # 将参数放入队列中
                            results_queue.put((user_group_id, user_id, user_info, user_score))
                            # 解锁
                            results_lock.release()
            except TypeError as e:
                print(f"Error: {e}")
        time.sleep(2)


def run_face_search_ALL_in_thread(filepath):
    global is_face_thread_running
    # Start a new thread only if there are no other threads running
    # Check if a thread is already running
    if not is_face_thread_running:
        # Set the flag indicating that a thread is now running
        is_face_thread_running = True
        # Start the new thread
        thread = threading.Thread(target=face_search_ALL, args=(filepath,))
        thread.start()


def run_client_in_thread():
    global is_client_thread_running
    if not is_client_thread_running:
        is_client_thread_running = True
        thread = threading.Thread(target=loadTheOnenet, args=())
        thread.start()


def showTest(img_np_array_bgr, user_group_id, user_id, user_info, user_score):
    y0, dy = 30, 20  # y0 - initial y position, dy - distance between lines
    texts = [f'Group ID: {user_group_id}', f'User ID: {user_id}', f'Info: {user_info}',
             f'Score: {user_score}']
    for i, line in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(img_np_array_bgr, line, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255,
                                                                                     255), 1)


def showTestinit():
    global user_group_id
    global user_id
    global user_info
    global user_score
    user_group_id = "N/A"
    user_id = "N/A"
    user_info = "N/A"
    user_score = "N/A"


# 当客户端收到来自服务器的CONNACK响应时的回调。也就是申请连接，服务器返回结果是否成功等
def on_connect(client, userdata, flags, rc):
    print("连接结果:" + mqtt.connack_string(rc))


# 从服务器接收发布消息时的回调。
def on_message(client, userdata, msg):
    print(str(msg.payload, 'utf-8'))


# 当消息已经被发送给中间人，on_publish()回调将会被触发
def on_publish(client, userdata, mid):
    print(str(mid))


def http_put_data(data):
    url = "http://api.heclouds.com/devices/" + ClientId + '/datapoints'
    d = time.strftime('%Y-%m-%dT%H:%M:%S')

    values = {
        "datastreams": [{"id": "Monitor", "datapoints": [{"value": data}]}]}

    jdata = json.dumps(values).encode("utf-8")
    request = urllib.request.Request(url, jdata)
    request.add_header('api-key', APIKey)
    request.get_method = lambda: 'POST'
    request = urllib.request.urlopen(request)
    print("发送信息：%s" % DetectedStatus)
    return request.read()


def http_put_data_init():
    global DetectedStatus
    DetectedStatus = False
    url = "http://api.heclouds.com/devices/" + ClientId + '/datapoints'
    d = time.strftime('%Y-%m-%dT%H:%M:%S')

    values = {
        "datastreams": [{"id": "Monitor", "datapoints": [{"value": DetectedStatus}]}]}

    jdata = json.dumps(values).encode("utf-8")
    request = urllib.request.Request(url, jdata)
    request.add_header('api-key', APIKey)
    request.get_method = lambda: 'POST'
    request = urllib.request.urlopen(request)
    print("发送信息：%s" % DetectedStatus)
    return request.read()


def loadTheOnenet():
    client = mqtt.Client(ClientId, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_message = on_message
    client.username_pw_set(Username, passwd)
    client.connect('183.230.40.39', port=6002, keepalive=120)
    client.loop_forever()


class ShowCaptureVideoWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        # 用来显示画面的QLabel
        self.video_label = QLabel("选择顶部的操作按钮...")
        self.video_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)
        self.setLayout(layout)


class CaptureThread(QThread):
    def __init__(self, ip, port):
        super().__init__()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.udp_socket.bind((ip, port))
        # 设置运行的标志
        self.run_flag = True
        self.record_flag = False

    def run(self):
        global CAPTURE_IMAGE_DATA
        global user_group_id, user_id, user_info, user_score
        count = 0

        while self.run_flag:
            data, ip = self.udp_socket.recvfrom(100000)
            bytes_stream = io.BytesIO(data)
            image = Image.open(bytes_stream)
            img = np.asarray(image)
            img_np_array_bgr = np.array(image.convert('RGB'))[:, :, ::-1].copy()

            boxes = boxSelectFace(img_np_array_bgr)
            for (x, y, w, h) in boxes:
                cv2.rectangle(img_np_array_bgr, (x, y), (w, h), (255, 255, 255), thickness=2)

            # 保存每帧图片作为人脸识别的素材
            temp_file_path = "./images/cache.jpg"  # TODO: Replace with actual path
            cv2.imwrite(temp_file_path, img_np_array_bgr)
            # 执行人脸识别函数(多线程)
            run_face_search_ALL_in_thread(temp_file_path)

            # 检查是否有新的结果在队列中
            if not results_queue.empty():
                # DetectedStatus is True
                DetectedStatus = True
                # 获取队列中函数返回值
                result = results_queue.get()
                # 获取内容信息
                user_group_id, user_id, user_info, user_score = result
                # 在UI中设置各项参数
                showTest(img_np_array_bgr, user_group_id, user_id, user_info, user_score)
                # 获取到数据后4s刷新内容为"N/A"
                threading.Timer(4, showTestinit).start()
                # 向Onenet发送监控确认
                http_put_data(DetectedStatus)
                # 定时5s进行恢复
                threading.Timer(5, http_put_data_init).start()
            else:
                threading.Timer(5, http_put_data).start()

            showTest(img_np_array_bgr, user_group_id, user_id, user_info, user_score)
            if self.record_flag:  # 开始录制视频
                # 转换数据格式，便于存储视频文件
                img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
                self.mp4_file.write(img_2)

            # PySide显示不需要转换，所以直接用img
            img_pil_rgb = Image.fromarray(cv2.cvtColor(img_np_array_bgr, cv2.COLOR_BGR2RGB))
            temp_image = QImage(img_pil_rgb.tobytes(), img_pil_rgb.size[0], img_pil_rgb.size[1], QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            CAPTURE_IMAGE_DATA = temp_pixmap

            # temp_image = QImage(img.flatten(), 480, 320, QImage.Format_RGB888)
            # temp_pixmap = QPixmap.fromImage(temp_image)
            # CAPTURE_IMAGE_DATA = temp_pixmap  # 暂时存储udp接收到的1帧视频画面

        # 结束后 关闭套接字
        self.udp_socket.close()

    def stop_run(self):
        self.run_flag = False
        self.record_flag = False
        try:
            self.mp4_file.release()
        except Exception as ret:
            pass

    def stop_record(self):
        self.mp4_file.release()
        self.record_flag = False

    def start_record(self):
        # 设置视频的编码解码方式avi
        video_type = cv2.VideoWriter_fourcc(*'XVID')  # 视频存储的格式
        # 保存的位置，以及编码解码方式，帧率，视频帧大小
        file_name = "{}.avi".format(time.time())
        file_path_name = os.path.join(SAVE_VIDEO_PATH, file_name)
        self.mp4_file = cv2.VideoWriter(file_path_name, video_type, 5, (480, 320))
        self.record_flag = True


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("远程摄像头监控 v2022.10.13.001")
        self.setWindowIcon(QIcon('./images/logo.png'))
        self.resize(777, 555)

        # 选择本电脑IP
        camera_label = QLabel("选择本电脑IP：")

        # ip列表
        # 获取本地电脑的ip地址列表
        hostname, alias_list, ip_addr_list = socket.gethostbyname_ex(socket.gethostname())
        # print(hostname)  # DESKTOP
        # print(alias_list)  # []
        # print(ip_addr_list)  # ['192.168.239.49', '192.168.239.94', '192.168.31.53']
        ip_addr_list.insert(0, "127.0.0.1")
        self.combox = QComboBox()
        self.combox.addItems(ip_addr_list)
        self.ip_addr_list = ip_addr_list

        # 本地端口
        port_label = QLabel("本地端口：")
        self.port_edit = QLineEdit("9090")

        g_1 = QGroupBox("监听信息")
        g_1.setFixedHeight(60)
        g_1_h_layout = QHBoxLayout()
        g_1_h_layout.addWidget(camera_label)
        g_1_h_layout.addWidget(self.combox)
        g_1_h_layout.addWidget(port_label)
        g_1_h_layout.addWidget(self.port_edit)
        g_1.setLayout(g_1_h_layout)

        # 启动显示
        self.camera_open_close_btn = QPushButton(QIcon("./images/shexiangtou.png"), "启动显示")
        self.camera_open_close_btn.clicked.connect(self.camera_open_close)

        self.record_video_btn = QPushButton(QIcon("./images/record.png"), "开始录制")
        self.record_video_btn.clicked.connect(self.recorde_video)

        save_video_path_setting_btn = QPushButton(QIcon("./images/folder.png"), "设置保存路径")
        save_video_path_setting_btn.clicked.connect(self.save_video_path_setting)

        g_2 = QGroupBox("功能操作")
        g_2.setFixedHeight(60)
        g_2_h_layout = QHBoxLayout()
        g_2_h_layout.addWidget(self.camera_open_close_btn)
        g_2_h_layout.addWidget(self.record_video_btn)
        g_2_h_layout.addWidget(save_video_path_setting_btn)
        g_2.setLayout(g_2_h_layout)

        # --------- 整体布局 ---------
        h_layout = QHBoxLayout()
        h_layout.addWidget(g_1)
        h_layout.addWidget(g_2)
        h_layout.addStretch(1)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)

        # 创建底部的显示区域
        self.stacked_layout_capture_view = ShowCaptureVideoWidget()
        v_layout.addWidget(self.stacked_layout_capture_view)

        self.setLayout(v_layout)

        # 定时刷新视频画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_video_image)
        self.load_time = 0
        self.load_time_all = 0

    def camera_open_close(self):
        """启动创建socket线程，用来接收显示数据"""
        if self.camera_open_close_btn.text() == "启动显示":
            ip = self.combox.currentText()
            try:
                port = int(self.port_edit.text())
            except Exception as ret:
                QMessageBox.about(self, '警告', '端口设置错误！！！')
                return

            self.thread = CaptureThread(ip, port)
            self.thread.daemon = True
            self.thread.start()
            self.timer.start(100)  # 设置计时间隔并启动
            self.camera_open_close_btn.setText("关闭显示")
            executor.shutdown(wait=True)
        else:
            self.camera_open_close_btn.setText("启动显示")
            self.timer.stop()
            self.stacked_layout_capture_view.video_label.clear()
            self.thread.stop_run()
            self.record_video_btn.setText("开始录制")

    def show_video_image(self):
        if CAPTURE_IMAGE_DATA:
            self.stacked_layout_capture_view.video_label.setPixmap(CAPTURE_IMAGE_DATA)
        else:
            if time.time() - self.load_time >= 1:
                self.load_time = time.time()
                self.load_time_all += 1
                self.stacked_layout_capture_view.video_label.setText("摄像头加载中...{}".format(self.load_time_all))

    @staticmethod
    def get_desktop():
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
        return winreg.QueryValueEx(key, "Desktop")[0]

    def save_video_path_setting(self):
        """视频保存路径"""
        global SAVE_VIDEO_PATH
        if SAVE_VIDEO_PATH:
            last_path = QSettings().value("LastFilePath")
        else:
            last_path = self.get_desktop()

        path_name = QFileDialog.getExistingDirectory(self, '请选择保存视频的路径', last_path)
        if not path_name:
            return

        SAVE_VIDEO_PATH = path_name

    def recorde_video(self):
        """录制视频"""
        if self.camera_open_close_btn.text() == "启动显示":
            QMessageBox.about(self, '警告', '请先启动显示，然后再开始录制！！！')
            return

        if not SAVE_VIDEO_PATH:
            QMessageBox.about(self, '警告', '请先配置视频保存路径！！！')
            return

        if self.record_video_btn.text() == "开始录制":
            self.record_video_btn.setText("停止录制")
            self.thread.start_record()
        else:
            self.record_video_btn.setText("开始录制")
            self.thread.stop_record()


if __name__ == "__main__":
    run_client_in_thread()
    app = QApplication(sys.argv)
    video_window = VideoWindow()
    video_window.show()
    app.exec()
