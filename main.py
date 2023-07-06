from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget,QMessageBox
from main_ui import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap,QPainter
import sys
import time
import platform
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = 'weights/best.pt'
        self.source = ''
        self.conf_thres = 0.25
        self.statistic_dic = {}

    @smart_inference_mode()
    def run(
            self,
            weights=ROOT / 'weights/best.pt',  # model path or triton URL
            source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/myvoc.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        conf_thres = self.conf_thres
        source = self.source
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights='weights/best.pt', device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 将预测信息映射到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                     # 打印检测到的类别数量
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        self.statistic_dic[names[int(c)]] = int(f"{n}")

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    try:
                        cv2.putText(im0,f"{n} {names[int(c)]}{'s' * (n > 1)}", (5,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    except:
                        pass
                    #cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
            time.sleep(1/40)
            # print(type(im0s))
            
            self.send_img.emit(im0)
            self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
            self.send_statistic.emit(self.statistic_dic)
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.model = 'weights/best.pt'
        self.det_thread = DetThread()
        self.is_showed = 0

        self.det_thread.source = '0'
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_result))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_raw))
        self.det_thread.send_statistic.connect(self.show_statistic)
        # self.RunProgram.triggered.connect(lambda: self.det_thread.start())
        self.RunProgram.triggered.connect(self.term_or_con)
        self.SelFile.triggered.connect(self.open_file)
        self.SelModel.triggered.connect(self.open_model)
        self.status_bar_init()
        self.cam_switch.triggered.connect(self.camera)
        self.horizontalSlider.valueChanged.connect(lambda: self.conf_change(self.horizontalSlider))
        self.spinBox.valueChanged.connect(lambda: self.conf_change(self.spinBox))

    # 更改置信度
    def conf_change(self, method):
        if method == self.horizontalSlider:
            self.spinBox.setValue(self.horizontalSlider.value())
        if method == self.spinBox:
            self.horizontalSlider.setValue(self.spinBox.value())
        self.det_thread.conf_thres = self.horizontalSlider.value()/100
        self.statusbar.showMessage("置信度已更改为："+str(self.det_thread.conf_thres))
    
    def status_bar_init(self):
        self.statusbar.showMessage('界面已准备')
    
    def open_file(self):
        source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                           "*.jpg *.png)")
        if source[0]:
            self.det_thread.source = source[0]
        print(self.det_thread.source)
        self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.det_thread.source)
                                                    if os.path.basename(self.det_thread.source) != '0'
                                                    else '摄像头设备'))
    
    def term_or_con(self):
        if self.RunProgram.isChecked():
            print(self.det_thread.source)
            self.det_thread.start()
            self.statusbar.showMessage('正在检测 >> 模型：{}，文件：{}'.
                                       format(os.path.basename(self.det_thread.weights),
                                              os.path.basename(self.det_thread.source)
                                                               if os.path.basename(self.det_thread.source) != '0'
                                                               else '摄像头设备'))
        else:
            self.det_thread.terminate()
            if hasattr(self.det_thread, 'vid_cap'):
                if self.det_thread.vid_cap:
                    self.det_thread.vid_cap.release()
            self.is_showed = 0
            self.statusbar.showMessage('结束检测')
        
    def open_model(self):
        self.model = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.pt)")[0]
        if self.model:
            self.det_thread.weights = self.model
        self.statusbar.showMessage('加载模型：' + os.path.basename(self.det_thread.weights))

    def camera(self):
        if self.cam_switch.isChecked():
            self.det_thread.source = '0'
            self.statusbar.showMessage('摄像头已打开')
        else:
            self.det_thread.terminate()
            if hasattr(self.det_thread, 'vid_cap'):
                self.det_thread.vid_cap.release()
            if self.RunProgram.isChecked():
                self.RunProgram.setChecked(False)
            self.statusbar.showMessage('摄像头已关闭')
    
    def show_statistic(self, statistic_dic):
        try:
            names = []
            self.listWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [str(i[0]) + ',' + str(i[1]) for i in statistic_dic]
            self.listWidget.addItems(results)
            for i in statistic_dic:
                names.append(i[0])
            if 'e_vehicle' in names and self.is_showed  == 0:
                try:
                    self.is_showed = QMessageBox()
                    self.is_showed.setWindowTitle('警告')
                    self.is_showed.setText('检测到电动车')
                    self.is_showed.setIcon(QMessageBox.Warning)
                    self.is_showed.setStandardButtons(QMessageBox.Ok)
                    self.is_showed.button(QMessageBox.Ok).animateClick(3*1000)
                    self.is_showed.exec_()
                except Exception as e:
                    print(repr(e))

        except Exception as e:
            print(repr(e))
    
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())