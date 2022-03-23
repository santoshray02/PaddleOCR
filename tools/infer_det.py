# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
import tools.infer_rec as infer_rec


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        dt_boxes = sorted(dt_boxes, key=lambda x: (x[0][0], x[0][1]))
        import cv2
        src_im = img.copy()
        result = ""
        char_infer = infer_rec.CharInfer()
        for i, box in enumerate(dt_boxes):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            a, b, c, d = np.squeeze(box).astype(np.int32).tolist()
            xmin = min(a[0], b[0], c[0], d[0])
            xmax = max(a[0], b[0], c[0], d[0])
            ymin = min(a[1], b[1], c[1], d[1])
            ymax = max(a[1], b[1], c[1], d[1])
            crop_img = img[ymin:ymax, xmin:xmax]
            crop_path = save_path+ f"crop_{i:>02d}.png"
            # logger.info(crop_path)
            cv2.imwrite(crop_path, crop_img)
            # print(save_path)
            res = char_infer.run_rec(crop_path)
            result += res[0]
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            font = cv2.FONT_HERSHEY_COMPLEX
            # {round(float(res[1]), 1)
            cv2.putText(src_im, f'{res[0]}',(xmin,ymin-10), font, 0.5, (0,255,100),1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {} {}".format(save_path, result))

@paddle.no_grad()
def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            # logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = box.tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(save_res_path) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(save_res_path) + "/det_results/"
                os.makedirs(save_det_path, exist_ok=True)
                
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

            # save_det_path = os.path.dirname(config['Global'][
            #     'save_res_path']) + "/det_results/"
            # draw_det_res(boxes, config, src_img, file, save_det_path)
    logger.info("success!")


if __name__ == '__main__':
    config_file = r"C:\Users\santosh\projects\paperentry\PaddleOCR\det_weights\config.yml"
    config, device, logger, vdl_writer = program.preprocess(config_file=config_file)
    weight_dir = r"C:\Users\santosh\projects\paperentry\PaddleOCR\det_weights\best_accuracy"
    input_path= r"G:\My Drive\Engineering\23 DataLabeling\3.Staging\Susmitha\micr_imgs"
    if os.path.isfile(input_path):
        out_path = os.path.splitext(input_path)[0] +"/pred.txt"
    else:
        out_path = os.path.dirname(input_path) + "/results/pred.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    config["Global"]["pretrained_model"]=weight_dir
    config["Global"]["save_res_path"]=out_path
    config['Global']['use_gpu'] = False
    config['Global']['infer_img'] = input_path
    main()
