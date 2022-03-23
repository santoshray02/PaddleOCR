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
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

class CharInfer():
    def __init__(self, config_file=r'C:\Users\santosh\projects\paperentry\PaddleOCR\rec_weights\config.yml', weight_dir=r'C:\Users\santosh\projects\paperentry\PaddleOCR\rec_weights\best_accuracy'):
    
        self.config, device, self.logger, vdl_writer = program.preprocess(config_file=config_file)
        global_config = self.config['Global']
        self.config['Global']['use_gpu'] = False
        # self.config['Global']['save_res_path']= 

        self.config["Global"]["pretrained_model"]=weight_dir
        # build post process
        self.post_process_class = build_post_process(self.config['PostProcess'],
                                                global_config)
        # build model
        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config['Architecture']["algorithm"] in ["Distillation",
                                                    ]:  # distillation model
                for key in self.config['Architecture']["Models"]:
                    self.config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
            else:  # base rec model
                self.config['Architecture']["Head"]['out_channels'] = char_num

        self.model = build_model(self.config['Architecture'])

        load_model(self.config, self.model)

        # create data ops
        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if self.config['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif self.config['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                else:
                    op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        global_config['infer_mode'] = True
        self.ops = create_operators(transforms, global_config)

        # save_res_path = self.config['Global'].get('save_res_path',
        #                                     "./output/rec/predicts_rec.txt")
        # if not os.path.exists(os.path.dirname(save_res_path)):
        #     os.makedirs(os.path.dirname(save_res_path))

        self.model.eval()
    def run_rec(self, image):
        results = []
        if 1==1:
        # with open(save_res_path, "w") as fout:
            file = ""
            for file in get_image_file_list(image):
                # self.logger.info("infer_img: {}".format(file))
                with open(file, 'rb') as f:
                    img = f.read()
                    data = {'image': img}
            # for img_index, img in enumerate(self.config['Global']['infer_imgs']):
            #     data = {'image': img}
                batch = transform(data, self.ops)
                if self.config['Architecture']['algorithm'] == "SRN":
                    encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                    gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                    gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                    gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                    others = [
                        paddle.to_tensor(encoder_word_pos_list),
                        paddle.to_tensor(gsrm_word_pos_list),
                        paddle.to_tensor(gsrm_slf_attn_bias1_list),
                        paddle.to_tensor(gsrm_slf_attn_bias2_list)
                    ]
                if self.config['Architecture']['algorithm'] == "SAR":
                    valid_ratio = np.expand_dims(batch[-1], axis=0)
                    img_metas = [paddle.to_tensor(valid_ratio)]

                images = np.expand_dims(batch[0], axis=0)
                images = paddle.to_tensor(images)
                if self.config['Architecture']['algorithm'] == "SRN":
                    preds = self.model(images, others)
                elif self.config['Architecture']['algorithm'] == "SAR":
                    preds = self.model(images, img_metas)
                else:
                    preds = self.model(images)
                post_result = self.post_process_class(preds)
                info = None
                if isinstance(post_result, dict):
                    rec_info = dict()
                    for key in post_result:
                        if len(post_result[key][0]) >= 2:
                            rec_info[key] = {
                                "label": post_result[key][0][0],
                                "score": float(post_result[key][0][1]),
                            }
                    info = json.dumps(rec_info)
                else:
                    if len(post_result[0]) >= 2:
                        info = post_result[0][0] + "\t" + str(post_result[0][1])
                        results.append(post_result[0])

                # if info is not None:
                #     self.logger.info("\t result: {}".format(info))
                    # fout.write(file + "\t" + info)
        return results[0]
    # self.logger.info("success!")

# if __name__ == '__main__':
#     config, device, self.logger, vdl_writer = program.preprocess()
#     main()
