import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import uuid
import os
from ocr import get_ocr_result

import cv2 as cv
from matplotlib import pyplot as plt

app = Flask(__name__, 
            static_url_path='',
            static_folder='./static')
CORS(app)

# used for serialize numpy data in json
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/v1/cv/ocr-roi/pattern', methods=["POST"])
def generate_pattern():
    if 'image' not in request.files:
        return 'no image field in form-data', 400
    file = request.files['image']
    if file.filename == '':
        return 'no file in image field', 400
    
    regions_data = request.form.get('regions')
    if regions_data == None:
        return 'no regions field', 400
    
    regions = json.loads(regions_data)
    print('regions', regions)
    markers = regions['markers']
    ocrs = regions['ocrs']
    
    image_data = np.asarray(bytearray(file.read()), dtype="uint8")
    image = cv.imdecode(image_data, cv.IMREAD_COLOR)

    target = np.zeros(image.shape, np.uint8)
    for region in markers:
        print('region', region)
        ox, oy, dx, dy = int(region['left']), int(region['top']), int(region['left'] + region['width']), int(region['top'] + region['height'])
        print('position', region, ox, oy, dx, dy)
        target[oy:dy, ox:dx] = image[oy:dy, ox:dx]

    # print('markers_data', markers_data)
    # print('cor regions', ocrs)
    _, markers_data = cv.imencode('.png', target)

    pattern_id = str(uuid.uuid4())
    os.makedirs('patterns/' + pattern_id)
    pattern_image_fd = os.open('./patterns/' + pattern_id + '/pattern.png', os.O_CREAT | os.O_BINARY | os.O_RDWR)
    os.write(pattern_image_fd, markers_data.tobytes())
    os.close(pattern_image_fd)
    pattern_region_fd = os.open('./patterns/' + pattern_id + '/regions.json', os.O_CREAT | os.O_RDWR)
    os.write(pattern_region_fd, regions_data.encode())
    os.close(pattern_region_fd)

    return jsonify({'ok': True, 'patternId': pattern_id })

@app.route('/api/v1/cv/ocr-roi/recognition', methods=["POST"])
def generate_recognition():
    if 'image' not in request.files:
        return 'no image field in form-data', 400
    file = request.files['image']
    if file.filename == '':
        return 'no file in image field', 400
    
    configs_data = request.form.get('configs')
    if configs_data == None or configs_data == '':
        return 'cannot get configs', 400

    configs = json.loads(configs_data)
    if 'patternId' not in configs or configs['patternId'] is None or configs['patternId'] is '':
        return 'cannot get pattern id from configs', 400
    pattern_id = configs['patternId']
    pre_processes = []
    if 'preProcesses' in configs:
        pre_processes = configs['preProcesses']
        
    # get image
    image_data = np.asarray(bytearray(file.read()), dtype="uint8")
    image = cv.imdecode(image_data, cv.IMREAD_COLOR)
    # cv.imshow('original', image),cv.waitKey(),cv.destroyAllWindows()
    
    pattern_image = cv.imread('./patterns/' + pattern_id + '/pattern.png')
    regions = {}
    # print('sizes', image.shape, pattern.shape)
    with open('./patterns/' + pattern_id + '/regions.json') as f:
        regions = json.load(f)

    # do pre-processes
    for process in pre_processes:
        if process == 'invert':
            image = 255 - image
            pattern_image = 255 - pattern_image
        else:
            return 'invalid process command', 400

    res = app.response_class(response=json.dumps(get_ocr_result(image, pattern_image, regions), cls=MyEncoder),
        status=200,
        mimetype='application/json')
    return res
