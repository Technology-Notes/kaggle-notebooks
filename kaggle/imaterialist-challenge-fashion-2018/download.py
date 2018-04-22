import json
import os
from urllib import request
from PIL import Image
import multiprocessing
from tqdm import tqdm
from functools import partial
from io import BytesIO


PATH = '/output/imaterialist-challenge-fashion-2018'


def download_image(img_data, prefix, path):
    filename = os.path.join(path, prefix, "{}.jpg".format(img_data['imageId']))

    if os.path.exists(filename):
        return 0

    try:
        response = request.urlopen(img_data['url'])
        image_data = response.read()
        pil_image = Image.open(BytesIO(image_data))
        pil_image_rgb = pil_image.convert('RGB')
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except Exception:
        return 1

    return 0


def download_set(set_name):

    print('Downloading {} set.'.format(set_name))

    os.makedirs('{}/{}'.format(PATH, set_name), exist_ok=True)

    set_json = json.load(open('{}/{}.json'.format(PATH, set_name)))

    pool = multiprocessing.Pool(processes=20)

    _download_image = partial(download_image, prefix=set_name, path=PATH)

    failures = sum(
        tqdm(pool.imap_unordered(
            _download_image,
            set_json['images']
        ), total=len(set_json['images'])))

    print('Total number of {} download failures: {}'.format(set_name, failures))

    pool.close()
    pool.terminate()


download_set('test')
download_set('validation')
download_set('train')
