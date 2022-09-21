from pathlib import Path
import json
import os
from tqdm import tqdm
import cv2

# from SwigCaptions import SWiGCaptions
from SwigCaptionsV2 import SwigCaptionV2

def save_captions_to_json(captions:dict, export_file:Path):
    a = {}
    if not os.path.isfile(export_file):
        for k in captions.keys():
            a[k] = captions[k]
        with open(export_file, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(export_file) as feedsjson:
            feeds = json.load(feedsjson)

        for k in captions.keys():
            feeds[k] = captions[k]
        with open(export_file, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def save_log(generator:SwigCaptionV2, current_batch:int, logfile: Path):
    log = {}

    if not os.path.isfile(logfile):
        log["target"] = str(generator.target_file)
        log["n_keys"] = generator.n_items
        log["total_batch"] = generator.total_batch
        log["completed_batch"] = current_batch
    else:
        with open(logfile) as feedsjson:
            log = json.load(feedsjson)
        log["target"] = str(generator.target_file)
        log["n_keys"] = generator.n_items
        log["total_batch"] = generator.total_batch
        log["completed_batch"] = current_batch

    with open(logfile, mode='w') as f:
        f.write(json.dumps(log, indent=2))

def get_file_stats(rootpath="SWiG", annotation_file='dev.json'):

    root = Path(rootpath)
    annotationfile = root / 'SWiG_jsons' / annotation_file

    with open(annotationfile) as f:
        all_json = json.load(f)

    count = dict()

    for item_key in all_json.keys():
        item = all_json[item_key]
        keys = item['bb'].keys()
        n_keys = len(keys)
        if n_keys in count:
            count[n_keys] += 1
        else:
            count[n_keys] = 1

    print(count)

def debug_batch_image(captions:dict, rootpath="SWiG", annotation_file='dev.json'):

    root = Path(rootpath)

    imgdir = root / 'images_512'
    annotationfile = root / 'SWiG_jsons' / annotation_file

    with open(annotationfile) as f:
        all = json.load(f)

    for key in captions:
        img_path = imgdir / key

        annotation = all[key]
        print(annotation)

        # read image
        img = cv2.imread(str(img_path))

        result = img.copy()
        boxes = annotation['bb']
        for bkey in boxes:
            b = boxes[bkey]
            cv2.rectangle(result, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        print(captions[key])

        # show thresh and result
        cv2.imshow("bounding_box", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_generation_on_file(targettype, rootpath="SWiG", exportpath="generated/v2"):
    target_to_file = {
        "validation": "dev.json",
        "test" : "test.json",
        "train": "train.json"
    }

    targetfile = target_to_file[targettype]
    root = Path(rootpath)
    validation_file = root / "SWiG_jsons" / targetfile

    exports = Path(exportpath)
    export_file = exports / targetfile
    log_file = exports / "log.json"

    # ensure path
    if not os.path.exists(exports):
        os.makedirs(exports)

    capgen = SwigCaptionV2(validation_file, batch_size=4)

    total_item = 0
    total_skipped = 0

    total_batch = capgen.total_batch
    for i in tqdm(range(0, total_batch + 1)):
    # for i in range(5, 6):
        captions = capgen.read_and_generate_batch(i)

        total_item += capgen.batch_size
        # total_skipped += skipped

        # debug_batch_image(captions)

        save_captions_to_json(captions, export_file)
        save_log(capgen, i, log_file)

    print(f'Total 0 skipped out of {total_item}')


if __name__ == '__main__':
    run_generation_on_file(
        "validation",
        rootpath="SWiG",
        exportpath="generated/v2",
    )
    # get_file_stats()

