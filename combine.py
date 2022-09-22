from pathlib import Path
import json
import os
from typing import List

from SwigCaptionsV2 import show_img

def get_max_len_caption(captions:List[str], fallback=''):
    if not captions:
        return fallback
    max_str = captions[0]
    for x in captions:
        if len(x) > len(max_str):
            max_str = x
    return max_str

def combine_data(dataset, swigdir="SWiG", generateddir="generated/v2"):
    dataset_to_file = {
        "validation": "dev.json",
        "test": "test.json",
        "train": "train.json"
    }

    target = dataset_to_file[dataset]
    swig = Path(swigdir)
    generated = Path(generateddir)


    swig_target = swig / "SWiG_jsons" / target
    generated_target = generated / target
    export_path = swig / "combined_jsons"
    export_target = swig / "combined_jsons" / target

    if not os.path.exists(swig_target):
        raise ValueError("SWiG target file does not exists")

    if not os.path.exists(generated_target):
        raise ValueError("Generated target file does not exists")

    # ensure path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Opening JSON file
    swig_file = open(swig_target)
    generated_file = open(generated_target)

    swig_json = json.load(swig_file)
    generated_json = json.load(generated_file)

    combined = {}

    total = 0
    updated = 0
    for key in swig_json.keys():
        temp = swig_json[key]
        try:
            captions = generated_json[key]
        except KeyError:
            total += 1
            continue

        temp['captions'] = captions
        temp['caption'] = get_max_len_caption(captions)
        combined[key] = temp
        updated += 1
        total += 1

    swig_file.close()
    generated_file.close()

    print(f'Total {updated} items updated out of {total}')

    with open(export_target, mode='w') as f:
        f.write(json.dumps(combined))

    print(f'Combined dataset saved to file {str(export_target)}')

def check_max_len(dataset, generateddir="generated/v2"):
    dataset_to_file = {
        "validation": "dev.json",
        "test": "test.json",
        "train": "train.json"
    }

    target = dataset_to_file[dataset]
    generated = Path(generateddir)

    generated_target = generated / target
    if not os.path.exists(generated_target):
        raise ValueError("Generated target file does not exists")

    generated_file = open(generated_target)
    generated_json = json.load(generated_file)

    max_len = 0
    for key in generated_json.keys():
        captions = generated_json[key]
        max_len_caption = get_max_len_caption(captions)
        caption_len = len(max_len_caption.split())
        if caption_len > max_len:
            max_len = caption_len
    generated_file.close()

    print("=============================")
    print(max_len)
    print("=============================")


def debug_dataset(dataset, generateddir="SWiG/combined_jsons", img_dir="SWiG/images_512"):
    dataset_to_file = {
        "validation": "dev.json",
        "test": "test.json",
        "train": "train.json"
    }

    target = dataset_to_file[dataset]
    generated = Path(generateddir)
    imgpath = Path(img_dir)
    targetfile = generated / target

    generated_file = open(targetfile)
    alljson = json.load(generated_file)

    count = 0
    for key in alljson.keys():
        annotation = alljson[key]
        targetimg = imgpath / key

        bboxes = annotation['bb']

        if len(bboxes.keys()) == 3:
            print(annotation)
            show_img(targetimg, annotation)

            count += 1
            if count >= 5:
                break


if __name__ == '__main__':
    # combine_data("validation")
    # check_max_len("train")
    debug_dataset('validation')