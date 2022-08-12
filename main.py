from pathlib import Path
import json
import os
from tqdm import tqdm

from SwigCaptions import SWiGCaptions

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

def save_log(generator:SWiGCaptions, current_batch:int, logfile: Path):
    log = {}

    if not os.path.isfile(logfile):
        log["target"] = str(generator.targetfile)
        log["n_keys"] = generator.n_items
        log["total_batch"] = generator.total_batch
        log["completed_batch"] = current_batch
    else:
        with open(logfile) as feedsjson:
            log = json.load(feedsjson)
        log["target"] = str(generator.targetfile)
        log["n_keys"] = generator.n_items
        log["total_batch"] = generator.total_batch
        log["completed_batch"] = current_batch

    with open(logfile, mode='w') as f:
        f.write(json.dumps(log, indent=2))

def run_generation_on_file(targettype, rootpath="SWiG", exportpath="generated"):
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

    capgen = SWiGCaptions(validation_file, batch_size=8)

    total_item = 0
    total_skipped = 0

    total_batch = capgen.total_batch
    for i in tqdm(range(1, total_batch + 1)):
        captions, skipped = capgen.read_and_generate_batch(i)

        total_item += capgen.batch_size
        total_skipped += skipped

        save_captions_to_json(captions, export_file)
        save_log(capgen, i, log_file)

    print(f'Total {total_skipped} skipped out of {total_item}')


if __name__ == '__main__':
    run_generation_on_file(
        "validation",
        rootpath="SWiG",
        exportpath="generated",
    )

