from pathlib import Path
from SwigCaptions import SWiGCaptions
import json
from json.decoder import JSONDecodeError
import os


if __name__ == '__main__':
    root = Path("SWiG")
    validation_file = root / "SWiG_jsons" / "dev.json"

    exports = Path("generated")
    export_file = exports / "dev.json"

    # with open(f'{root}/SWiG_jsons/imsitu_space.json') as f:
    #     all = json.load(f)
    #     verb_orders = all['verbs']
    #
    #     print(verb_orders)

    # ensure path
    if not os.path.exists(exports):
        os.makedirs(exports)

    capgen = SWiGCaptions()

    total_item = 0
    total_skipped = 0

    total_batch = 4
    for i in range(1, total_batch):
        captions, skipped = capgen.read_and_generate_batch(validation_file, i)

        total_item += capgen.batch_size
        total_skipped += skipped

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

        print(f'Total {i} batches completed out of {total_batch}')

    print(f'Total {total_skipped} skipped out of {total_item}')

