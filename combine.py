from pathlib import Path
import json
import os

def combine_data(dataset, swigdir="SWiG", generateddir="generated"):
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
            temp['captions'] = generated_json[key]
            combined[key] = temp
            updated += 1
            total += 1
        except KeyError:
            total += 1
            continue

    swig_file.close()
    generated_file.close()

    print(f'Total {updated} items updated out of {total}')

    with open(export_target, mode='w') as f:
        f.write(json.dumps(combined))

    print(f'Combined dataset saved to file {str(export_target)}')

if __name__ == '__main__':
    combine_data("validation")