from pathlib import Path
from SwigCaptions import SWiGCaptions


if __name__ == '__main__':
    root = Path("SWiG")

    validation_file = root / "SWiG_jsons" / "dev.json"

    capgen = SWiGCaptions()
    captions = capgen.read_and_generate_batch(validation_file, 1)
    print(captions)
