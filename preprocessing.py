import os
from datasets import load_dataset


def coco_to_yolo(x, y, w, h, width, height):
    return [((2*x + w)/(2*width)), ((2*y + h)/(2*height)), w/width, h/height]


def preprocessing(partition: str, data: object):
    os.makedirs(f"datasets/images/{partition}", exist_ok=True)
    os.makedirs(f"datasets/labels/{partition}", exist_ok=True)

    for i, sample in enumerate(data[partition]):
        img     = sample["image"]
        labels  = sample["objects"]["category"]
        bboxes  = sample["objects"]["bbox"]
        width   = sample["width"]
        height  = sample["height"]
        items   = []
        
        for label, box in zip(labels, bboxes):
            xc, yc, w, h = coco_to_yolo(box[0], box[1], box[2], box[3], width, height)
            items.append(f"{label} {xc} {yc} {w} {h}")

        with open(f"datasets/labels/{partition}/{i}.txt", "w") as f:
            for item in items:
                f.write(f"{item}\n")

        img.save(f"datasets/images/{partition}/{i}.png")

if __name__ == "__main__":
    dataset = load_dataset("nflechas/recycling_app", name="full")
    preprocessing("train", dataset)
    preprocessing("test", dataset)
    preprocessing("validation", dataset)