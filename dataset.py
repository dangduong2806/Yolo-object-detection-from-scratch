import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def iou_width_height(box_wh, anchors_wh, eps=1e-6):
    """
    Tính IoU giữa width-height của ground truth box và anchors.

    box_wh: Tensor shape [2]
        [w, h]

    anchors_wh: Tensor shape [9, 2]
        [[w1, h1], [w2, h2], ...]

    Tất cả w, h nên cùng đơn vị, ví dụ đều normalized [0, 1].
    """
    box_w, box_h = box_wh[0], box_wh[1]

    anchor_w = anchors_wh[:, 0]
    anchor_h = anchors_wh[:, 1]

    intersection = torch.min(box_w, anchor_w) * torch.min(box_h, anchor_h)

    box_area = box_w * box_h
    anchor_area = anchor_w * anchor_h

    union = box_area + anchor_area - intersection + eps

    return intersection / union

class YOLODataset(Dataset):
    def __init__(
            self, 
            json_path,
            root_dir,
            anchors,
            image_size=416,
            scales=(13,26,52),
            ignore_iou_thresh=0.5,
            use_letterbox=True
    ):
        """
        json_path:
            Đường dẫn tới train.json hoặc val.json.

        root_dir:
            Thư mục gốc chứa train/images hoặc val/images.

            Ví dụ:
            root_dir = "/kaggle/input/my-dataset"

            Khi đó file_name trong json:
            train/images/img_xxx.jpg

            sẽ được nối thành:
            /kaggle/input/my-dataset/train/images/img_xxx.jpg

        anchors:
            Tensor/list shape [3, 3, 2]

            3 scale × 3 anchors × [w, h]

            Có thể truyền anchors theo pixel trên ảnh 416×416.
            Code sẽ tự normalize về [0, 1].

        image_size:
            Kích thước input của YOLOv3, thường là 416.

        scales:
            3 grid sizes tương ứng 3 detection heads.

        use_letterbox:
            True: resize giữ tỉ lệ + padding.
            False: resize thẳng về 416×416, có thể làm méo ảnh.
        """
        super().__init__()

        self.json_path = json_path
        self.root_dir = root_dir
        self.image_size = image_size
        self.scales = scales
        self.ignore_iou_thresh = ignore_iou_thresh
        self.use_letterbox = use_letterbox

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.classes = data["classes"]
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        self.images = data["images"]

        self.annotations_by_image = defaultdict(list)
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            self.annotations_by_image[image_id].append(ann)

        anchors = torch.tensor(anchors, dtype=torch.float32)

        if anchors.ndim != 3 or anchors.shape != (3, 3, 2):
            raise ValueError(
                "anchors phải có shape [3, 3, 2]: "
                "3 scales × 3 anchors × [w, h]"
            )

        # Nếu anchors đang ở đơn vị pixel, normalize về [0, 1]
        if anchors.max() > 1.0:
            anchors = anchors / image_size
        
        self.anchors = anchors
        self.all_anchors = anchors.reshape(-1, 2)
    
    def __len__(self):
        return len(self.images)
    
    def _resolve_image_path(self, file_name):
        image_path = os.path.join(self.root_dir, file_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Không tìm thấy ảnh: {image_path}\n"
                f"Hãy kiểm tra root_dir. file_name trong json là: {file_name}"
            )
        
        return image_path
    
    # Hàm này làm gì ?
    def _clip_box(self, x1, y1, x2, y2, width, height):
        x1 = max(0.0, min(float(x1), float(width)))
        y1 = max(0.0, min(float(y1), float(height)))
        x2 = max(0.0, min(float(x2), float(width)))
        y2 = max(0.0, min(float(y2), float(height)))

        return x1, y1, x2, y2
    
    def _letterbox(self, image, boxes_xyxy_cls):
        """
        Resize giữ tỉ lệ ảnh và padding về image_size × image_size.

        boxes_xyxy_cls:
            list các box dạng:
            [x1, y1, x2, y2, class_id]

        Trả về:
            image đã letterbox
            boxes normalized dạng:
            [class_id, x_center, y_center, width, height]
        """
        original_w, original_h = image.size

        scale = min(
            self.image_size / original_w,
            self.image_size / original_h
        )

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized_image = image.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=(128, 128, 128)
        )

        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2

        canvas.paste(resized_image, (pad_x, pad_y))

        boxes_yolo = []

        for x1, y1, x2, y2, class_id in boxes_xyxy_cls:
            x1_new = x1 * scale + pad_x
            y1_new = y1 * scale + pad_y
            x2_new = x2 * scale + pad_x
            y2_new = y2 * scale + pad_y

            x1_new = max(0.0, min(x1_new, self.image_size))
            y1_new = max(0.0, min(y1_new, self.image_size))
            x2_new = max(0.0, min(x2_new, self.image_size))
            y2_new = max(0.0, min(y2_new, self.image_size))

            box_w = x2_new - x1_new
            box_h = y2_new - y1_new

            if box_w <= 0 or box_h <= 0:
                continue

            x_center = (x1_new + x2_new) / 2.0
            y_center = (y1_new + y2_new) / 2.0

            x_center /= self.image_size
            y_center /= self.image_size
            box_w /= self.image_size
            box_h /= self.image_size

            boxes_yolo.append(
                [class_id, x_center, y_center, box_w, box_h]
            ) # đã được normalize
        
        return canvas, boxes_yolo
    
    def _resize_direct(self, image, boxes_xyxy_cls):
        """
        Resize trực tiếp về 416×416.
        Cách này đơn giản hơn nhưng làm méo ảnh nếu ảnh không vuông.
        """
        original_w, original_h = image.size

        image = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        boxes_yolo = []

        for x1, y1, x2, y2, class_id in boxes_xyxy_cls:
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            x_center = (x1 + x2) / 2.0 / original_w
            y_center = (y1 + y2) / 2.0 / original_h
            box_w = box_w / original_w
            box_h = box_h / original_h

            boxes_yolo.append(
                [class_id, x_center, y_center, box_w, box_h]
            )

        return image, boxes_yolo
    
    def _create_targets(self, boxes_yolo):
        """
        boxes_yolo:
            list các box dạng:
            [class_id, x_center, y_center, width, height]

            Tất cả x, y, w, h đã normalized theo ảnh 416×416 sau transform.

        targets:
            [
                [3, 13, 13, 6],
                [3, 26, 26, 6],
                [3, 52, 52, 6],
            ]
        """
        targets = [
            torch.zeros((3, S, S, 6), dtype=torch.float32)
            for S in self.scales
        ]

        if len(boxes_yolo) == 0:
            return tuple(targets)
        
        positive_slots = set()

        # Pass 1: assign exactly one positive anchor per GT if possible.
        for gt_idx, box in enumerate(boxes_yolo):
            class_id, x, y, w, h = box

            class_id = int(class_id)

            if w <= 0 or h <= 0:
                continue

            box_wh = torch.tensor([w, h], dtype=torch.float32)

            iou_anchors = iou_width_height(
                box_wh=box_wh,
                anchors_wh=self.all_anchors
            )

            anchor_indices = iou_anchors.argsort(descending=True)

            assigned = False

            for anchor_idx in anchor_indices:
                anchor_idx = int(anchor_idx)

                scale_idx = anchor_idx // 3
                anchor_on_scale = anchor_idx % 3

                S = self.scales[scale_idx]

                i = int(S * y)
                j = int(S * x)

                # Tránh trường hợp x hoặc y đúng bằng 1.0
                i = min(S - 1, max(0, i))
                j = min(S - 1, max(0, j))

                slot = (scale_idx, anchor_on_scale, i, j)

                if slot in positive_slots:
                    continue
                    
                if targets[scale_idx][anchor_on_scale, i, j, 0] != 0:
                    continue

                
                # target[..., 0] = 1 # objectness
                targets[scale_idx][
                    anchor_on_scale, i, j, 0
                ] = 1.0

                x_cell = S * x - j
                y_cell = S * y - i
                w_cell = w * S
                h_cell = h * S

                targets[scale_idx][
                    anchor_on_scale, i, j, 1:5
                ] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell],
                    dtype=torch.float32
                )

                targets[scale_idx][
                    anchor_on_scale, i, j, 5
                ] = class_id

                positive_slots.add(slot)
                assigned = True
                break
        
        # Pass 2: mark high-IoU non-positive anchors as ignore/free.
        for box in boxes_yolo:
            class_id, x, y, w, h = box

            if w <= 0 or h <= 0:
                continue

            box_wh = torch.tensor([w, h], dtype=torch.float32)

            iou_anchors = iou_width_height(
                box_wh=box_wh,
                anchors_wh=self.all_anchors,
            )

            for anchor_idx in range(len(self.all_anchors)):
                if iou_anchors[anchor_idx] <= self.ignore_iou_thresh:
                    continue

                scale_idx = anchor_idx // 3
                anchor_on_scale = anchor_idx % 3

                S = self.scales[scale_idx]

                i = int(S * y)
                j = int(S * x)

                i = min(S - 1, max(0, i))
                j = min(S - 1, max(0, j))

                current_obj = targets[scale_idx][anchor_on_scale, i, j, 0]

                if current_obj == 0:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1.0

        return tuple(targets)
    
    def __getitem__(self, index):
        image_info = self.images[index]

        image_id = image_info["id"]
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        image_path = self._resolve_image_path(file_name=file_name)

        # Load ảnh
        image = Image.open(image_path).convert("RGB")

        anns = self.annotations_by_image.get(image_id, [])

        boxes_xyxy_cls = []

        for ann in anns:
            class_name = ann["class"]
            bbox = ann["bbox"]

            if class_name not in self.class_to_idx:
                continue
            
            class_id = self.class_to_idx[class_name]

            x1, y1, x2, y2 = bbox

            x1, y1, x2, y2 = self._clip_box(
                x1, y1, x2, y2, width, height
            )

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xyxy_cls.append(
                [x1, y1, x2, y2, class_id]
            )

        if self.use_letterbox:
            image, boxes_yolo = self._letterbox(
                image=image,
                boxes_xyxy_cls=boxes_xyxy_cls
            )
        else:
            image, boxes_yolo = self._resize_direct(
                image=image,
                boxes_xyxy_cls=boxes_xyxy_cls
            )

        image_np = np.array(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        targets = self._create_targets(boxes_yolo=boxes_yolo)

        return image_tensor, targets
    



# Test thử
# from torch.utils.data import DataLoader

# train_dataset = YOLODataset(
#     json_path="train.json",
#     root_dir="/path/to/dataset",
#     anchors=ANCHORS,
#     image_size=IMAGE_SIZE,
#     scales=SCALES,
#     use_letterbox=True,
# )

# image, targets = train_dataset[0]

# print(image.shape)
# print(targets[0].shape)
# print(targets[1].shape)
# print(targets[2].shape)

# for idx, target in enumerate(targets):
#     print(
#         f"Scale {SCALES[idx]}:",
#         "positive boxes =",
#         (target[..., 0] == 1).sum().item(),
#         "ignored boxes =",
#         (target[..., 0] == -1).sum().item()
#     )