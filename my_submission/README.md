# YOLOv3 ResNet50 Submission

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu dùng Docker, môi trường cần có PyTorch, TorchVision, Pillow, NumPy, tqdm và PyYAML.

## Trọng số mô hình

Checkpoint dùng để test được lưu tại:

```text
models/best.pth
```

Nếu `models/best.pth` không có sẵn, `predict.py` sẽ tự tải weight từ một trong các nguồn sau, theo thứ tự ưu tiên:

```text
1. --weights-url
2. MODEL_WEIGHTS_URL
3. weights.url trong config.yaml
```

Ví dụ:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --weights-url "https://your-direct-download-url/best.pth"
```

Khi thầy chạy bằng lệnh Docker, model weight sẽ được load từ Hugging face.


Checkpoint này là `best_map_resnet50_mse_phase2.pth`, được chọn bằng grid search inference trên validation với:

```text
conf_thresh = 0.30
nms_iou_thresh = 0.30
class-wise threshold = bật
max detections/image = 50
```

Kết quả validation tương ứng:

```text
mAP@0.5        : 0.608503
micro precision: 0.112185
micro recall   : 0.826818
pred boxes     : 14895
```

## Suy luận

Lệnh bắt buộc:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Mặc định script dùng `models/best.pth`. Có thể chỉ định checkpoint khác:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint models/best.pth
```

File `predictions.json` là một mảng JSON, mỗi phần tử có dạng:

```json
{
  "image_id": "img_7fd91a4c2e30.jpg",
  "boxes": [
    {
      "class": "person",
      "confidence": 0.91,
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```

Ảnh không có object vẫn được xuất với `"boxes": []`.

## Huấn luyện

Đặt dữ liệu theo cấu trúc public chuẩn:

```text
public/
  annotations/
    train.json
    val.json
  train/images/
  val/images/
```

Lệnh bắt buộc:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Sau khi train, model tốt nhất theo mAP@0.5 được lưu tại:

```text
models/best.pth
```

Có thể thêm tham số:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 20 \
  --batch-size 8 \
  --lr 0.00001
```
