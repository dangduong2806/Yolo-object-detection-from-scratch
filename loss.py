import torch 
import torch.nn as nn
import torch.nn.functional as F

def xywh_to_xyxy(box):
    x, y, w, h = box.unbind(dim=-1)

    return torch.stack(
        [
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2,
        ],
        dim=-1,
    )


def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(dim=-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)

    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2

    center_dist = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)

    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w2 / h2) - torch.atan(w1 / h1)
    ) ** 2

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - center_dist / enc_diag - alpha * v

    return ciou

class YOLOv3Loss(nn.Module):
    def __init__(
            self,
            num_classes = 5,
            lambda_box = 10.0,
            lambda_obj = 1.0,
            lambda_noobj = 10.0,
            lambda_class=1.0,
            box_loss_type="mse",
            obj_label_smoothing=0.0,
            class_label_smoothing=0.0,
            eps=1e-16,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.box_loss_type = box_loss_type
        if self.box_loss_type not in ["mse", "ciou"]:
            raise ValueError("box_loss_type must be 'mse' or 'ciou'")
        
        self.obj_label_smoothing = obj_label_smoothing
        self.class_label_smoothing = class_label_smoothing

        self.eps = eps

    def _bce_with_logits(self, pred, target):
        """
        BCEWithLogitsLoss an toàn trong trường hợp pred rỗng.
        Ví dụ batch không có object nào ở scale hiện tại.
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0
        
        return F.binary_cross_entropy_with_logits(
            input=pred,
            target=target,
            reduction="mean"
        )
    
    def _mse(self, pred, target):
        """
        MSELoss an toàn trong trường hợp pred rỗng.
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0

        return F.mse_loss(input=pred, target=target, reduction="mean")
    
    def forward(self, predictions, target, anchors):
        """
        predictions shape:
            [B, 3, S, S, 5 + num_classes]

        target shape:
            [B, 3, S, S, 6]

        target[..., 0] = objectness
            1  : anchor này chịu trách nhiệm dự đoán object
            0  : no object
            -1 : ignore, không tính loss

        target[..., 1] = x_cell
        target[..., 2] = y_cell
        target[..., 3] = w_cell
        target[..., 4] = h_cell
        target[..., 5] = class_id

        anchors shape:
            [3, 2]

        anchors phải được scale theo grid size S.
        Ví dụ với scale 13×13 thì anchor cũng phải ở đơn vị cell của grid 13.
        """
        device = predictions.device

        B, A, S, _, D = predictions.shape

        assert A == 3
        assert D == 5 + self.num_classes
        assert anchors.shape == (3, 2)

        anchors = anchors.to(device)

        # Mask object/ no-object
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        # 1. No-objectness loss
        # Anchor ko có object thì objectness target  = 0
        noobj_pred = predictions[..., 0][noobj_mask]
        noobj_target = torch.zeros_like(noobj_pred)

        noobj_loss = self._bce_with_logits(
            pred=noobj_pred,
            target=noobj_target
        )

        # 2. Objectness loss
        # Anchor có objec thì objectness target = 1
        obj_pred = predictions[..., 0][obj_mask]
        # obj_target = torch.ones_like(obj_pred)
        obj_target_value = 1.0 - self.obj_label_smoothing
        obj_target = torch.full_like(obj_pred, obj_target_value)

        obj_loss = self._bce_with_logits(
            pred=obj_pred,
            target=obj_target
        )

        # 3. Box coordinate loss
        # predictions[..., 1:3] là tx, ty

        # Đưa qua sigmoid về khoảng 0,1
        pred_xy = torch.sigmoid(predictions[..., 1:3])

        # predictions[..., 3:5] là tw, th dạng raw log-space
        pred_wh_raw = predictions[..., 3:5]

        # Target x, y đã là offset trong cell, nằm trong [0, 1].
        target_xy = target[..., 1:3]

        # Target w, h đang là w_cell, h_cell.
        target_wh = target[..., 3:5]

        # Tạo anchor grid shape [B, 3, S, S, 2]
        anchor_grid = anchors.reshape(1, 3, 1, 1, 2)
        anchor_grid = anchor_grid.expand(B, 3, S, S, 2)

        # Vì YOLOv3 decode:
        #   b_w = anchor_w * exp(t_w)
        # nên target cho t_w là:
        #   t_w* = log(b_w / anchor_w)
        target_wh_raw = torch.log(
            target_wh / anchor_grid + self.eps
        )

        # pred_box_for_loss = torch.cat(
        #     [pred_xy, pred_wh_raw],
        #     dim=-1
        # )

        # target_box_for_loss = torch.cat(
        #     [target_xy, target_wh_raw],
        #     dim=-1
        # )

        # box_loss = self._mse(
        #     pred=pred_box_for_loss[obj_mask],
        #     target=target_box_for_loss[obj_mask]
        # )
        if self.box_loss_type == "mse":
            pred_box_for_loss = torch.cat(
                [pred_xy, pred_wh_raw],
                dim=-1
            )

            target_box_for_loss = torch.cat(
                [target_xy, target_wh_raw],
                dim=-1
            )

            box_loss = self._mse(
                pred=pred_box_for_loss[obj_mask],
                target=target_box_for_loss[obj_mask]
            )

        else:
            pred_wh = torch.exp(pred_wh_raw.clamp(min=-10, max=10)) * anchor_grid

            grid_y, grid_x = torch.meshgrid(
                torch.arange(S, device=device),
                torch.arange(S, device=device),
                indexing="ij",
            )

            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.reshape(1, 1, S, S, 2)

            pred_xy_grid = pred_xy + grid
            target_xy_grid = target_xy + grid

            pred_box_xywh = torch.cat(
                [pred_xy_grid, pred_wh],
                dim=-1,
            )

            target_box_xywh = torch.cat(
                [target_xy_grid, target_wh],
                dim=-1,
            )

            pred_box_xyxy = xywh_to_xyxy(pred_box_xywh)
            target_box_xyxy = xywh_to_xyxy(target_box_xywh)

            ciou = bbox_ciou(
                pred_box_xyxy[obj_mask],
                target_box_xyxy[obj_mask],
                eps=self.eps,
            )

            if ciou.numel() == 0:
                box_loss = predictions.sum() * 0.0
            else:
                box_loss = (1.0 - ciou).mean()

        # 4. Class loss
        # YOLOv3 dùng independent logistic classifiers,
        # nên ta dùng BCEWithLogits với one-hot target.
        pred_class_logits = predictions[..., 5:][obj_mask]

        target_class_ids = target[..., 5][obj_mask].long()

        if target_class_ids.numel() == 0:
            class_loss = pred_class_logits.sum() * 0.0
        else:
            target_class_onehot = F.one_hot(
                target_class_ids,
                num_classes=self.num_classes
            ).float()
            
            # added
            if self.class_label_smoothing > 0:
                smooth = self.class_label_smoothing
                target_class_onehot = (
                    target_class_onehot * (1.0 - smooth)
                    + smooth / self.num_classes
                )

            class_loss = self._bce_with_logits(
                pred=pred_class_logits,
                target=target_class_onehot
            )
        
        # Total loss
        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )

        # dictionary
        loss_items = {
            "box_loss": box_loss.detach(),
            "obj_loss": obj_loss.detach(),
            "noobj_loss": noobj_loss.detach(),
            "class_loss": class_loss.detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss, loss_items


         