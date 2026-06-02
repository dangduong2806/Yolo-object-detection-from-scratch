# Nhận image, boxes_xyxy_cls
# Return image, boxes_xyxy_cls

class DetectionAugmenter:
    def __init__(self, config):
        self.config = config

    def __call__(self, image, boxes_xyxy_cls):
        image, boxes_xyxy_cls = self.random_hflip(image, boxes_xyxy_cls)
        image, boxes_xyxy_cls = self.random_scale_translate(image, boxes_xyxy_cls)
        image = self.random_color(image)
        return image, boxes_xyxy_cls