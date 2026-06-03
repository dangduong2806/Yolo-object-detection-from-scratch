# Nhận image, boxes_xyxy_cls
# Return image, boxes_xyxy_cls

import random

from PIL import Image, ImageEnhance

class DetectionAugmenter:
    def __init__(self, config):
        self.config = config

    def __call__(self, image, boxes_xyxy_cls):
        image, boxes_xyxy_cls = self.random_hflip(image, boxes_xyxy_cls)
        image, boxes_xyxy_cls = self.random_scale_translate(image, boxes_xyxy_cls)
        image = self.random_color(image)
        return image, boxes_xyxy_cls
    
    def random_hflip(self, image, boxes_xyxy_cls):
        prob = self.config.get("hflip_prob", 0.0)

        if random.random() >= prob:
            return image, boxes_xyxy_cls

        width, _ = image.size
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        flipped_boxes = []
        for x1, y1, x2, y2, class_id in boxes_xyxy_cls:
            new_x1 = width - x2
            new_x2 = width - x1

            if new_x2 <= new_x1:
                continue

            flipped_boxes.append([new_x1, y1, new_x2, y2, class_id])

        return image, flipped_boxes
    
    def random_color(self, image):
        prob = self.config.get("color_prob", 0.0)

        if random.random() >= prob:
            return image

        brightness = self.config.get("brightness", 0.0)
        contrast = self.config.get("contrast", 0.0)
        saturation = self.config.get("saturation", 0.0)

        if brightness > 0:
            factor = random.uniform(1.0 - brightness, 1.0 + brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if contrast > 0:
            factor = random.uniform(1.0 - contrast, 1.0 + contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if saturation > 0:
            factor = random.uniform(1.0 - saturation, 1.0 + saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        return image
    
    def random_scale_translate(self, image, boxes_xyxy_cls):
        prob = self.config.get("scale_translate_prob", 0.0)

        if random.random() >= prob:
            return image, boxes_xyxy_cls

        width, height = image.size

        scale_min = self.config.get("scale_min", 1.0)
        scale_max = self.config.get("scale_max", 1.0)
        translate = self.config.get("translate", 0.0)

        scale = random.uniform(scale_min, scale_max)
        dx = random.uniform(-translate, translate) * width
        dy = random.uniform(-translate, translate) * height

        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR

        image = image.transform(
            size=(width, height),
            method=Image.Transform.AFFINE,
            data=(
                1.0 / scale,
                0.0,
                -dx / scale,
                0.0,
                1.0 / scale,
                -dy / scale,
            ),
            resample=resample,
            fillcolor=(128, 128, 128),
        )

        transformed_boxes = []

        for x1, y1, x2, y2, class_id in boxes_xyxy_cls:
            new_x1 = x1 * scale + dx
            new_y1 = y1 * scale + dy
            new_x2 = x2 * scale + dx
            new_y2 = y2 * scale + dy

            new_x1 = max(0.0, min(float(new_x1), float(width)))
            new_y1 = max(0.0, min(float(new_y1), float(height)))
            new_x2 = max(0.0, min(float(new_x2), float(width)))
            new_y2 = max(0.0, min(float(new_y2), float(height)))

            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue

            transformed_boxes.append(
                [new_x1, new_y1, new_x2, new_y2, class_id]
            )

        return image, transformed_boxes