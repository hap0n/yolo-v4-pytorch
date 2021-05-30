from detection.detect_image import YOLOv4Detector

D = YOLOv4Detector()

D.detect_image('../data/people.jpg', 896, iou_threshold=0.2, score_threshold=0.035)
D.detect_image('../data/dog.jpg', 800, iou_threshold=0.8, score_threshold=0.15)
D.detect_image('../data/cars.jpg', 896, iou_threshold=0.2, score_threshold=0.05)
