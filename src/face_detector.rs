// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use crate::image_util;
use crate::ssd;
use crate::tflite;
use std::*;

#[derive(Debug)]
pub struct BBox {
    pub center: (f32, f32),
    pub size: (f32, f32),
    pub key_points: [(f32, f32); 6],
    pub score: f32,
}

pub struct FaceDetector {
    size: usize,
    anchors: Vec<ssd::Anchor>,
    interp: tflite::Interpreter,
}

impl FaceDetector {
    pub fn new() -> Self {
        let size = 128;
        let anchors = ssd::generate(&ssd::Options {
            input_size_width: size,
            input_size_height: size,
            min_scale: 0.1484375,
            max_scale: 0.75,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            num_layers: 4,
            feature_map_width: vec![],
            feature_map_height: vec![],
            strides: vec![8, 16, 16, 16],
            aspect_ratios: vec![1.0],
            reduce_boxes_in_lowest_layer: false,
            interpolated_scale_aspect_ratio: 1.0,
            fixed_anchor_size: true,
        });
        assert_eq!(anchors.len(), 896);

        let interp = tflite::Interpreter::new(include_bytes!("../models/face_detection_front.tflite"));

        FaceDetector {
            size: size,
            anchors: anchors,
            interp: interp,
        }
    }

    pub fn run<I: image::GenericImageView<Pixel = image::Rgb<u8>>>(&self, image: &I) -> Vec<BBox> {
        let inputs = self.interp.inputs();
        let transform = image_util::resize_keeping_aspect(inputs[0].data_mut(), self.size, image);

        self.interp.invoke();

        let outputs = self.interp.outputs();
        let boxes: &[f32] = outputs[0].data();
        let scores: &[f32] = outputs[1].data();
        self.decode_output(boxes, scores, &transform)
    }

    // ref. <https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_front_cpu.pbtxt>.
    // ref. <https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc>.
    fn decode_output(&self, boxes: &[f32], scores: &[f32], transform: &image_util::Transform) -> Vec<BBox> {
        let mut dst = Vec::new();
        for (i, anchor) in self.anchors.iter().enumerate() {
            let ay = anchor.size.0 * transform.ay;
            let ax = anchor.size.1 * transform.ax;
            let by = anchor.center.0 * (self.size as f32 * transform.ay) + transform.by;
            let bx = anchor.center.1 * (self.size as f32 * transform.ax) + transform.bx;
            let translate = |y, x| (ay * y + by, ax * x + bx);
            dst.push(BBox {
                center: translate(boxes[16 * i + 1], boxes[16 * i + 0]),
                size: (ay * boxes[16 * i + 3], ax * boxes[16 * i + 2]),
                key_points: [
                    translate(boxes[16 * i + 5], boxes[16 * i + 4]),
                    translate(boxes[16 * i + 7], boxes[16 * i + 6]),
                    translate(boxes[16 * i + 9], boxes[16 * i + 8]),
                    translate(boxes[16 * i + 11], boxes[16 * i + 10]),
                    translate(boxes[16 * i + 13], boxes[16 * i + 12]),
                    translate(boxes[16 * i + 15], boxes[16 * i + 14]),
                ],
                score: 1.0 / (1.0 + f32::exp(-scores[i])),
            });
        }
        dst
    }
}
