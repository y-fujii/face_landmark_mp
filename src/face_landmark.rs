// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use crate::image_util;
use crate::tflite;
use std::*;

pub struct FaceLandmark {
    size: usize,
    interp: tflite::Interpreter,
}

impl FaceLandmark {
    pub fn new() -> Self {
        let interp = tflite::Interpreter::new(include_bytes!("../models/face_landmark.tflite"));

        FaceLandmark {
            size: 192,
            interp: interp,
        }
    }

    pub fn run<I: image::GenericImageView<Pixel = image::Rgb<u8>>>(&self, image: &I) -> (Vec<(f32, f32, f32)>, f32) {
        let inputs = self.interp.inputs();
        let transform = image_util::resize_keeping_aspect(inputs[0].data_mut(), self.size, image);

        self.interp.invoke();

        let outputs = self.interp.outputs();
        let landmarks: &[f32] = outputs[0].data();
        let likelihood: &[f32] = outputs[1].data();
        assert_eq!(landmarks.len() % 3, 0);

        let mut dst = Vec::new();
        for i in 0..landmarks.len() / 3 {
            let x = transform.ax * landmarks[3 * i + 0] + transform.bx;
            let y = transform.ay * landmarks[3 * i + 1] + transform.by;
            let z = transform.ax * landmarks[3 * i + 2];
            dst.push((x, y, z));
        }

        (dst, likelihood[0])
    }
}
