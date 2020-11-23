// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use crate::face_detector;
use crate::face_landmark;
use imageproc::geometric_transformations;
use nalgebra::{Matrix3, Vector2, Vector3};
use std::convert::TryInto;
use std::*;

pub struct FaceLoop {
    detector: face_detector::FaceDetector,
    landmark: face_landmark::FaceLandmark,
}

impl FaceLoop {
    pub fn new() -> Self {
        FaceLoop {
            detector: face_detector::FaceDetector::new(),
            landmark: face_landmark::FaceLandmark::new(),
        }
    }

    pub fn run(&self, image: &image::RgbImage) -> (Vec<(f32, f32, f32)>, f32) {
        let bboxes = self.detector.run(image);
        let bbox = bboxes
            .into_iter()
            .max_by(|e0, e1| e0.score.partial_cmp(&e1.score).unwrap())
            .unwrap();

        let size = f32::round(1.5 * f32::max(bbox.size.0, bbox.size.1));
        let dir_y = bbox.key_points[0].0 - bbox.key_points[1].0;
        let dir_x = bbox.key_points[1].1 - bbox.key_points[0].1;
        let len = f32::hypot(dir_y, dir_x);
        let sin = dir_y / len;
        let cos = dir_x / len;
        let n_transform = Matrix3::new_translation(&(Vector2::new(size, size) / 2.0))
            * Matrix3::new(cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0)
            * Matrix3::new_translation(&-Vector2::new(bbox.center.1, bbox.center.0));
        let i_transform = Matrix3::new_translation(&Vector2::new(bbox.center.1, bbox.center.0))
            * Matrix3::new(cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0)
            * Matrix3::new_translation(&(-Vector2::new(size, size) / 2.0));

        let mut cropped = image::RgbImage::new(size as u32, size as u32);
        geometric_transformations::warp_into(
            image,
            &geometric_transformations::Projection::from_matrix(n_transform.transpose().as_slice().try_into().unwrap())
                .unwrap(),
            geometric_transformations::Interpolation::Bilinear,
            image::Rgb([127, 127, 127]),
            &mut cropped,
        );
        cropped.save("intermediate.png");

        let (landmarks, likelihood) = self.landmark.run(&cropped);

        let mut dst = Vec::new();
        for (x, y, z) in landmarks.iter() {
            let v = i_transform * Vector3::new(*x, *y, 1.0);
            dst.push((v[0], v[1], *z));
        }

        (dst, likelihood)
    }
}
