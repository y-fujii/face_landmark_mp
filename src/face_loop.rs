// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use crate::face_detector;
use crate::face_landmark;
use imageproc::geometric_transformations::Projection;

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

    pub fn run<I: image::GenericImageView<Pixel = image::Rgb<u8>>>(&self, image: &I) -> (Vec<(f32, f32, f32)>, f32) {
        let bboxes = self.detector.run(image);
        let bbox = bboxes
            .into_iter()
            .max_by(|e0, e1| e0.score.partial_cmp(&e1.score).unwrap())
            .unwrap();
        /*
        let k = imageproc::geometric_transformations::warp_into(
            &image,
            &Projection::scale(0.1, 0.1),
            imageproc::geometric_transformations::Interpolation::Bilinear,
            image::Rgb([127, 127, 127]),
        );
        */
        let image = image::imageops::crop_imm(
            image,
            ((5.0 / 4.0) * bbox.p0.1 - (1.0 / 4.0) * bbox.p1.1) as u32,
            ((5.0 / 4.0) * bbox.p0.0 - (1.0 / 4.0) * bbox.p1.0) as u32,
            ((3.0 / 2.0) * (bbox.p1.1 - bbox.p0.1)) as u32,
            ((3.0 / 2.0) * (bbox.p1.0 - bbox.p0.0)) as u32,
        );
        self.landmark.run(&image)
    }

}
