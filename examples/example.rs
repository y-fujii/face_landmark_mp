// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use face_landmark_mp::*;
use opencv::prelude::*;
use std::*;

fn main() -> Result<(), Box<dyn error::Error>> {
    let mut capture = opencv::videoio::VideoCapture::new(0, 0)?;
    let face_loop = face_loop::FaceLoop::new();
    loop {
        let mut src = opencv::core::Mat::default()?;
        capture.read(&mut src)?;

        let mut img = image::RgbImage::from_raw(
            src.cols() as u32,
            src.rows() as u32,
            src.reshape(1, 1)?.data_typed()?.to_vec(),
        ).unwrap();
        swap_rgb(&mut img);

        let (landmarks, landmarks_likelihood) = face_loop.run(&img);
        dbg!(landmarks_likelihood);
        for (x, y, _) in landmarks.iter() {
            if 0.0 <= *x && *x < img.width() as f32 && 0.0 <= *y && *y < img.height() as f32 {
                img[(*x as u32, *y as u32)] = image::Rgb([0, 255, 255]);
            }
        }

        swap_rgb(&mut img);
        let dst = Mat::from_slice(&img.to_vec())?.reshape(3, src.rows())?;

        opencv::highgui::imshow("camera", &dst)?;
        opencv::highgui::wait_key(1)?;
    }
}

fn swap_rgb(img: &mut image::RgbImage) {
    for px in img.pixels_mut() {
        px.0.swap(0, 2);
    }
}
