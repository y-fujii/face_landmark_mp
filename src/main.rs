// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use std::*;
mod face_detector;
mod face_landmark;
mod face_loop;
mod image_util;
mod ssd;
mod tflite;

fn main() -> Result<(), Box<dyn error::Error>> {
    let mut image = image::open("src.jpg")?.to_rgb();

    let face_loop = face_loop::FaceLoop::new();
    let (landmarks, landmarks_likelihood) = face_loop.run(&image);
    dbg!(landmarks_likelihood);
    for (x, y, _) in landmarks.iter() {
        image[(*x as u32, *y as u32)] = image::Rgb([0, 255, 255]);
    }
    image.save("dst.png")?;

    Ok(())
}
