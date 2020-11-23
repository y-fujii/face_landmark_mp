// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use std::*;

#[derive(Debug)]
pub struct Transform {
    pub ay: f32,
    pub ax: f32,
    pub by: f32,
    pub bx: f32,
}

pub fn resize_keeping_aspect<I: image::GenericImageView<Pixel = image::Rgb<u8>>>(
    dst: &mut [f32], size: usize, src: &I,
) -> Transform {
    let src_size = (src.height() as usize, src.width() as usize);
    let rect = if cmp::max(src_size.0, src_size.1) == size {
        convert_with_padding(dst, size, src)
    } else {
        let scale = cmp::min(size * src_size.1, size * src_size.0);
        let scaled_h = (scale + src_size.1 / 2) / src_size.1;
        let scaled_w = (scale + src_size.0 / 2) / src_size.0;
        let tmp = image::imageops::resize(
            src,
            scaled_w as u32,
            scaled_h as u32,
            image::imageops::FilterType::Triangle,
        );
        convert_with_padding(dst, size, &tmp)
    };

    let ay = src_size.0 as f32 / (rect.2 - rect.0) as f32;
    let ax = src_size.1 as f32 / (rect.3 - rect.1) as f32;
    Transform {
        ay: ay,
        ax: ax,
        by: -ay * rect.0 as f32,
        bx: -ax * rect.1 as f32,
    }
}

fn convert_with_padding<I: image::GenericImageView<Pixel = image::Rgb<u8>>>(
    dst: &mut [f32], size: usize, src: &I,
) -> (usize, usize, usize, usize) {
    let y0 = (1 + size - src.height() as usize) / 2;
    let x0 = (1 + size - src.width() as usize) / 2;
    let y1 = y0 + src.height() as usize;
    let x1 = x0 + src.width() as usize;
    // XXX: optimize.
    for y in 0..size {
        for x in 0..size {
            for c in 0..3 {
                dst[(3 * size) * y + 3 * x + c] = if y0 <= y && y < y1 && x0 <= x && x < x1 {
                    src.get_pixel((x - x0) as u32, (y - y0) as u32)[c] as f32 / 255.0 - 0.5
                } else {
                    0.0
                }
            }
        }
    }
    (y0, x0, y1, x1)
}
