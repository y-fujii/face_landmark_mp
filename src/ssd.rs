// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
//
// almost 1:1 translation of:
//     mediapipe/framework/formats/object_detection/anchor.proto
//     mediapipe/calculators/tflite/ssd_anchors_calculator.proto
//     mediapipe/calculators/tflite/ssd_anchors_calculator.cc
//     Copyright 2019 The MediaPipe Authors, under Apache License v2.0.

#[derive(Debug)]
pub struct Anchor {
    pub center: (f32, f32),
    pub size: (f32, f32),
}

pub struct Options {
    // Size of input images.
    pub input_size_width: usize,
    pub input_size_height: usize,

    // Min and max scales for generating anchor boxes on feature maps.
    pub min_scale: f32,
    pub max_scale: f32,

    // The offset for the center of anchors. The value is in the scale of stride.
    // E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    pub anchor_offset_x: f32, // default: 0.5.
    pub anchor_offset_y: f32, // default: 0.5.

    // Number of output feature maps to generate the anchors on.
    pub num_layers: usize,
    // Sizes of output feature maps to create anchors. Either feature_map size or
    // stride should be provided.
    pub feature_map_width: Vec<usize>,
    pub feature_map_height: Vec<usize>,
    // Strides of each output feature maps.
    pub strides: Vec<usize>,

    // List of different aspect ratio to generate anchors.
    pub aspect_ratios: Vec<f32>,

    // A boolean to indicate whether the fixed 3 boxes per location is used in the
    // lowest layer.
    pub reduce_boxes_in_lowest_layer: bool, // optional, default: false.
    // An additional anchor is added with this aspect ratio and a scale
    // interpolated between the scale for a layer and the scale for the next layer
    // (1.0 for the last layer). This anchor is not included if this value is 0.
    pub interpolated_scale_aspect_ratio: f32, // optional, default: 1.0.

    // Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    // This option can be used when the predicted anchor width and height are in
    // pixels.
    pub fixed_anchor_size: bool, // optional, default: false.
}

pub fn generate(options: &Options) -> Vec<Anchor> {
    let mut anchors = Vec::new();
    let mut layer_id = 0;
    while layer_id < options.num_layers {
        let mut anchor_height = Vec::new();
        let mut anchor_width = Vec::new();
        let mut aspect_ratios = Vec::new();
        let mut scales = Vec::new();

        // For same strides, we merge the anchors in the same order.
        let mut last_same_stride_layer = layer_id;
        while last_same_stride_layer < options.strides.len()
            && options.strides[last_same_stride_layer] == options.strides[layer_id]
        {
            let scale = calculate_scale(
                options.min_scale,
                options.max_scale,
                last_same_stride_layer,
                options.strides.len(),
            );
            if last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push(1.0);
                aspect_ratios.push(2.0);
                aspect_ratios.push(0.5);
                scales.push(0.1);
                scales.push(scale);
                scales.push(scale);
            } else {
                for aspect_ratio_id in 0..options.aspect_ratios.len() {
                    aspect_ratios.push(options.aspect_ratios[aspect_ratio_id]);
                    scales.push(scale);
                }
                if options.interpolated_scale_aspect_ratio > 0.0 {
                    let scale_next = if last_same_stride_layer + 1 == options.strides.len() {
                        1.0
                    } else {
                        calculate_scale(
                            options.min_scale,
                            options.max_scale,
                            last_same_stride_layer + 1,
                            options.strides.len(),
                        )
                    };
                    scales.push(f32::sqrt(scale * scale_next));
                    aspect_ratios.push(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer += 1;
        }

        for i in 0..aspect_ratios.len() {
            let ratio_sqrts = f32::sqrt(aspect_ratios[i]);
            anchor_height.push(scales[i] / ratio_sqrts);
            anchor_width.push(scales[i] * ratio_sqrts);
        }

        let feature_map_height;
        let feature_map_width;
        if !options.feature_map_height.is_empty() {
            feature_map_height = options.feature_map_height[layer_id];
            feature_map_width = options.feature_map_width[layer_id];
        } else {
            let stride = options.strides[layer_id];
            feature_map_height = (options.input_size_height + stride - 1) / stride;
            feature_map_width = (options.input_size_width + stride - 1) / stride;
        }

        for y in 0..feature_map_height {
            for x in 0..feature_map_width {
                for anchor_id in 0..anchor_height.len() {
                    // TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    let center_y = (y as f32 + options.anchor_offset_y) / feature_map_height as f32;
                    let center_x = (x as f32 + options.anchor_offset_x) / feature_map_width as f32;

                    let (anchor_h, anchor_w);
                    if options.fixed_anchor_size {
                        anchor_h = 1.0;
                        anchor_w = 1.0;
                    } else {
                        anchor_h = anchor_height[anchor_id];
                        anchor_w = anchor_width[anchor_id];
                    }

                    anchors.push(Anchor {
                        center: (center_y, center_x),
                        size: (anchor_h, anchor_w),
                    });
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    anchors
}

fn calculate_scale(min_scale: f32, max_scale: f32, stride_index: usize, num_strides: usize) -> f32 {
    if num_strides == 1 {
        (min_scale + max_scale) * 0.5
    } else {
        min_scale + (max_scale - min_scale) * stride_index as f32 / (num_strides - 1) as f32
    }
}
