// Entirely stub code: classification is a later milestone (docs/TODO.md).
#![allow(dead_code)]

use super::ObjectClass;
use anyhow::Result;
use opencv::core::Mat;

pub trait Classifier: Send + Sync {
    fn classify(&self, image: &Mat) -> Result<(ObjectClass, f32)>;
}

pub struct ColorShapeClassifier;

impl ColorShapeClassifier {
    pub fn new() -> Self {
        Self
    }
}

impl Classifier for ColorShapeClassifier {
    fn classify(&self, _image: &Mat) -> Result<(ObjectClass, f32)> {
        // Placeholder until real classification (color histogram / ONNX)
        // lands. Deliberately returns Unknown at zero confidence so a stub
        // can never route bricks into the wrong bin.
        Ok((ObjectClass::Unknown, 0.0))
    }
}

pub struct BrickLinkClient {
    // api_key: String,
}

impl BrickLinkClient {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn search_part(&self, part_id: &str) -> Result<String> {
        // Mock query
        Ok(format!("Part {}: details...", part_id))
    }
}
