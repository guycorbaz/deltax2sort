// Entirely stub code: classification is a later milestone (docs/TODO.md).
#![allow(dead_code)]

use super::ClassName;
use anyhow::Result;
use opencv::core::Mat;

pub trait Classifier: Send + Sync {
    /// Recognise the object in `image`, returning `(class, confidence)`.
    /// `None` = unrecognised (rides the belt to the catch bin).
    fn classify(&self, image: &Mat) -> Result<(Option<ClassName>, f32)>;
}

pub struct ColorShapeClassifier;

impl ColorShapeClassifier {
    pub fn new() -> Self {
        Self
    }
}

impl Classifier for ColorShapeClassifier {
    fn classify(&self, _image: &Mat) -> Result<(Option<ClassName>, f32)> {
        // Placeholder until real recognition (embeddings + nearest-neighbour
        // over the learned catalogue) lands in Phase B. Deliberately returns
        // None at zero confidence so a stub can never route a part into the
        // wrong bin — an unrecognised object is simply not picked.
        Ok((None, 0.0))
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
