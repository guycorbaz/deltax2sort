// Recognition is opt-in (`[recognition].enabled`); until a model is present
// these types are constructed only when it is turned on.
#![allow(dead_code)]

use super::ClassName;
use super::catalog::Catalog;
use crate::app_config::RecognitionConfig;
use anyhow::{Context, Result, anyhow};
use log::info;
use opencv::core::{Mat, Size};
use opencv::{imgproc, prelude::*};
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, PoisonError};
use tract_onnx::prelude::*;

/// Shared handle to the learned catalogue: the vision loop reads it (classify)
/// while the learning UI writes to it (teach). A std mutex is fine — contention
/// is negligible (teaching is a rare operator action).
pub type SharedCatalog = Arc<StdMutex<Catalog>>;

/// ImageNet normalisation — must match `models/export_embedder.py` exactly.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

type Plan = std::sync::Arc<TypedRunnableModel>;

/// Wraps the ONNX embedder: takes an object crop, returns its embedding vector.
pub struct OnnxEmbedder {
    model: Plan,
    input_size: i32,
}

impl OnnxEmbedder {
    /// Load and optimise the ONNX model for a fixed `1x3xN xN` f32 input.
    pub fn load(model_path: &str, input_size: i32) -> Result<Self> {
        let s = input_size as usize;
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("loading ONNX embedder from {model_path}"))?
            .with_input_fact(0, f32::fact([1, 3, s, s]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model, input_size })
    }

    /// Embed a BGR crop into the model's feature vector (not yet normalised —
    /// the catalogue L2-normalises on store/compare).
    pub fn embed(&self, crop: &Mat) -> Result<Vec<f32>> {
        let data = preprocess(crop, self.input_size)?;
        let s = self.input_size as usize;
        let input = Tensor::from_shape(&[1, 3, s, s], &data)?;
        let result = self.model.run(tvec!(input.into()))?;
        let out = result[0].clone().into_tensor();
        Ok(out.view().as_slice::<f32>()?.to_vec())
    }
}

/// Resize a BGR crop to `size`x`size`, convert to RGB, scale to [0,1] and apply
/// ImageNet normalisation, laid out as a CHW `3*size*size` f32 buffer — the
/// exact input the exported embedder expects.
fn preprocess(crop: &Mat, size: i32) -> Result<Vec<f32>> {
    let mut resized = Mat::default();
    imgproc::resize(
        crop,
        &mut resized,
        Size::new(size, size),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    if !rgb.is_continuous() {
        return Err(anyhow!("preprocessed crop is not contiguous"));
    }
    let bytes = rgb.data_bytes()?; // HWC u8, length size*size*3
    let s = size as usize;
    let mut data = vec![0f32; 3 * s * s];
    for y in 0..s {
        for x in 0..s {
            for c in 0..3 {
                let v = bytes[(y * s + x) * 3 + c] as f32 / 255.0;
                data[c * s * s + y * s + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(data)
}

/// The full recogniser: an embedder plus the (shared) learned catalogue and
/// the match threshold. Built only when `[recognition].enabled`.
pub struct Recognizer {
    embedder: OnnxEmbedder,
    catalog: SharedCatalog,
    threshold: f32,
}

impl Recognizer {
    /// Load the embedder and catalogue from config. A missing catalogue is not
    /// an error — recognition then starts empty (nothing recognised until the
    /// operator teaches classes). A missing/broken model IS an error.
    pub fn load(cfg: &RecognitionConfig) -> Result<Self> {
        let embedder = OnnxEmbedder::load(&cfg.model_path, cfg.input_size as i32)?;
        let catalog = if Path::new(&cfg.catalog_path).exists() {
            Catalog::load(&cfg.catalog_path)?
        } else {
            info!(
                "Recognition: catalogue {} not found — starting from an empty catalogue",
                cfg.catalog_path
            );
            Catalog::new()
        };
        Ok(Self {
            embedder,
            catalog: Arc::new(StdMutex::new(catalog)),
            threshold: cfg.match_threshold,
        })
    }

    /// A clone of the shared catalogue handle, so the learning UI can teach
    /// (add examples) the same catalogue the vision loop classifies against.
    pub fn catalog_handle(&self) -> SharedCatalog {
        self.catalog.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Catalog> {
        self.catalog.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn class_count(&self) -> usize {
        self.lock().classes().len()
    }

    /// Embed a BGR object crop into its feature vector.
    pub fn embed(&self, crop: &Mat) -> Result<Vec<f32>> {
        self.embedder.embed(crop)
    }

    /// Nearest catalogue class to `embedding` above the match threshold, else
    /// `None`. Second element is the cosine similarity (confidence).
    pub fn classify_embedding(&self, embedding: &[f32]) -> (Option<ClassName>, f32) {
        self.lock().classify(embedding, self.threshold)
    }

    /// The `n` most similar catalogue classes to `embedding` — the candidates
    /// the learning UI offers for an unrecognised object.
    pub fn nearest(&self, embedding: &[f32], n: usize) -> Vec<(ClassName, f32)> {
        self.lock().nearest(embedding, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC3, Scalar};

    #[test]
    fn preprocess_produces_a_chw_tensor_of_the_right_size() {
        // A 10x10 BGR crop resized to 8x8 → 3*8*8 f32 values.
        let crop = Mat::new_rows_cols_with_default(10, 10, CV_8UC3, Scalar::all(128.0)).unwrap();
        let data = preprocess(&crop, 8).unwrap();
        assert_eq!(data.len(), 3 * 8 * 8);
        // Mid-grey (128/255 ≈ 0.502) after ImageNet norm stays within a sane
        // range for every channel.
        assert!(data.iter().all(|v| v.abs() < 5.0), "values out of range");
    }

    #[test]
    fn loading_a_missing_model_errors_cleanly() {
        let err = match OnnxEmbedder::load("does/not/exist.onnx", 224) {
            Ok(_) => panic!("loading a missing model must fail"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("does/not/exist.onnx"),
            "error should name the missing model"
        );
    }
}
