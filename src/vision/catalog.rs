// The catalogue core (recognition + portable file) lands here first; the
// vision loop starts using it in phase B2 (embedder + config wiring). Until
// then it is exercised only by its unit tests.
#![allow(dead_code)]

use super::ClassName;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One labelled example: a class name and the embedding of a cropped view of
/// that object. Embeddings are stored L2-normalised so similarity is a plain
/// dot product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub class: ClassName,
    pub embedding: Vec<f32>,
}

/// The object catalogue — the portable "learned file". It holds labelled
/// example embeddings and recognises new objects by nearest-neighbour cosine
/// similarity. Deliberately bin-agnostic: recognition only. Learned by example
/// (`add`), saved/loaded as one file, and copied from the teaching workstation
/// to the Pi.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    examples: Vec<Example>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Distinct class names, sorted — for the assignment UI and diagnostics.
    pub fn classes(&self) -> Vec<ClassName> {
        let mut names: Vec<ClassName> = self.examples.iter().map(|e| e.class.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Teach one example. The embedding is stored L2-normalised so
    /// classification is a dot product (cosine similarity).
    pub fn add(&mut self, class: impl Into<ClassName>, embedding: &[f32]) {
        self.examples.push(Example {
            class: class.into(),
            embedding: normalize(embedding),
        });
    }

    /// The `n` most similar examples to `embedding`, most similar first, as
    /// `(class, cosine_similarity in [-1, 1])`. Drives both classification and
    /// the "show me the closest known objects" learning UI.
    pub fn nearest(&self, embedding: &[f32], n: usize) -> Vec<(ClassName, f32)> {
        let query = normalize(embedding);
        let mut scored: Vec<(ClassName, f32)> = self
            .examples
            .iter()
            .map(|e| (e.class.clone(), dot(&query, &e.embedding)))
            .collect();
        // Descending similarity; total_cmp keeps NaN from panicking the sort.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(n);
        scored
    }

    /// Recognise `embedding`: returns the nearest example's class if its cosine
    /// similarity is at least `min_similarity`, else `None` (unrecognised — the
    /// object is then left on the belt). The second element is that similarity,
    /// carried as the confidence.
    pub fn classify(&self, embedding: &[f32], min_similarity: f32) -> (Option<ClassName>, f32) {
        match self.nearest(embedding, 1).into_iter().next() {
            Some((class, sim)) if sim >= min_similarity => (Some(class), sim),
            Some((_, sim)) => (None, sim),
            None => (None, 0.0),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = toml::to_string(self).context("serialising the object catalogue")?;
        std::fs::write(path, text)
            .with_context(|| format!("writing the object catalogue to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading the object catalogue from {}", path.display()))?;
        toml::from_str(&text).context("parsing the object catalogue")
    }
}

/// L2-normalise a vector; a zero vector is returned unchanged (its similarity
/// to anything is 0, which classification treats as unrecognised).
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Dot product of two equal-length vectors (cosine similarity when both are
/// normalised). Mismatched lengths only arise from mixing embedders, a bug;
/// zip truncates rather than panicking.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Three orthogonal unit directions stand in for three visually distinct
    // object classes; a query near one of them must recognise it.
    fn catalog_of_three() -> Catalog {
        let mut c = Catalog::new();
        c.add("red", &[1.0, 0.0, 0.0]);
        c.add("green", &[0.0, 1.0, 0.0]);
        c.add("blue", &[0.0, 0.0, 1.0]);
        c
    }

    #[test]
    fn classifies_the_nearest_class_above_threshold() {
        let c = catalog_of_three();
        // Close to "red" (small perturbation): recognised.
        let (class, sim) = c.classify(&[0.95, 0.1, 0.05], 0.8);
        assert_eq!(class.as_deref(), Some("red"));
        assert!(sim > 0.8, "similarity {sim}");
    }

    #[test]
    fn unrecognised_when_below_threshold() {
        let c = catalog_of_three();
        // Equidistant-ish from all three axes: nearest similarity ~0.577 < 0.8.
        let (class, _) = c.classify(&[1.0, 1.0, 1.0], 0.8);
        assert!(class.is_none(), "diffuse embedding must be unrecognised");
    }

    #[test]
    fn empty_catalog_recognises_nothing() {
        let (class, sim) = Catalog::new().classify(&[1.0, 0.0, 0.0], 0.0);
        assert!(class.is_none());
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn nearest_returns_classes_by_descending_similarity() {
        let c = catalog_of_three();
        let near = c.nearest(&[0.9, 0.4, 0.0], 3);
        assert_eq!(near.len(), 3);
        assert_eq!(near[0].0, "red"); // most similar
        assert_eq!(near[1].0, "green");
        assert!(near[0].1 >= near[1].1 && near[1].1 >= near[2].1);
    }

    #[test]
    fn classes_are_distinct_and_sorted() {
        let mut c = Catalog::new();
        c.add("brick", &[1.0, 0.0]);
        c.add("plate", &[0.0, 1.0]);
        c.add("brick", &[0.9, 0.1]); // second example of an existing class
        assert_eq!(c.len(), 3);
        assert_eq!(c.classes(), vec!["brick".to_string(), "plate".to_string()]);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut c = Catalog::new();
        c.add("red", &[3.0, 4.0]); // stored normalised to (0.6, 0.8)
        let path = std::env::temp_dir().join(format!(
            "deltax2sort_catalog_test_{}.toml",
            std::process::id()
        ));
        c.save(&path).unwrap();
        let back = Catalog::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.len(), 1);
        // Classification survives the roundtrip.
        let (class, _) = back.classify(&[3.0, 4.0], 0.99);
        assert_eq!(class.as_deref(), Some("red"));
    }
}
