use crate::hardware::WorkspaceLimits;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub robot: RobotConfig,
    pub conveyor: ConveyorConfig,
    pub camera: CameraConfig,
    #[serde(default)]
    pub sorting: SortingConfig,
    #[serde(default)]
    pub vision: VisionConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RobotConfig {
    pub port_name: String,
    pub baud_rate: u32,
    pub home_on_connect: bool,
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    #[serde(default = "default_z_min")]
    pub z_min: f32,
    #[serde(default = "default_z_max")]
    pub z_max: f32,
    pub z_pick: f32,
    pub z_travel: f32,
    /// G-code feed rate in mm/min (F parameter of G01).
    #[serde(default = "default_feed_rate")]
    pub feed_rate: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConveyorConfig {
    pub port_name: String,
    pub baud_rate: u32,
    /// Raw S-value sent with the M3 start command.
    pub default_speed: u32,
    /// Belt speed in robot coordinates, mm/s. Signed: positive means objects
    /// travel toward +Y. Used by the trajectory planner.
    #[serde(default = "default_belt_speed")]
    pub speed_mm_s: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CameraConfig {
    pub device_id: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct SortingConfig {
    pub drop_x: f32,
    pub drop_y: f32,
    pub drop_z: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct VisionConfig {
    /// Fixed binary threshold (0-255) applied after grayscale + blur.
    pub threshold: f64,
    /// Contour area limits in pixels² for a blob to count as an object.
    pub min_area: f64,
    pub max_area: f64,
    /// true: objects darker than the belt (THRESH_BINARY_INV);
    /// false: objects lighter than the belt.
    pub invert: bool,
}

fn default_z_min() -> f32 {
    -200.0
}
fn default_z_max() -> f32 {
    0.0
}
fn default_feed_rate() -> u32 {
    15000
}
fn default_belt_speed() -> f32 {
    100.0
}

impl Default for SortingConfig {
    fn default() -> Self {
        Self {
            drop_x: 150.0,
            drop_y: 0.0,
            drop_z: -100.0,
        }
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            threshold: 60.0,
            min_area: 500.0,
            max_area: 10000.0,
            invert: true,
        }
    }
}

impl RobotConfig {
    pub fn workspace_limits(&self) -> WorkspaceLimits {
        WorkspaceLimits {
            x_min: self.x_min,
            x_max: self.x_max,
            y_min: self.y_min,
            y_max: self.y_max,
            z_min: self.z_min,
            z_max: self.z_max,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            robot: RobotConfig {
                port_name: "/dev/ttyUSB0".to_string(),
                baud_rate: 115200,
                home_on_connect: true,
                x_min: -160.0,
                x_max: 160.0,
                y_min: -160.0,
                y_max: 160.0,
                z_min: default_z_min(),
                z_max: default_z_max(),
                z_pick: -180.0,
                z_travel: -50.0,
                feed_rate: default_feed_rate(),
            },
            conveyor: ConveyorConfig {
                port_name: "/dev/ttyUSB1".to_string(),
                baud_rate: 115200,
                default_speed: 1000,
                speed_mm_s: default_belt_speed(),
            },
            camera: CameraConfig {
                device_id: 0,
                width: 1280,
                height: 720,
            },
            sorting: SortingConfig::default(),
            vision: VisionConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            let default_config = Self::default();
            default_config.save(path)?;
            return Ok(default_config);
        }
        let content = fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Reject configurations that would command the robot outside its
    /// physical workspace before any hardware is touched.
    pub fn validate(&self) -> Result<()> {
        let r = &self.robot;
        ensure!(r.x_min < r.x_max, "robot.x_min must be < robot.x_max");
        ensure!(r.y_min < r.y_max, "robot.y_min must be < robot.y_max");
        ensure!(r.z_min < r.z_max, "robot.z_min must be < robot.z_max");
        for (name, z) in [("z_pick", r.z_pick), ("z_travel", r.z_travel)] {
            ensure!(
                z >= r.z_min && z <= r.z_max,
                "robot.{} ({}) is outside the Z range [{}, {}]",
                name,
                z,
                r.z_min,
                r.z_max
            );
        }
        ensure!(
            r.z_travel > r.z_pick,
            "robot.z_travel ({}) must be above robot.z_pick ({})",
            r.z_travel,
            r.z_pick
        );
        let s = &self.sorting;
        ensure!(
            s.drop_x >= r.x_min
                && s.drop_x <= r.x_max
                && s.drop_y >= r.y_min
                && s.drop_y <= r.y_max
                && s.drop_z >= r.z_min
                && s.drop_z <= r.z_max,
            "sorting drop position ({}, {}, {}) is outside the robot workspace",
            s.drop_x,
            s.drop_y,
            s.drop_z
        );
        ensure!(
            self.camera.width > 0 && self.camera.height > 0,
            "camera resolution must be non-zero"
        );
        let v = &self.vision;
        ensure!(
            (0.0..=255.0).contains(&v.threshold),
            "vision.threshold must be within 0-255"
        );
        ensure!(
            v.min_area < v.max_area,
            "vision.min_area must be < vision.max_area"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_roundtrips() {
        let cfg = AppConfig::default();
        cfg.validate().unwrap();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: AppConfig = toml::from_str(&text).unwrap();
        back.validate().unwrap();
        assert_eq!(back.robot.z_pick, cfg.robot.z_pick);
        assert_eq!(back.sorting.drop_x, cfg.sorting.drop_x);
    }

    #[test]
    fn legacy_config_without_new_fields_parses_with_defaults() {
        let legacy = r#"
            [robot]
            port_name = "/dev/ttyUSB0"
            baud_rate = 115200
            home_on_connect = true
            x_min = -150.0
            x_max = 150.0
            y_min = -150.0
            y_max = 150.0
            z_pick = -180.0
            z_travel = -150.0

            [conveyor]
            port_name = "/dev/ttyUSB1"
            baud_rate = 115200
            default_speed = 1000

            [camera]
            device_id = 0
            width = 1280
            height = 720
        "#;
        let cfg: AppConfig = toml::from_str(legacy).unwrap();
        assert_eq!(cfg.robot.z_min, -200.0);
        assert_eq!(cfg.robot.feed_rate, 15000);
        assert_eq!(cfg.conveyor.speed_mm_s, 100.0);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_z_pick_outside_z_range() {
        let mut cfg = AppConfig::default();
        cfg.robot.z_pick = -250.0; // below z_min (-200): physically unreachable
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_drop_position_outside_workspace() {
        let mut cfg = AppConfig::default();
        cfg.sorting.drop_x = 500.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn load_creates_default_file_when_missing() {
        let path = std::env::temp_dir().join(format!(
            "deltax2sort_test_config_{}.toml",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let cfg = AppConfig::load(&path).unwrap();
        assert!(path.exists());
        cfg.validate().unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
