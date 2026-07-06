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
    #[serde(default)]
    pub logging: LoggingConfig,
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
    /// Whether an emergency stop should open the gripper (M05 sent just before
    /// M112 on the E-stop halt sequence). Default false: a held part stays
    /// held, so parts are not dropped at an arbitrary position. Set true only
    /// for cells where dropping on E-stop is the safe, desired behaviour.
    #[serde(default)]
    pub release_gripper_on_estop: bool,
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
    /// Requested capture resolution. The camera may pick the nearest mode it
    /// supports; the driver logs the resolution actually in effect.
    pub width: u32,
    pub height: u32,
    /// Requested capture frame rate.
    #[serde(default = "default_camera_fps")]
    pub fps: u32,
    /// Optional pixel-format FOURCC (e.g. "MJPG"); many UVC cameras need it
    /// to reach full frame rate at higher resolutions. None = driver default.
    #[serde(default)]
    pub fourcc: Option<String>,
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
    /// Camera scale in mm per pixel at the belt plane. Feeds the
    /// pixel→robot transform (`CalibrationParams::centered`).
    #[serde(default = "default_mm_per_px")]
    pub mm_per_px: f32,
}

/// Logging to a daily-rotating file (for debugging) plus, optionally, the
/// console. Backed by flexi_logger; the `log` macros elsewhere are unchanged.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct LoggingConfig {
    /// Level spec in the `RUST_LOG` grammar — a bare level (`"info"`,
    /// `"debug"`) or a per-module filter (`"info,deltax2sort=debug"`). A
    /// `RUST_LOG` environment variable, if set, overrides this at startup.
    pub level: String,
    /// Directory the log files are written to (created if absent).
    pub directory: String,
    /// Also mirror log records to stderr (handy in dev / mock; a pure kiosk
    /// can turn it off).
    pub to_console: bool,
    /// How many rotated daily files to keep; older ones are pruned. 0 keeps
    /// them all (watch disk on the Pi).
    pub keep_days: u16,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            directory: "logs".to_string(),
            to_console: true,
            keep_days: 7,
        }
    }
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
fn default_camera_fps() -> u32 {
    30
}
fn default_mm_per_px() -> f32 {
    0.5
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
            mm_per_px: default_mm_per_px(),
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
                release_gripper_on_estop: false,
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
                fps: default_camera_fps(),
                fourcc: None,
            },
            sorting: SortingConfig::default(),
            vision: VisionConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Delta X2 (SP-X2) physical envelope. Configured workspace bounds may be
/// tighter than this but never wider — a typo like `x_max = 1600.0` must not
/// let `move_to` accept unreachable targets.
const ENVELOPE_XY_MM: f32 = 160.0;
const ENVELOPE_Z_MIN_MM: f32 = -200.0;
const ENVELOPE_Z_MAX_MM: f32 = 0.0;

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            let default_config = Self::default();
            // Guard the default-creation path too: never write or return a
            // config that would not pass validation.
            default_config.validate()?;
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
        // Workspace must stay within the physical envelope (also rejects NaN
        // bounds, which fail every comparison).
        ensure!(
            r.x_min >= -ENVELOPE_XY_MM && r.x_max <= ENVELOPE_XY_MM,
            "robot X bounds [{}, {}] exceed the physical envelope [±{} mm]",
            r.x_min,
            r.x_max,
            ENVELOPE_XY_MM
        );
        ensure!(
            r.y_min >= -ENVELOPE_XY_MM && r.y_max <= ENVELOPE_XY_MM,
            "robot Y bounds [{}, {}] exceed the physical envelope [±{} mm]",
            r.y_min,
            r.y_max,
            ENVELOPE_XY_MM
        );
        ensure!(
            r.z_min >= ENVELOPE_Z_MIN_MM && r.z_max <= ENVELOPE_Z_MAX_MM,
            "robot Z bounds [{}, {}] exceed the physical envelope [{}, {}] mm",
            r.z_min,
            r.z_max,
            ENVELOPE_Z_MIN_MM,
            ENVELOPE_Z_MAX_MM
        );
        ensure!(r.baud_rate > 0, "robot.baud_rate must be non-zero");
        ensure!(
            r.feed_rate > 0,
            "robot.feed_rate must be non-zero (F0 would stall every move)"
        );
        let c = &self.conveyor;
        ensure!(c.baud_rate > 0, "conveyor.baud_rate must be non-zero");
        ensure!(
            c.speed_mm_s.is_finite(),
            "conveyor.speed_mm_s must be finite (NaN/inf poisons belt-shift and the planner)"
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
        ensure!(self.camera.fps > 0, "camera.fps must be non-zero");
        if let Some(fourcc) = &self.camera.fourcc {
            ensure!(
                fourcc.len() == 4 && fourcc.is_ascii(),
                "camera.fourcc must be a 4-character ASCII code (e.g. \"MJPG\")"
            );
        }
        let v = &self.vision;
        ensure!(
            v.mm_per_px.is_finite() && v.mm_per_px > 0.0,
            "vision.mm_per_px must be a finite value > 0"
        );
        ensure!(
            (0.0..=255.0).contains(&v.threshold),
            "vision.threshold must be within 0-255"
        );
        ensure!(
            v.min_area.is_finite() && v.min_area >= 0.0 && v.max_area.is_finite(),
            "vision.min_area/max_area must be finite and non-negative"
        );
        ensure!(
            v.min_area < v.max_area,
            "vision.min_area must be < vision.max_area"
        );
        let lg = &self.logging;
        ensure!(
            !lg.directory.trim().is_empty(),
            "logging.directory must not be empty"
        );
        // Catch a typo'd level spec at startup rather than silently logging
        // nothing (or panicking inside the logger).
        ensure!(
            flexi_logger::LogSpecification::parse(&lg.level).is_ok(),
            "logging.level is not a valid RUST_LOG spec (e.g. \"info\" or \
             \"info,deltax2sort=debug\"): {:?}",
            lg.level
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
        // Safety-relevant default: E-stop keeps the part held unless opted in.
        assert!(!cfg.robot.release_gripper_on_estop);
        assert_eq!(cfg.conveyor.speed_mm_s, 100.0);
        assert_eq!(cfg.camera.fps, 30);
        assert_eq!(cfg.camera.fourcc, None);
        assert_eq!(cfg.vision.mm_per_px, 0.5);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_z_pick_outside_z_range() {
        let mut cfg = AppConfig::default();
        cfg.robot.z_pick = -250.0; // below z_min (-200): physically unreachable
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_nonpositive_mm_per_px() {
        let mut cfg = AppConfig::default();
        cfg.vision.mm_per_px = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_drop_position_outside_workspace() {
        let mut cfg = AppConfig::default();
        cfg.sorting.drop_x = 500.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_feed_rate() {
        let mut cfg = AppConfig::default();
        cfg.robot.feed_rate = 0; // would emit F0 and stall every move
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn logging_defaults_are_sane() {
        let lg = AppConfig::default().logging;
        assert_eq!(lg.level, "info");
        assert_eq!(lg.directory, "logs");
        assert!(lg.to_console);
        assert_eq!(lg.keep_days, 7);
    }

    #[test]
    fn validate_rejects_bad_logging_level_and_empty_directory() {
        let mut cfg = AppConfig::default();
        cfg.logging.level = "not a level".to_string();
        assert!(cfg.validate().is_err());

        let mut cfg = AppConfig::default();
        cfg.logging.directory = "   ".to_string();
        assert!(cfg.validate().is_err());

        // A per-module spec (used to trace G-code) stays valid.
        let mut cfg = AppConfig::default();
        cfg.logging.level = "info,deltax2sort=debug".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_baud_rate() {
        let mut cfg = AppConfig::default();
        cfg.robot.baud_rate = 0;
        assert!(cfg.validate().is_err());
        let mut cfg = AppConfig::default();
        cfg.conveyor.baud_rate = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_belt_speed() {
        let mut cfg = AppConfig::default();
        cfg.conveyor.speed_mm_s = f32::NAN;
        assert!(cfg.validate().is_err());
        cfg.conveyor.speed_mm_s = f32::INFINITY;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_min_area() {
        let mut cfg = AppConfig::default();
        cfg.vision.min_area = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_workspace_beyond_physical_envelope() {
        let mut cfg = AppConfig::default();
        cfg.robot.x_max = 1600.0; // typo: 10x the real envelope
        assert!(cfg.validate().is_err());
        let mut cfg = AppConfig::default();
        cfg.robot.z_min = -500.0; // below the -200 mm floor
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
