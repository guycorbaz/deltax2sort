use crate::hardware::Position;
use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub struct CalibrationParams {
    pub scale_x: f32, // mm per pixel
    pub scale_y: f32,
    pub offset_x: f32, // robot x at pixel 0
    pub offset_y: f32, // robot y at pixel 0
    pub rotation: f32, // radians, camera rotation relative to belt axes
}

impl CalibrationParams {
    /// Parameters for a camera whose optical center maps to the robot
    /// origin: pixel (width/2, height/2) -> world (0, 0).
    pub fn centered(width: u32, height: u32, scale_mm_per_px: f32) -> Self {
        Self {
            scale_x: scale_mm_per_px,
            scale_y: scale_mm_per_px,
            offset_x: -(width as f32) * scale_mm_per_px / 2.0,
            offset_y: -(height as f32) * scale_mm_per_px / 2.0,
            rotation: 0.0,
        }
    }
}

// Deliberately NO `Default` impl: calibration depends on the capture
// resolution actually in effect (`CameraDriver::resolution()`), so callers
// must construct params explicitly — e.g. `centered(width, height, mm_per_px)`.

pub struct CoordinateTransformer {
    params: CalibrationParams,
}

impl CoordinateTransformer {
    pub fn new(params: CalibrationParams) -> Self {
        Self { params }
    }

    /// Affine pixel -> robot transform: scale, rotate, translate.
    ///
    /// Z is reported as 0.0 (the belt plane in vision terms); the pick
    /// height comes from `robot.z_pick` in the configuration, applied by
    /// the orchestrator.
    pub fn pixel_to_world(&self, px: f32, py: f32) -> Result<Position> {
        let sx = px * self.params.scale_x;
        let sy = py * self.params.scale_y;
        let (sin_r, cos_r) = self.params.rotation.sin_cos();
        let wx = sx * cos_r - sy * sin_r + self.params.offset_x;
        let wy = sx * sin_r + sy * cos_r + self.params.offset_y;

        Ok(Position {
            x: wx,
            y: wy,
            z: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
    }

    #[test]
    fn image_center_maps_to_robot_origin() {
        let t = CoordinateTransformer::new(CalibrationParams::centered(1280, 720, 0.5));
        let p = t.pixel_to_world(640.0, 360.0).unwrap();
        assert_close(p.x, 0.0);
        assert_close(p.y, 0.0);
    }

    #[test]
    fn scale_is_applied() {
        let t = CoordinateTransformer::new(CalibrationParams::centered(1280, 720, 0.5));
        // 100 px right of center = 50 mm in robot X.
        let p = t.pixel_to_world(740.0, 360.0).unwrap();
        assert_close(p.x, 50.0);
        assert_close(p.y, 0.0);
    }

    #[test]
    fn rotation_is_applied() {
        let params = CalibrationParams {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: std::f32::consts::FRAC_PI_2, // 90 degrees
        };
        let t = CoordinateTransformer::new(params);
        // Pixel +X axis maps to world +Y under a 90 degree rotation.
        let p = t.pixel_to_world(100.0, 0.0).unwrap();
        assert_close(p.x, 0.0);
        assert_close(p.y, 100.0);
    }
}
