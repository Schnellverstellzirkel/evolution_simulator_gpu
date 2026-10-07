//! The game's art: skies, skyline layers, surface materials, sprites and
//! fonts under assets/ui/. `tools/ui_assets.py` builds them from free
//! sources (assets/ui/CREDITS.md). They are compiled into the binary, and
//! a background thread decodes them at start, so no frame waits on a decode
//! unless it asks for an image in the first moments.
use eframe::egui::{self, FontData, FontDefinitions, FontFamily, TextureHandle, TextureId};
use std::sync::{Arc, OnceLock};

/// One image of the art set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Art {
    /// Overcast City 17 sky, 360 degrees around, 48 degrees up.
    SkyCity,
    /// A heavy storm sky.
    SkyStorm,
    /// A low sun breaking through at dusk.
    SkyDusk,
    /// Skyline layers, far to near, each tiling sideways.
    SkylineFar,
    /// Mid skyline layer.
    SkylineMid,
    /// Near skyline layer.
    SkylineNear,
    /// Surface materials, each tiling both ways.
    ConcreteDark,
    ConcreteLight,
    Cobble,
    RustSteel,
    CombinePlate,
    Mud,
    Sand,
    Dirt,
    Rust,
    /// A lit sphere in white, tinted per node.
    Sphere,
    /// A soft round glow in white.
    Glow,
}

/// Every art variant, in enum order.
const ALL: [Art; 17] = [
    Art::SkyCity,
    Art::SkyStorm,
    Art::SkyDusk,
    Art::SkylineFar,
    Art::SkylineMid,
    Art::SkylineNear,
    Art::ConcreteDark,
    Art::ConcreteLight,
    Art::Cobble,
    Art::RustSteel,
    Art::CombinePlate,
    Art::Mud,
    Art::Sand,
    Art::Dirt,
    Art::Rust,
    Art::Sphere,
    Art::Glow,
];

impl Art {
    /// Raw encoded bytes of this image.
    fn bytes(self) -> &'static [u8] {
        match self {
            Art::SkyCity => include_bytes!("../assets/ui/sky/city.jpg"),
            Art::SkyStorm => include_bytes!("../assets/ui/sky/storm.jpg"),
            Art::SkyDusk => include_bytes!("../assets/ui/sky/dusk.jpg"),
            Art::SkylineFar => include_bytes!("../assets/ui/skyline/far.png"),
            Art::SkylineMid => include_bytes!("../assets/ui/skyline/mid.png"),
            Art::SkylineNear => include_bytes!("../assets/ui/skyline/near.png"),
            Art::ConcreteDark => include_bytes!("../assets/ui/textures/concrete_dark.jpg"),
            Art::ConcreteLight => include_bytes!("../assets/ui/textures/concrete_light.jpg"),
            Art::Cobble => include_bytes!("../assets/ui/textures/cobble.jpg"),
            Art::RustSteel => include_bytes!("../assets/ui/textures/rust_steel.jpg"),
            Art::CombinePlate => include_bytes!("../assets/ui/textures/combine_plate.jpg"),
            Art::Mud => include_bytes!("../assets/ui/textures/mud.jpg"),
            Art::Sand => include_bytes!("../assets/ui/textures/sand.jpg"),
            Art::Dirt => include_bytes!("../assets/ui/textures/dirt.jpg"),
            Art::Rust => include_bytes!("../assets/ui/textures/rust.jpg"),
            Art::Sphere => include_bytes!("../assets/ui/sprites/sphere.png"),
            Art::Glow => include_bytes!("../assets/ui/sprites/glow.png"),
        }
    }

    /// Tiling images repeat past their edges; sprites clamp.
    fn tiles(self) -> bool {
        !matches!(self, Art::Sphere | Art::Glow)
    }

    /// Cache slot for this image's texture and size, filled on first access.
    fn slot(self) -> &'static OnceLock<(TextureHandle, [usize; 2])> {
        static SLOTS: [OnceLock<(TextureHandle, [usize; 2])>; ALL.len()] =
            [const { OnceLock::new() }; ALL.len()];
        &SLOTS[self as usize]
    }

    /// Decodes and uploads this image to the GPU, returning its texture and size.
    fn load(self, ctx: &egui::Context) -> (TextureHandle, [usize; 2]) {
        let image = image::load_from_memory(self.bytes())
            .map(|image| image.to_rgba8())
            .unwrap_or_else(|_| image::RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 0])));
        let size = [image.width() as usize, image.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
        let wrap = if self.tiles() {
            egui::TextureWrapMode::Repeat
        } else {
            egui::TextureWrapMode::ClampToEdge
        };
        let options = egui::TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            wrap_mode: wrap,
            mipmap_mode: Some(egui::TextureFilter::Linear),
        };
        let handle = ctx.load_texture(format!("art-{self:?}"), color, options);
        (handle, size)
    }

    /// The texture, decoded and uploaded the first time it is asked for.
    pub fn texture(self, ctx: &egui::Context) -> TextureId {
        self.slot().get_or_init(|| self.load(ctx)).0.id()
    }

    /// The image size in texels.
    pub fn size(self, ctx: &egui::Context) -> egui::Vec2 {
        let size = self.slot().get_or_init(|| self.load(ctx)).1;
        egui::Vec2::new(size[0] as f32, size[1] as f32)
    }
}

/// Decodes every image on a background thread, so the first frames that
/// draw them find them ready.
pub fn preload(ctx: &egui::Context) {
    let ctx = ctx.clone();
    let _ = std::thread::Builder::new()
        .name("art".into())
        .spawn(move || {
            for art in ALL {
                art.texture(&ctx);
            }
            ctx.request_repaint();
        });
}

/// Font families of the game, besides egui's proportional and monospace.
pub fn hud() -> FontFamily {
    FontFamily::Name("hud".into())
}
/// Heavier HUD digits and titles.
pub fn hud_bold() -> FontFamily {
    FontFamily::Name("hud-bold".into())
}
/// Bold labels, like the words on the HUD.
pub fn label_bold() -> FontFamily {
    FontFamily::Name("label-bold".into())
}

/// Installs the fonts: a condensed humanist sans for the interface (the
/// Tahoma look of the Source menus), a DIN-like condensed face for the HUD
/// numbers, and a bold face for HUD labels. egui's own fonts stay as
/// fallbacks for symbols and emoji.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    for (name, bytes) in [
        (
            "sans",
            &include_bytes!("../assets/ui/fonts/DejaVuSansCondensed.ttf")[..],
        ),
        (
            "sans-bold",
            &include_bytes!("../assets/ui/fonts/DejaVuSansCondensed-Bold.ttf")[..],
        ),
        (
            "din",
            &include_bytes!("../assets/ui/fonts/BarlowSemiCondensed-Regular.ttf")[..],
        ),
        (
            "din-bold",
            &include_bytes!("../assets/ui/fonts/BarlowSemiCondensed-SemiBold.ttf")[..],
        ),
    ] {
        fonts
            .font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    }
    let fallbacks = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "sans".to_owned());
    for (family, first) in [
        (hud(), "din"),
        (hud_bold(), "din-bold"),
        (label_bold(), "sans-bold"),
    ] {
        let mut list = vec![first.to_owned()];
        list.extend(fallbacks.iter().cloned());
        fonts.families.insert(family, list);
    }
    ctx.set_fonts(fonts);
}
