//! The game's art under `assets/ui/`: skies, skyline layers, surface
//! materials, sprites and fonts, all compiled into the binary.
//! `tools/ui_assets.py` builds the images from free sources. The fonts are
//! other files that the script does not touch, and `assets/ui/CREDITS.md`
//! lists the sources of both. At start the UI installs the fonts, and a
//! background thread decodes the images, so no frame waits on a decode
//! unless it asks for an image in the first moments.
use eframe::egui::{self, FontData, FontDefinitions, FontFamily, TextureHandle, TextureId};
use std::sync::{Arc, OnceLock};

/// One image of the art set. The skyline layers tile sideways, the surface
/// materials tile both ways, and the two sprites do not tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Art {
    /// Overcast City 17 sky, 360 degrees around, 48 degrees up.
    SkyCity,
    /// A heavy storm sky.
    SkyStorm,
    /// A low sun breaking through at dusk.
    SkyDusk,
    /// The farthest skyline layer.
    SkylineFar,
    /// The middle skyline layer.
    SkylineMid,
    /// The nearest skyline layer.
    SkylineNear,
    /// Dark concrete, the usual body of the ground.
    ConcreteDark,
    /// Light concrete, for the hurdles.
    ConcreteLight,
    /// Cobblestones, the usual crust of the ground.
    Cobble,
    /// Grey steel plate with rust streaks. No scene draws it at present.
    RustSteel,
    /// Dark plate of the Combine wall, for the highest hurdle level.
    CombinePlate,
    /// Mud, for the ground in mud.
    Mud,
    /// Sand, the crust of dry ground.
    Sand,
    /// Dirt, the body of dry ground.
    Dirt,
    /// Heavy brown rust, for the walls of pits.
    Rust,
    /// A lit sphere in white, tinted per node.
    Sphere,
    /// A soft round glow in white.
    Glow,
}

/// Every art variant, in enum order. `slot` sizes its texture cache from the
/// length of this list, so a new variant must be added here too.
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

    /// Whether the texture repeats past its edges. Every image does except the
    /// sprites, which clamp to their edge.
    fn tiles(self) -> bool {
        !matches!(self, Art::Sphere | Art::Glow)
    }

    /// Cache slot for this image's texture and size, filled on first access.
    fn slot(self) -> &'static OnceLock<(TextureHandle, [usize; 2])> {
        static SLOTS: [OnceLock<(TextureHandle, [usize; 2])>; ALL.len()] =
            [const { OnceLock::new() }; ALL.len()];
        &SLOTS[self as usize]
    }

    /// Decodes this image and gives it to egui as a texture, which egui
    /// uploads to the GPU. Returns the texture and the image size in texels.
    /// Bytes that fail to decode become one transparent pixel.
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

    /// The egui texture of this image. The first call, from `preload` or from
    /// a frame, decodes the image and makes the texture. Later calls reuse it.
    pub fn texture(self, ctx: &egui::Context) -> TextureId {
        self.slot().get_or_init(|| self.load(ctx)).0.id()
    }

    /// The image size in texels. Like `texture`, it decodes the image if
    /// nothing has yet.
    pub fn size(self, ctx: &egui::Context) -> egui::Vec2 {
        let size = self.slot().get_or_init(|| self.load(ctx)).1;
        egui::Vec2::new(size[0] as f32, size[1] as f32)
    }
}

/// Decodes every image on a background thread, so the first frames that
/// draw them find them ready. It asks for a repaint when it is done. If the
/// thread does not start, each image decodes when a frame first asks for it.
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

/// The HUD font family: a DIN-like condensed face for the numbers and units
/// on the HUD.
pub fn hud() -> FontFamily {
    FontFamily::Name("hud".into())
}
/// A heavier cut of the HUD face, for large digits and titles.
pub fn hud_bold() -> FontFamily {
    FontFamily::Name("hud-bold".into())
}
/// Bold labels, like the capital words on the HUD.
pub fn label_bold() -> FontFamily {
    FontFamily::Name("label-bold".into())
}

/// Installs the fonts. DejaVu Sans Condensed, a condensed humanist sans like
/// the Tahoma of the Source menus, goes first in egui's proportional family
/// and sets the look of the interface. The `hud` family starts with Barlow
/// Semi Condensed, a DIN-like face for HUD numbers. The `hud_bold` family
/// starts with the SemiBold cut of Barlow Semi Condensed, and the `label_bold`
/// family starts with DejaVu Sans Condensed Bold. Every family keeps egui's
/// own fonts behind its first one, as fallbacks for symbols and emoji.
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
