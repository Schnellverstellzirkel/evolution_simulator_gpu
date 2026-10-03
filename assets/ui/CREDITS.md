# Art credits

`tools/ui_assets.py` builds everything in this directory from the sources below.

- Skies (`sky/`): Poly Haven HDRIs, CC0. City: kloofendal_overcast_puresky. Storm: overcast_soil_puresky. Dusk: kloppenheim_01_puresky. https://polyhaven.com
- Materials (`textures/`, and the facades in `skyline/`): ambientCG, CC0. Concrete042A, Concrete034, Concrete031, Concrete047A, PavingStones070, MetalPlates013, MetalPlates006, Ground036, Ground054, Ground023, Rust009, Bricks075A. https://ambientcg.com
- Skyline layers (`skyline/`), sprites (`sprites/`) and the menu backdrop are painted by the script.
- Fonts (`fonts/`, subset to Latin): DejaVu Sans Condensed (Bitstream Vera license, `fonts/DejaVu-LICENSE.txt`) and Barlow Semi Condensed (SIL Open Font License 1.1, `fonts/Barlow-OFL.txt`).

## How evolution works poster

src/schematic.rs draws the poster with egui painter shapes only. The look (warm cream, mustard, brick red and blue, thick dark outlines, sunburst rays, mid-century infographic cards) follows the general style of Team Fortress 2 promotional posters. No Valve art, logos or fonts are used. Titles use Barlow Semi Condensed (OFL, already listed above) and body text DejaVu Sans Condensed.
