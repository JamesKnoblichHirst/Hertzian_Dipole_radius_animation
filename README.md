# Hertzian Dipole Field Animator

This project generates an animated 3D visualization of the electric field produced by a Hertzian dipole over a spherical grid.

The animation can be previewed live or saved as a `.gif` file.

---

##  How to Run

First, make sure you have the required Python packages installed:

```bash
pip install numpy scipy matplotlib
```

Then, you can run the script:
Preview only (no file saved)

```bash
python Hertz_anim.py
```

Save the animation to a GIF

```bash
python Hertz_anim.py --save_path outputs/field_animation.gif
```

Command Line Options
Option	Description	Default
--save_path	File path to save the output GIF. If not set, displays the animation live.	None (show only)
--interval	Animation frame interval in milliseconds (smaller = faster).	100
--points	Number of radial points to simulate.	30

Example with custom settings:
```bash
python myfile.py --save_path outputs/my_animation.gif --interval 50 --points 60
```
