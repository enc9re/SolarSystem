import math
import time
import numpy as np
import dearpygui.dearpygui as dpg

G = 6.67408e-11
dt = 10
speedup = 1e3
window_size = 800
center = (window_size // 2, window_size // 2)
sun_mass = 1.98847e30
sun_draw_radius = 10

B = 159.2
C = -1682.192

sun = {
    "name": "Sun",
    "orbit_m": 0,
    "orbit_px": 0,
    "angle": 0,
    "omega": 0,
    "pos": np.array([float(center[0]), float(center[1])], dtype=np.float64),
    "radius": sun_draw_radius,
    "color": (255,215,0,255)
}

planets_data = [
    {"name": "Mercury", "orbit_m": 57.9e9,  "draw_radius": 2, "mass": 3.3011e23, "color": (200,200,200,255)},
    {"name": "Venus",   "orbit_m": 108.2e9, "draw_radius": 3, "mass": 4.8675e24, "color": (255,165,0,255)},
    {"name": "Earth",   "orbit_m": 149.6e9, "draw_radius": 3, "mass": 5.972e24,  "color": (100,149,237,255)},
    {"name": "Mars",    "orbit_m": 227.9e9, "draw_radius": 2, "mass": 6.4171e23, "color": (188,39,50,255)},
    {"name": "Jupiter", "orbit_m": 778.5e9, "draw_radius": 6, "mass": 1.898e27,  "color": (222,184,135,255)},
    {"name": "Saturn",  "orbit_m": 1.433e12,"draw_radius": 5, "mass": 5.683e26,  "color": (210,180,140,255)},
    {"name": "Uranus",  "orbit_m": 2.872e12,"draw_radius": 4, "mass": 8.681e25,  "color": (175,238,238,255)},
    {"name": "Neptune", "orbit_m": 4.495e12,"draw_radius": 4, "mass": 1.024e26,  "color": (72,61,139,255)},
    {"name": "Pluto",   "orbit_m": 5.906e12,"draw_radius": 2, "mass": 1.309e22,  "color": (205,197,191,255)}
]

bodies = [sun]
for planet in planets_data:
    orbit_m = planet["orbit_m"]
    orbit_px = B * math.log10(orbit_m) + C
    omega = math.sqrt(G * sun_mass / (orbit_m**3)) if orbit_m != 0 else 0
    bodies.append({
        "name": planet["name"],
        "orbit_m": orbit_m,
        "orbit_px": orbit_px,
        "angle": 0,
        "omega": omega,
        "pos": np.array([center[0] + orbit_px, center[1]], dtype=np.float64),
        "radius": planet["draw_radius"],
        "color": planet["color"]
    })

dpg.create_context()
with dpg.window(label="Solar System Simulation", width=window_size, height=window_size):
    with dpg.drawlist(width=window_size, height=window_size, tag="drawlist"):
        dpg.draw_circle(center, sun["radius"], color=sun["color"], fill=sun["color"])
dpg.create_viewport(title="Solar System Circular Orbit Simulation", width=window_size, height=window_size)
dpg.setup_dearpygui()
dpg.show_viewport()

last_time = time.time()
while dpg.is_dearpygui_running():
    current_time = time.time()
    steps = int((current_time - last_time) / dt)
    last_time = current_time
    for i in range(1, len(bodies)):
        bodies[i]["angle"] += bodies[i]["omega"] * dt * speedup
        bodies[i]["angle"] %= 2 * math.pi
        bodies[i]["pos"][0] = center[0] + bodies[i]["orbit_px"] * math.cos(bodies[i]["angle"])
        bodies[i]["pos"][1] = center[1] + bodies[i]["orbit_px"] * math.sin(bodies[i]["angle"])
    dpg.delete_item("drawlist", children_only=True)
    for i in range(1, len(bodies)):
        dpg.draw_circle(center, bodies[i]["orbit_px"], parent="drawlist", color=(100,100,100,100))
    for body in bodies:
        dpg.draw_circle((body["pos"][0], body["pos"][1]), body["radius"], parent="drawlist", color=body["color"], fill=body["color"])
    dpg.render_dearpygui_frame()
dpg.destroy_context()