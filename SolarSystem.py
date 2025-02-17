import math
import time
import numpy as np
import dearpygui.dearpygui as dpg


G = 6.67408e-11
speedup = 1e4
window_size = 800
center = (window_size // 2, window_size // 2)
sun_mass = 1.98847e30
sun_draw_radius = 10
B = 159.2
C = -1682.192
camera_distance = 800
camera_yaw = 0.0
camera_pitch = 0.2
focal_length = 500
zoom_speed = 5
rotate_speed = 0.02
cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
camera_mode = "orbit"
free_move_speed = 300
free_cam_pos = None
free_cam_yaw = 0.0
free_cam_pitch = 0.0
view_mode = "3d"

planet_info = {
    "Mercury": "Mercury is the smallest and closest planet to the Sun. It has a rocky surface and experiences extreme temperature fluctuations.",
    "Venus":   "Venus is the second planet from the Sun with a thick, toxic atmosphere and very high surface temperatures.",
    "Earth":   "Earth is the third planet from the Sun and the only known planet that supports life.",
    "Mars":    "Mars is the red planet, known for its canyons, volcanoes, and signs of ancient water.",
    "Jupiter": "Jupiter is the largest planet in the Solar System, famous for its giant storms and numerous moons.",
    "Saturn":  "Saturn is known for its brilliant rings and beautiful cloud bands.",
    "Uranus":  "Uranus is a planet with a unique sideways rotation and a cold, methane-rich atmosphere.",
    "Neptune": "Neptune is the most distant planet, noted for its deep blue color and strong winds.",
    "Pluto":   "Pluto is a dwarf planet with an eccentric orbit, once considered the ninth planet."
}

sun = {
    "name": "Sun",
    "orbit_m": 0,
    "orbit_px": 0,
    "angle": 0,
    "omega": 0,
    "pos3d": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "radius": sun_draw_radius,
    "color": (255, 215, 0, 255),
    "is_pluto": False
}
planets_data = [
    {"name": "Mercury", "orbit_m": 57.9e9,  "draw_radius": 4,  "mass": 3.3011e23, "color": (200, 200, 200, 255)},
    {"name": "Venus",   "orbit_m": 108.2e9, "draw_radius": 6,  "mass": 4.8675e24, "color": (255, 165, 0, 255)},
    {"name": "Earth",   "orbit_m": 149.6e9, "draw_radius": 6,  "mass": 5.972e24,  "color": (100, 149, 237, 255)},
    {"name": "Mars",    "orbit_m": 227.9e9, "draw_radius": 4,  "mass": 6.4171e23, "color": (188, 39, 50, 255)},
    {"name": "Jupiter", "orbit_m": 778.5e9, "draw_radius": 10, "mass": 1.898e27,  "color": (222, 184, 135, 255)},
    {"name": "Saturn",  "orbit_m": 1.433e12,"draw_radius": 9,  "mass": 5.683e26,  "color": (210, 180, 140, 255)},
    {"name": "Uranus",  "orbit_m": 2.872e12,"draw_radius": 7,  "mass": 8.681e25,  "color": (175, 238, 238, 255)},
    {"name": "Neptune", "orbit_m": 4.495e12,"draw_radius": 7,  "mass": 1.024e26,  "color": (72, 61, 139, 255)},
    {"name": "Pluto",   "orbit_m": 5.906e12,"draw_radius": 3,  "mass": 1.309e22,  "color": (205, 197, 191, 255)}
]
bodies = [sun]
for planet in planets_data:
    orbit_m = planet["orbit_m"]
    orbit_px = B * math.log10(orbit_m) + C
    omega = math.sqrt(G * sun_mass / (orbit_m ** 3))
    is_pluto = (planet["name"] == "Pluto")
    bodies.append({
        "name": planet["name"],
        "orbit_m": orbit_m,
        "orbit_px": orbit_px,
        "angle": 0,
        "omega": omega,
        "pos3d": np.array([orbit_px, 0.0, 0.0], dtype=np.float64),
        "radius": planet["draw_radius"],
        "color": planet["color"],
        "is_pluto": is_pluto
    })

def get_camera_position_orbit():
    """Calculate camera position for orbit mode."""
    cam_x = cam_target[0] + camera_distance * math.cos(camera_pitch) * math.cos(camera_yaw)
    cam_y = cam_target[1] + camera_distance * math.sin(camera_pitch)
    cam_z = cam_target[2] + camera_distance * math.cos(camera_pitch) * math.sin(camera_yaw)
    return np.array([cam_x, cam_y, cam_z], dtype=np.float64)

def compute_camera_basis(cam_pos, target):
    """
    Compute the camera coordinate basis vectors (right, up, forward)
    based on the camera position and target.
    """
    forward = target - cam_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0:
        forward_norm = 1
    forward = forward / forward_norm
    world_up = np.array([0, 1, 0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        right_norm = 1
    right = right / right_norm
    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        up_norm = 1
    up = up / up_norm
    return right, up, forward

def project(cam_coord):
    """
    Simple perspective projection from 3D camera coordinates to 2D screen coordinates.
    """
    if cam_coord[2] <= 0:
        return (-1000, -1000)
    x = (cam_coord[0] * focal_length / cam_coord[2]) + center[0]
    y = (-cam_coord[1] * focal_length / cam_coord[2]) + center[1]
    return (x, y)

dpg.create_context()
# Position the info window at the center top of the screen.
# Calculated as (center_x - width/2, top margin)
info_window_width = 350
info_window_height = 250
top_margin = 20
info_pos = (center[0] - info_window_width // 2, top_margin)
with dpg.window(label="Planet Info", tag="info_window", pos=info_pos, width=info_window_width, height=info_window_height, show=False):
    dpg.add_text("", tag="info_text")
with dpg.window(label="Real 3D Solar System", tag="main_window", width=window_size, height=window_size):
    with dpg.drawlist(width=window_size, height=window_size, tag="drawlist"):
        pass
dpg.set_primary_window("main_window", True)
dpg.create_viewport(title="Real 3D Solar System", width=window_size, height=window_size)
dpg.setup_dearpygui()
dpg.show_viewport()

last_time = time.time()
orbit_resolution = 100

if free_cam_pos is None:
    free_cam_pos = get_camera_position_orbit()
    free_cam_yaw = camera_yaw
    free_cam_pitch = camera_pitch

while dpg.is_dearpygui_running():
    current_time = time.time()
    dt_frame = current_time - last_time
    last_time = current_time

    if dpg.is_key_pressed(dpg.mvKey_T):
        view_mode = "2d" if view_mode == "3d" else "3d"
    if dpg.is_key_pressed(dpg.mvKey_C):
        if camera_mode == "orbit":
            camera_mode = "free"
            free_cam_pos = get_camera_position_orbit()
            free_cam_yaw = camera_yaw
            free_cam_pitch = camera_pitch
        else:
            camera_mode = "orbit"
            diff = free_cam_pos - cam_target
            camera_distance = np.linalg.norm(diff)
            camera_pitch = math.asin(diff[1] / camera_distance) if camera_distance != 0 else 0
            camera_yaw = math.atan2(diff[2], diff[0])
    if dpg.is_key_down(dpg.mvKey_Plus):
        speedup *= 1.01
    if dpg.is_key_down(dpg.mvKey_Minus):
        speedup *= 0.99

    for key, index in [(dpg.mvKey_1, 1), (dpg.mvKey_2, 2), (dpg.mvKey_3, 3),
                       (dpg.mvKey_4, 4), (dpg.mvKey_5, 5), (dpg.mvKey_6, 6),
                       (dpg.mvKey_7, 7), (dpg.mvKey_8, 8), (dpg.mvKey_9, 9)]:
        if dpg.is_key_down(key):
            if index < len(bodies):
                cam_target = bodies[index]["pos3d"].copy()
                if camera_mode == "orbit":
                    camera_distance = 100
                info = ("Name: " + bodies[index]["name"] + "\nOrbit (m): " + 
                        f"{bodies[index]['orbit_m']:.2e}" + "\nAngular Speed: " + 
                        f"{bodies[index]['omega']:.2e}" + " rad/s\n\n" + 
                        planet_info.get(bodies[index]["name"], ""))
                dpg.configure_item("info_text", default_value=info)
                dpg.configure_item("info_window", show=True)

    if view_mode == "2d":
        cam_pos = np.array([cam_target[0], cam_target[1] + camera_distance, cam_target[2]], dtype=np.float64)
        right = np.array([1, 0, 0], dtype=np.float64)
        up = np.array([0, 0, -1], dtype=np.float64)
        forward_vec = np.array([0, -1, 0], dtype=np.float64)
        target = cam_target
    elif camera_mode == "orbit":
        if dpg.is_key_down(dpg.mvKey_W):
            camera_distance = max(50, camera_distance - zoom_speed)
        if dpg.is_key_down(dpg.mvKey_S):
            camera_distance += zoom_speed
        if dpg.is_key_down(dpg.mvKey_Left):
            camera_yaw -= rotate_speed
        if dpg.is_key_down(dpg.mvKey_Right):
            camera_yaw += rotate_speed
        if dpg.is_key_down(dpg.mvKey_Up):
            camera_pitch += rotate_speed
            if camera_pitch > math.pi/2 - 0.1:
                camera_pitch = math.pi/2 - 0.1
        if dpg.is_key_down(dpg.mvKey_Down):
            camera_pitch -= rotate_speed
            if camera_pitch < -math.pi/2 + 0.1:
                camera_pitch = -math.pi/2 + 0.1
        cam_pos = get_camera_position_orbit()
        target = cam_target
        right, up, forward_vec = compute_camera_basis(cam_pos, target)
    else:
        forward = np.array([math.cos(free_cam_pitch) * math.cos(free_cam_yaw),
                            math.sin(free_cam_pitch),
                            math.cos(free_cam_pitch) * math.sin(free_cam_yaw)], dtype=np.float64)
        world_up = np.array([0, 1, 0], dtype=np.float64)
        right_vector = np.cross(forward, world_up)
        rn = np.linalg.norm(right_vector)
        if rn != 0:
            right_vector /= rn
        if dpg.is_key_down(dpg.mvKey_W):
            free_cam_pos += forward * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_S):
            free_cam_pos -= forward * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_A):
            free_cam_pos -= right_vector * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_D):
            free_cam_pos += right_vector * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_U):
            free_cam_pos += world_up * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_O):
            free_cam_pos -= world_up * free_move_speed * dt_frame
        if dpg.is_key_down(dpg.mvKey_Left):
            free_cam_yaw -= rotate_speed
        if dpg.is_key_down(dpg.mvKey_Right):
            free_cam_yaw += rotate_speed
        if dpg.is_key_down(dpg.mvKey_Up):
            free_cam_pitch += rotate_speed
            if free_cam_pitch > math.pi/2 - 0.1:
                free_cam_pitch = math.pi/2 - 0.1
        if dpg.is_key_down(dpg.mvKey_Down):
            free_cam_pitch -= rotate_speed
            if free_cam_pitch < -math.pi/2 + 0.1:
                free_cam_pitch = -math.pi/2 + 0.1
        cam_pos = free_cam_pos
        target = free_cam_pos + np.array([
            math.cos(free_cam_pitch) * math.cos(free_cam_yaw),
            math.sin(free_cam_pitch),
            math.cos(free_cam_pitch) * math.sin(free_cam_yaw)
        ], dtype=np.float64)
        right, up, forward_vec = compute_camera_basis(cam_pos, target)

    for i in range(1, len(bodies)):
        body = bodies[i]
        body["angle"] = (body["angle"] + body["omega"] * dt_frame * speedup) % (2 * math.pi)
        r = body["orbit_px"]
        pos = np.array([r * math.cos(body["angle"]), 0, r * math.sin(body["angle"])], dtype=np.float64)
        if body["is_pluto"]:
            tilt = math.radians(17)
            transform = np.array([[1, 0, 0],
                                  [0, math.cos(tilt), -math.sin(tilt)],
                                  [0, math.sin(tilt),  math.cos(tilt)]], dtype=np.float64)
            pos = np.dot(transform, pos)
        bodies[i]["pos3d"] = pos

    dpg.delete_item("drawlist", children_only=True)
    for i in range(1, len(bodies)):
        body = bodies[i]
        pts = []
        for j in range(orbit_resolution + 1):
            ang = 2 * math.pi * j / orbit_resolution
            r = body["orbit_px"]
            pos_orbit = np.array([r * math.cos(ang), 0, r * math.sin(ang)], dtype=np.float64)
            if body["is_pluto"]:
                tilt = math.radians(17)
                transform = np.array([[1, 0, 0],
                                      [0, math.cos(tilt), -math.sin(tilt)],
                                      [0, math.sin(tilt),  math.cos(tilt)]], dtype=np.float64)
                pos_orbit = np.dot(transform, pos_orbit)
            cam_coord = np.array([np.dot(pos_orbit - cam_pos, right),
                                  np.dot(pos_orbit - cam_pos, up),
                                  np.dot(pos_orbit - cam_pos, forward_vec)])
            pts.append(project(cam_coord))
        dpg.draw_polyline(pts, color=(100, 100, 100, 100), parent="drawlist")

    positions_2d = []
    cam_coord_sun = np.array([np.dot(sun["pos3d"] - cam_pos, right),
                               np.dot(sun["pos3d"] - cam_pos, up),
                               np.dot(sun["pos3d"] - cam_pos, forward_vec)])
    sun_proj = project(cam_coord_sun)
    positions_2d.append(sun_proj)
    dpg.draw_circle(sun_proj, max(sun["radius"], 3), parent="drawlist", color=sun["color"], fill=sun["color"])

    for body in bodies[1:]:
        cam_coord = np.array([np.dot(body["pos3d"] - cam_pos, right),
                              np.dot(body["pos3d"] - cam_pos, up),
                              np.dot(body["pos3d"] - cam_pos, forward_vec)])
        proj_coord = project(cam_coord)
        positions_2d.append(proj_coord)
        fixed_radius = body["radius"]
        dpg.draw_circle(proj_coord, fixed_radius, parent="drawlist", color=body["color"], fill=body["color"])

    if dpg.is_mouse_button_down(0):
        mouse_pos = dpg.get_mouse_pos(local=True)
        for i, body in enumerate(bodies):
            proj_coord = positions_2d[i]
            dx = mouse_pos[0] - proj_coord[0]
            dy = mouse_pos[1] - proj_coord[1]
            if math.sqrt(dx * dx + dy * dy) < body["radius"] + 5:
                info = ("Name: " + body["name"] +
                        "\nOrbit (m): " + f"{body['orbit_m']:.2e}" +
                        "\nAngular Speed: " + f"{body['omega']:.2e}" + " rad/s\n\n" +
                        planet_info.get(body["name"], ""))
                dpg.configure_item("info_text", default_value=info)
                dpg.configure_item("info_window", show=True)
                break

    dpg.render_dearpygui_frame()

dpg.destroy_context()

if __name__ == "__main__":
    pass