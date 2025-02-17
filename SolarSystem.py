import math
import time
import random
import sys
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
import dearpygui.dearpygui as dpg

G = 6.67408e-11
speedup = 1e4
initial_width = 1200
initial_height = 800
focal_length = 500
B = 159.2
C = -1682.192
sun_mass = 1.98847e30
sun_draw_radius = 10
camera_distance = 800
camera_yaw = 0.0
camera_pitch = 0.2
cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
camera_mode = "orbit"  
view_mode = "3d"     
free_move_speed = 300
free_cam_pos = None
free_cam_yaw = 0.0
free_cam_pitch = 0.0
mouse_wheel_delta = 0.0
is_fullscreen = False
selected_planet = None
tracked_planet = None
simulation_time = 0
time_scale = 1.0
is_paused = False
show_predictions = False
current_date = datetime.now()
prediction_length = 1000  
speed_multiplier = 1.0  
planet_info = {
    "Sun": {
        "description": "The Sun is the star at the center of our Solar System",
        "mass": "1.989 × 10^30 kg",
        "diameter": "1,392,700 km",
        "surface_temp": "5,500°C",
        "rotation_period": "27 Earth days",
        "age": "4.6 billion years",
        "composition": "Hydrogen (73%), Helium (25%)",
    },
    "Mercury": {
        "description": "Mercury is the smallest and closest planet to the Sun",
        "mass": "3.285 × 10^23 kg",
        "diameter": "4,879 km",
        "orbital_period": "88 Earth days",
        "rotation_period": "59 Earth days",
        "average_temp": "167°C",
        "atmosphere": "Minimal",
    },
    "Venus": {
        "description": "Venus is the second planet from the Sun",
        "mass": "4.867 × 10^24 kg",
        "diameter": "12,104 km",
        "orbital_period": "225 Earth days",
        "rotation_period": "243 Earth days",
        "average_temp": "462°C",
        "atmosphere": "96% Carbon dioxide",
    },
    "Earth": {
        "description": "Earth is the third planet from the Sun",
        "mass": "5.972 × 10^24 kg",
        "diameter": "12,742 km",
        "orbital_period": "365.25 days",
        "rotation_period": "24 hours",
        "average_temp": "15°C",
        "atmosphere": "78% Nitrogen, 21% Oxygen",
        "moons": ["Moon"],
    },
    "Mars": {
        "description": "Mars is the red planet",
        "mass": "6.39 × 10^23 kg",
        "diameter": "6,779 km",
        "orbital_period": "687 Earth days",
        "rotation_period": "24h 37m",
        "average_temp": "-63°C",
        "atmosphere": "95% Carbon dioxide",
    },
    "Jupiter": {
        "description": "Jupiter is the largest planet in the Solar System",
        "mass": "1.898 × 10^27 kg",
        "diameter": "139,820 km",
        "orbital_period": "11.9 Earth years",
        "rotation_period": "9h 56m",
        "average_temp": "-110°C",
        "atmosphere": "90% Hydrogen",
    },
    "Saturn": {
        "description": "Saturn is known for its brilliant rings",
        "mass": "5.683 × 10^26 kg",
        "diameter": "116,460 km",
        "orbital_period": "29.5 Earth years",
        "rotation_period": "10h 42m",
        "average_temp": "-140°C",
        "atmosphere": "96% Hydrogen",
    },
    "Uranus": {
        "description": "Uranus is a planet with a unique sideways rotation",
        "mass": "8.681 × 10^25 kg",
        "diameter": "50,724 km",
        "orbital_period": "84 Earth years",
        "rotation_period": "17h 14m",
        "average_temp": "-195°C",
        "atmosphere": "83% Hydrogen",
    },
    "Neptune": {
        "description": "Neptune is the most distant planet",
        "mass": "1.024 × 10^26 kg",
        "diameter": "49,244 km",
        "orbital_period": "165 Earth years",
        "rotation_period": "16h 6m",
        "average_temp": "-200°C",
        "atmosphere": "80% Hydrogen",
    },
    "Pluto": {
        "description": "Pluto is a dwarf planet with an eccentric orbit",
        "mass": "1.309 × 10^22 kg",
        "diameter": "2,377 km",
        "orbital_period": "248 Earth years",
        "rotation_period": "6.4 Earth days",
        "average_temp": "-230°C",
        "atmosphere": "Nitrogen, Methane",
    }
}

moon_data = {
    "name": "Moon",
    "orbit_m": 384400e3,  
    "draw_radius": 2,
    "mass": 7.34767309e22,
    "color": (200, 200, 200, 255),
    "period": 27.322,
    "angle": 0,
    "omega": 2 * math.pi / (27.322 * 24 * 3600),  
    "parent": "Earth",
    "pos3d": np.zeros(3),
    "orbit_scale": 20  
}

class AsteroidBelt:
    def __init__(self):
        self.inner_radius = 329.12e9 
        self.outer_radius = 478.71e9  
        self.num_asteroids = 300
        self.asteroids = []
        self.belt_scale = 5 
        self.generate_asteroids()

    def generate_asteroids(self):
        for _ in range(self.num_asteroids):
            radius = random.uniform(self.inner_radius, self.outer_radius)
            angle = random.uniform(0, 2 * math.pi)
            inclination = random.gauss(0, 0.15)  
            self.asteroids.append({
                'radius': radius,
                'angle': angle,
                'inclination': inclination,
                'speed': math.sqrt(G * sun_mass / (radius**3)),
                'size': random.uniform(0.5, 1.5),
                'color': (169, 169, 169, random.randint(150, 200))
            })

    def update(self, dt):
        if not is_paused:
            for asteroid in self.asteroids:
                asteroid['angle'] += asteroid['speed'] * dt
                if asteroid['angle'] > 2 * math.pi:
                    asteroid['angle'] -= 2 * math.pi

    def get_positions(self):
        positions = []
        for asteroid in self.asteroids:
            r = (B * math.log10(asteroid['radius']) + C) * self.belt_scale
            angle = asteroid['angle']
            inclination = asteroid['inclination']
            x = r * math.cos(angle)
            y = r * math.sin(angle) * math.sin(inclination)
            z = r * math.sin(angle) * math.cos(inclination)
            positions.append((np.array([x, y, z]), asteroid['size'], asteroid['color']))
        return positions

asteroid_belt = AsteroidBelt()

sun = {
    "name": "Sun",
    "orbit_m": 0,
    "orbit_px": 0,
    "angle": 0,
    "omega": 0,
    "pos3d": np.array([0,0,0], dtype=np.float64),
    "radius": sun_draw_radius,
    "color": (255,215,0,255),
    "is_pluto": False,
    "draw_details": None,
}

def calculate_moon_position(moon, parent_pos):
    scaled_orbit = moon["orbit_m"] * moon["orbit_scale"] / 1e6
    x = parent_pos[0] + scaled_orbit * math.cos(moon["angle"])
    y = parent_pos[1] + scaled_orbit * math.sin(moon["angle"]) * 0.2
    z = parent_pos[2] + scaled_orbit * math.sin(moon["angle"])
    return np.array([x, y, z], dtype=np.float64)

def calculate_planet_position(planet, angle):
    rdist = planet["orbit_px"]
    if planet["is_pluto"]:
        inclination = math.radians(17)
        x = rdist * math.cos(angle)
        y = rdist * math.sin(angle) * math.sin(inclination)
        z = rdist * math.sin(angle) * math.cos(inclination)
        return np.array([x, y, z], dtype=np.float64)
    else:
        return np.array([rdist * math.cos(angle), 0, rdist * math.sin(angle)], dtype=np.float64)

def draw_details_earth(center_xy, radius):
    dpg.draw_circle(center=(center_xy[0] + radius*0.3, center_xy[1] - radius*0.1),
                   radius=radius*0.3, color=(34,139,34,255), fill=(34,139,34,255))
    dpg.draw_circle(center=(center_xy[0] - radius*0.2, center_xy[1] + radius*0.2),
                   radius=radius*0.2, color=(34,139,34,255), fill=(34,139,34,255))

def draw_details_saturn(center_xy, radius):
    cx, cy = center_xy
    ring_color = (200,200,200,100)
    rx_outer = radius * 2.0
    ry_outer = rx_outer * 0.6
    pmin_outer = (cx - rx_outer, cy - ry_outer)
    pmax_outer = (cx + rx_outer, cy + ry_outer)
    dpg.draw_ellipse(pmin_outer, pmax_outer, color=ring_color, thickness=2)
    rx_inner = radius * 1.5
    ry_inner = rx_inner * 0.6
    pmin_inner = (cx - rx_inner, cy - ry_inner)
    pmax_inner = (cx + rx_inner, cy + ry_inner)
    dpg.draw_ellipse(pmin_inner, pmax_inner, color=ring_color, thickness=2)

planets_data = [
    {"name":"Mercury","orbit_m":57.9e9,"draw_radius":4,"mass":3.3011e23,"color":(200,200,200,255)},
    {"name":"Venus","orbit_m":108.2e9,"draw_radius":6,"mass":4.8675e24,"color":(255,165,0,255)},
    {"name":"Earth","orbit_m":149.6e9,"draw_radius":6,"mass":5.972e24,"color":(100,149,237,255)},
    {"name":"Mars","orbit_m":227.9e9,"draw_radius":4,"mass":6.4171e23,"color":(188,39,50,255)},
    {"name":"Jupiter","orbit_m":778.5e9,"draw_radius":10,"mass":1.898e27,"color":(222,184,135,255)},
    {"name":"Saturn","orbit_m":1.433e12,"draw_radius":9,"mass":5.683e26,"color":(210,180,140,255)},
    {"name":"Uranus","orbit_m":2.872e12,"draw_radius":7,"mass":8.681e25,"color":(175,238,238,255)},
    {"name":"Neptune","orbit_m":4.495e12,"draw_radius":7,"mass":1.024e26,"color":(72,61,139,255)},
    {"name":"Pluto","orbit_m":5.906e12,"draw_radius":3,"mass":1.309e22,"color":(205,197,191,255)},
]
bodies = [sun]
for planet in planets_data:
    orbit_m = planet["orbit_m"]
    orbit_px = B * math.log10(orbit_m) + C
    omega = math.sqrt(sun_mass/(orbit_m**3) * G)
    details_func = None
    if planet["name"]=="Earth":
        details_func = draw_details_earth
    elif planet["name"]=="Saturn":
        details_func = draw_details_saturn

    is_pluto = (planet["name"]=="Pluto")
    bodies.append({
        "name": planet["name"],
        "orbit_m": orbit_m,
        "orbit_px": orbit_px,
        "angle": 0,
        "omega": omega,
        "pos3d": np.array([orbit_px,0,0],dtype=np.float64),
        "radius": planet["draw_radius"],
        "color": planet["color"],
        "is_pluto": is_pluto,
        "draw_details": details_func
    })


star_count = random.randint(200,400)
stars = []
star_radius = 1
max_star_dist = 50000  

for _ in range(star_count):
    x = random.uniform(-max_star_dist, max_star_dist)
    y = random.uniform(-max_star_dist, max_star_dist)
    z = random.uniform(-max_star_dist, max_star_dist)
    brightness = random.randint(100, 255)
    stars.append({
        "pos": np.array([x, y, z], dtype=np.float64),
        "color": (brightness, brightness, brightness, brightness),
        "size": random.uniform(0.5, 1.5)
    })

def format_time_info(simulation_time):
    years = int(simulation_time / (365.25 * 24 * 3600))
    remaining = simulation_time % (365.25 * 24 * 3600)
    days = int(remaining / (24 * 3600))
    remaining = remaining % (24 * 3600)
    hours = int(remaining / 3600)
    minutes = int((remaining % 3600) / 60)
    
    date = current_date + timedelta(seconds=simulation_time)
    
    return (f"Date: {date.strftime('%Y-%m-%d %H:%M')}\n"
            f"Simulation: {years}y {days}d {hours}h {minutes}m\n"
            f"Speed: {speed_multiplier:.1f}x")

def calculate_orbit_prediction(body, steps=100):
    predictions = []
    current_angle = body["angle"]
    step_size = prediction_length / steps
    
    for i in range(steps):
        future_angle = current_angle + body["omega"] * speedup * speed_multiplier * i * step_size
        if body.get("is_pluto", False):
            predictions.append(calculate_planet_position(body, future_angle))
        else:
            rdist = body["orbit_px"]
            x = rdist * math.cos(future_angle)
            y = 0
            z = rdist * math.sin(future_angle)
            predictions.append(np.array([x, y, z], dtype=np.float64))
    return predictions

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    dpg.toggle_viewport_fullscreen()
    
    if is_fullscreen:
        dpg.configure_item("interface_region", show=False)
        dpg.configure_item("fullscreen_info", show=True)
        dpg.configure_item("time_control_window", show=True)
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item("draw_region", width=w)
        dpg.configure_item("drawlist", width=w)
    else:
        dpg.configure_item("interface_region", show=True)
        dpg.configure_item("fullscreen_info", show=False)
        dpg.configure_item("time_control_window", show=False)
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        half_w = int(w * 0.5)
        dpg.configure_item("draw_region", width=half_w)
        dpg.configure_item("interface_region", width=w-half_w)
        dpg.configure_item("drawlist", width=half_w)

def mouse_wheel_callback(sender, app_data):
    global mouse_wheel_delta
    mouse_wheel_delta += app_data

def get_camera_position_orbit():
    cx = cam_target[0] + camera_distance*math.cos(camera_pitch)*math.cos(camera_yaw)
    cy = cam_target[1] + camera_distance*math.sin(camera_pitch)
    cz = cam_target[2] + camera_distance*math.cos(camera_pitch)*math.sin(camera_yaw)
    return np.array([cx,cy,cz],dtype=np.float64)

def compute_camera_basis(cam_pos, target):
    forward = target - cam_pos
    fn=np.linalg.norm(forward)
    if fn==0:
        fn=1
    forward/=fn
    world_up=np.array([0,1,0],dtype=np.float64)
    right=np.cross(world_up, forward)
    rn=np.linalg.norm(right)
    if rn==0:
        rn=1
    right/=rn
    up=np.cross(forward,right)
    un=np.linalg.norm(up)
    if un==0:
        un=1
    up/=un
    return right, up, forward

def project(cam_coord, center_x, center_y):
    if cam_coord[2]<=0:
        return (-10000,-10000)
    x=(cam_coord[0]*focal_length/cam_coord[2]) + center_x
    y=-(cam_coord[1]*focal_length/cam_coord[2]) + center_y
    return (x,y)

def focus_on_body(index):
    global cam_target, camera_distance, camera_mode, selected_planet, tracked_planet
    if index<len(bodies):
        bd=bodies[index]
        cam_target = bd["pos3d"].copy()
        camera_distance=100
        camera_mode="orbit"
        selected_planet = bd
        tracked_planet = bd
        info_str = format_body_info(bd)
        dpg.set_value("info_text", info_str)
        if is_fullscreen:
            dpg.set_value("fullscreen_info_text", info_str)

def format_body_info(body):
    info = planet_info.get(body["name"], {})
    base_info = (f"Name: {body['name']}\n"
                f"Orbit(m): {body['orbit_m']:.2e}\n"
                f"Omega: {body['omega']:.2e} rad/s\n\n")
    
    detailed_info = ""
    for key, value in info.items():
        if key != "description":
            detailed_info += f"{key}: {value}\n"
    
    return base_info + detailed_info

def handle_number_key(key_num):
    if key_num < len(bodies):
        focus_on_body(key_num)

def adjust_speed(sender, app_data, user_data):
    global speed_multiplier
    if user_data == "increase":
        speed_multiplier *= 1.5
    else:
        speed_multiplier /= 1.5
    speed_multiplier = max(0.1, min(1000, speed_multiplier))
    dpg.set_value("speed_text", f"Speed: {speed_multiplier:.1f}x")

def resize_viewport_callback(sender, app_data):
    try:
        if isinstance(app_data, tuple) and len(app_data) == 2:
            new_w, new_h = app_data
        else:
            new_w = dpg.get_viewport_width()
            new_h = dpg.get_viewport_height()
            
        if new_w < 300 or new_h < 300:
            return
            
        dpg.configure_item("main_window", width=new_w, height=new_h)
        
        if is_fullscreen:
            dpg.configure_item("draw_region", width=new_w, height=new_h)
            dpg.configure_item("drawlist", width=new_w, height=new_h)
            dpg.configure_item("fullscreen_info", pos=(10, 10))
            dpg.configure_item("time_control_window", pos=(10, new_h - 140))
        else:
            half_w = int(new_w * 0.5)
            dpg.configure_item("draw_region", width=half_w, height=new_h)
            dpg.configure_item("interface_region", pos=(half_w,0),
                              width=new_w-half_w, height=new_h)
            dpg.configure_item("drawlist", width=half_w, height=new_h)
    except Exception as e:
        logger.error(f"Error in resize callback: {e}")

def create_time_control_window():
    with dpg.window(tag="time_control_window",
                   pos=(10, initial_height - 140),
                   no_title_bar=True,
                   no_resize=True,
                   no_move=True,
                   width=250,
                   height=130,
                   show=False):
        dpg.add_text("", tag="time_info")
        with dpg.group(horizontal=True):
            dpg.add_button(label="-", callback=adjust_speed, user_data="decrease", width=30)
            dpg.add_text("Speed: 1.0x", tag="speed_text")
            dpg.add_button(label="+", callback=adjust_speed, user_data="increase", width=30)
        dpg.add_checkbox(label="Pause", default_value=is_paused, tag="pause_checkbox")
        dpg.add_checkbox(label="Show Predictions", default_value=show_predictions,
                        tag="predictions_checkbox")
        dpg.add_slider_int(label="Prediction Length", default_value=prediction_length,
                          min_value=100, max_value=5000, tag="prediction_length")
        
        
def main():
    global camera_mode, camera_distance, camera_yaw, camera_pitch
    global cam_target, view_mode, selected_planet, tracked_planet
    global free_cam_pos, free_cam_yaw, free_cam_pitch
    global speedup, mouse_wheel_delta, simulation_time, time_scale
    global is_paused, show_predictions, prediction_length, speed_multiplier

    dpg.create_context()

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (60,60,60), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text,(220,220,220), category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)

    with dpg.window(tag="main_window",
                    no_move=True, no_close=True, no_collapse=True,
                    width=initial_width, height=initial_height):
        with dpg.child_window(tag="draw_region",
                              width=int(initial_width*0.5),
                              height=initial_height,
                              no_scrollbar=True):
            with dpg.drawlist(width=int(initial_width*0.5), height=initial_height, tag="drawlist"):
                pass
        
        with dpg.child_window(tag="interface_region",
                              pos=(int(initial_width*0.5),0),
                              width=int(initial_width*0.5),
                              height=initial_height,
                              no_scrollbar=True):
            dpg.add_text("Solar System Demo", bullet=True)
            dpg.add_separator()
            dpg.add_text("Keys:\n [T] => toggle 2D/3D\n [C] => orbit/free\n [W,S,A,D,U,O] => move free\n"
                        " [Arrows] => rotate\n [+/-] => time\n [Wheel] => zoom\n [F11] => fullscreen\n"
                        " [1-9] => select & track planet\n [X] => disable tracking")
            dpg.add_separator()
            for i, bd in enumerate(bodies):
                dpg.add_button(label=f"{i}. {bd['name']}", width=120,
                               callback=lambda s,a,u: focus_on_body(u),
                               user_data=i)
            dpg.add_separator()
            dpg.add_text("", tag="info_text", wrap=300)

    with dpg.window(tag="fullscreen_info", 
                   pos=(10, 10), 
                   no_title_bar=True,
                   no_resize=True,
                   no_move=True,
                   width=300,
                   height=200,
                   show=False):
        dpg.add_text("", tag="fullscreen_info_text", wrap=280)

    create_time_control_window()

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=mouse_wheel_callback)

    dpg.set_primary_window("main_window", True)
    dpg.create_viewport(title="Solar System",
                        width=initial_width,
                        height=initial_height,
                        resizable=True)
    dpg.set_viewport_resize_callback(resize_viewport_callback)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    if free_cam_pos is None:
        free_cam_pos = get_camera_position_orbit()
        free_cam_yaw = camera_yaw
        free_cam_pitch = camera_pitch

    last_time=time.time()

    try:
        while dpg.is_dearpygui_running():
            current_time=time.time()
            dt_frame=current_time - last_time
            last_time=current_time

            if dpg.is_key_pressed(dpg.mvKey_F11):
                toggle_fullscreen()

            if dpg.is_key_pressed(dpg.mvKey_X):
                tracked_planet = None
                if is_fullscreen:
                    dpg.set_value("fullscreen_info_text", "Tracking disabled")

            for i in range(10):
                if dpg.is_key_pressed(getattr(dpg, f"mvKey_{i}")):
                    handle_number_key(i)

            if tracked_planet:
                cam_target = tracked_planet["pos3d"].copy()

            is_paused = dpg.get_value("pause_checkbox")
            show_predictions = dpg.get_value("predictions_checkbox")
            prediction_length = dpg.get_value("prediction_length")

            if not is_paused:
                simulation_time += dt_frame * speedup * speed_multiplier
                dpg.set_value("time_info", format_time_info(simulation_time))

            if dpg.is_key_pressed(dpg.mvKey_T):
                view_mode="2d" if view_mode=="3d" else "3d"
            if dpg.is_key_pressed(dpg.mvKey_C):
                if camera_mode=="orbit":
                    camera_mode="free"
                    free_cam_pos = get_camera_position_orbit()
                    free_cam_yaw = camera_yaw
                    free_cam_pitch= camera_pitch
                else:
                    camera_mode="orbit"
                    diff=free_cam_pos - cam_target
                    camera_distance=np.linalg.norm(diff)
                    if camera_distance!=0:
                        camera_pitch=math.asin(diff[1]/camera_distance)
                    else:
                        camera_pitch=0
                    camera_yaw=math.atan2(diff[2], diff[0])

            if dpg.is_key_down(dpg.mvKey_Plus):
                speed_multiplier *= 1.01
                dpg.set_value("speed_text", f"Speed: {speed_multiplier:.1f}x")
            if dpg.is_key_down(dpg.mvKey_Minus):
                speed_multiplier *= 0.99
                dpg.set_value("speed_text", f"Speed: {speed_multiplier:.1f}x")

            if view_mode=="2d":
                cam_pos=np.array([cam_target[0],
                                  cam_target[1]+camera_distance,
                                  cam_target[2]],dtype=np.float64)
                right=np.array([1,0,0],dtype=np.float64)
                up=np.array([0,0,-1],dtype=np.float64)
                forward_vec=np.array([0,-1,0],dtype=np.float64)

            elif camera_mode=="orbit":
                if dpg.is_key_down(dpg.mvKey_W):
                    camera_distance=max(10,camera_distance-5)
                if dpg.is_key_down(dpg.mvKey_S):
                    camera_distance+=5
                if dpg.is_key_down(dpg.mvKey_Left):
                    camera_yaw+=0.02
                if dpg.is_key_down(dpg.mvKey_Right):
                    camera_yaw-=0.02
                if dpg.is_key_down(dpg.mvKey_Up):
                    camera_pitch+=0.02
                    if camera_pitch>math.pi/2-0.1:
                        camera_pitch=math.pi/2-0.1
                if dpg.is_key_down(dpg.mvKey_Down):
                    camera_pitch-=0.02
                    if camera_pitch<-math.pi/2+0.1:
                        camera_pitch=-math.pi/2+0.1
                cam_pos = get_camera_position_orbit()
                right, up, forward_vec= compute_camera_basis(cam_pos, cam_target)

            else:
                forward=np.array([
                    math.cos(free_cam_pitch)*math.cos(free_cam_yaw),
                    math.sin(free_cam_pitch),
                    math.cos(free_cam_pitch)*math.sin(free_cam_yaw)
                ],dtype=np.float64)
                wup=np.array([0,1,0],dtype=np.float64)
                rv=np.cross(forward,wup)
                rn=np.linalg.norm(rv)
                if rn!=0:
                    rv/=rn
                if dpg.is_key_down(dpg.mvKey_W):
                    free_cam_pos+=forward*free_move_speed*dt_frame
                if dpg.is_key_down(dpg.mvKey_S):
                    free_cam_pos-=forward*free_move_speed*dt_frame
                if dpg.is_key_down(dpg.mvKey_A):
                    free_cam_pos-=rv*free_move_speed*dt_frame
                if dpg.is_key_down(dpg.mvKey_D):
                    free_cam_pos+=rv*free_move_speed*dt_frame
                if dpg.is_key_down(dpg.mvKey_U):
                    free_cam_pos+=wup*free_move_speed*dt_frame
                if dpg.is_key_down(dpg.mvKey_O):
                    free_cam_pos-=wup*free_move_speed*dt_frame

                if dpg.is_key_down(dpg.mvKey_Left):
                    free_cam_yaw+=0.02
                if dpg.is_key_down(dpg.mvKey_Right):
                    free_cam_yaw-=0.02
                if dpg.is_key_down(dpg.mvKey_Up):
                    free_cam_pitch+=0.02
                    if free_cam_pitch>math.pi/2-0.1:
                        free_cam_pitch=math.pi/2-0.1
                if dpg.is_key_down(dpg.mvKey_Down):
                    free_cam_pitch-=0.02
                    if free_cam_pitch<-math.pi/2+0.1:
                        free_cam_pitch=-math.pi/2+0.1
                cam_pos=free_cam_pos
                right,up,forward_vec=compute_camera_basis(cam_pos, cam_pos+forward)

            if mouse_wheel_delta!=0:
                if camera_mode=="orbit":
                    camera_distance-=mouse_wheel_delta*10
                    if camera_distance<10:
                        camera_distance=10
                else:
                    forward2=np.array([
                        math.cos(free_cam_pitch)*math.cos(free_cam_yaw),
                        math.sin(free_cam_pitch),
                        math.cos(free_cam_pitch)*math.sin(free_cam_yaw)
                    ],dtype=np.float64)
                    free_cam_pos+=forward2*(mouse_wheel_delta*20)
            mouse_wheel_delta=0.0

            dt_s = dt_frame * speedup * speed_multiplier * (0 if is_paused else 1)
            
            
            for i in range(1, len(bodies)):
                bd = bodies[i]
                bd["angle"] = (bd["angle"] + bd["omega"] * dt_s) % (2 * math.pi)
                bd["pos3d"] = calculate_planet_position(bd, bd["angle"])
                
                
                if bd["name"] == "Earth":
                    moon_data["angle"] = (moon_data["angle"] + moon_data["omega"] * dt_s) % (2 * math.pi)
                    moon_data["pos3d"] = calculate_moon_position(moon_data, bd["pos3d"])

           
            asteroid_belt.update(dt_s)

            dpg.delete_item("drawlist", children_only=True)

            dw = dpg.get_item_width("draw_region")
            dh = dpg.get_item_height("draw_region")
            cx,cy=(dw//2, dh//2)

            
            for star in stars:
                rel = star["pos"] - cam_pos
                ccoord = np.array([
                    np.dot(rel, right),
                    np.dot(rel, up),
                    np.dot(rel, forward_vec)
                ])
                star_2d = project(ccoord, cx, cy)
                dpg.draw_circle(star_2d, star["size"], color=star["color"],
                              fill=star["color"], parent="drawlist")

            
            if show_predictions:
                for bd in bodies[1:]:
                    predictions = calculate_orbit_prediction(bd, steps=100)
                    pts = []
                    for pos in predictions:
                        ccoord = np.array([
                            np.dot(pos-cam_pos, right),
                            np.dot(pos-cam_pos, up),
                            np.dot(pos-cam_pos, forward_vec)
                        ])
                        pts.append(project(ccoord,cx,cy))
                    dpg.draw_polyline(pts, color=(100,100,255,100), parent="drawlist")

            
            orbit_res=100
            for i in range(1,len(bodies)):
                b=bodies[i]
                pts=[]
                for j in range(orbit_res+1):
                    ang=2*math.pi*j/orbit_res
                    radius_m=b["orbit_px"]
                    if b["is_pluto"]:
                        inclination = math.radians(17)
                        x = radius_m * math.cos(ang)
                        y = radius_m * math.sin(ang) * math.sin(inclination)
                        z = radius_m * math.sin(ang) * math.cos(inclination)
                        orbit_xyz = np.array([x, y, z], dtype=np.float64)
                    else:
                        orbit_xyz=np.array([radius_m*math.cos(ang),0,radius_m*math.sin(ang)],dtype=np.float64)
                    ccoord=np.array([
                        np.dot(orbit_xyz-cam_pos, right),
                        np.dot(orbit_xyz-cam_pos, up),
                        np.dot(orbit_xyz-cam_pos, forward_vec)
                    ])
                    pts.append(project(ccoord,cx,cy))
                dpg.draw_polyline(pts, color=(100,100,100,100), parent="drawlist")

            
            asteroid_positions = asteroid_belt.get_positions()
            for pos, size, color in asteroid_positions:
                rel = pos - cam_pos
                ccoord = np.array([
                    np.dot(rel, right),
                    np.dot(rel, up),
                    np.dot(rel, forward_vec)
                ])
                p2d = project(ccoord, cx, cy)
                dpg.draw_circle(p2d, size, color=color,
                              fill=color, parent="drawlist")

            positions_2d=[]
            
            # Draw Sun
            sun_cam=np.array([
                np.dot(sun["pos3d"]-cam_pos,right),
                np.dot(sun["pos3d"]-cam_pos,up),
                np.dot(sun["pos3d"]-cam_pos,forward_vec)
            ])
            sun_2d=project(sun_cam,cx,cy)
            positions_2d.append(sun_2d)
            dpg.draw_circle(sun_2d, sun["radius"],
                            color=sun["color"], fill=sun["color"],
                            parent="drawlist")

            
            for b in bodies[1:]:
                ccoord2=np.array([
                    np.dot(b["pos3d"]-cam_pos,right),
                    np.dot(b["pos3d"]-cam_pos,up),
                    np.dot(b["pos3d"]-cam_pos,forward_vec)
                ])
                p2d=project(ccoord2,cx,cy)
                positions_2d.append(p2d)
                dpg.draw_circle(p2d, b["radius"],
                                color=b["color"], fill=b["color"],
                                parent="drawlist")
                
                
                if b["name"] == "Earth":
                    moon_coord = np.array([
                        np.dot(moon_data["pos3d"]-cam_pos,right),
                        np.dot(moon_data["pos3d"]-cam_pos,up),
                        np.dot(moon_data["pos3d"]-cam_pos,forward_vec)
                    ])
                    moon_2d = project(moon_coord,cx,cy)
                    dpg.draw_circle(moon_2d, moon_data["draw_radius"],
                                  color=moon_data["color"], fill=moon_data["color"],
                                  parent="drawlist")

            dpg.push_container_stack("drawlist")
            for b in bodies[1:]:
                if b["draw_details"]:
                    ccoord3=np.array([
                        np.dot(b["pos3d"]-cam_pos,right),
                        np.dot(b["pos3d"]-cam_pos,up),
                        np.dot(b["pos3d"]-cam_pos,forward_vec)
                    ])
                    xy3=project(ccoord3,cx,cy)
                    b["draw_details"](xy3, b["radius"])
            dpg.pop_container_stack()

            if dpg.is_mouse_button_down(0):
                mpos=dpg.get_mouse_pos(local=True)
                for i, bd in enumerate(bodies):
                    if i<len(positions_2d):
                        px,py=positions_2d[i]
                        dx=mpos[0]-px
                        dy=mpos[1]-py
                        if math.sqrt(dx*dx+dy*dy)< bd["radius"]+5:
                            info = format_body_info(bd)
                            dpg.set_value("info_text", info)
                            if is_fullscreen:
                                dpg.set_value("fullscreen_info_text", info)
                            break

            dpg.render_dearpygui_frame()

    except Exception as ex:
        logger.exception("Error in main loop.")
    finally:
        logger.info("Closing application...")

    dpg.destroy_context()
    logger.success("App closed successfully.")

if __name__=="__main__":
    main()