"""SkyRoute 3D — native OpenGL UAV path-planning simulation."""
from __future__ import annotations
import math
import random
import sys
from time import perf_counter
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "vendor"))
import pyglet
pyglet.options["shadow_window"] = False
pyglet.options["debug_gl"] = False
from pyglet.gl import *
from pyglet.gl.gl_compat import *
from pyglet.window import key, mouse
from skyroute_ai_model import (
    DStarLitePlanner,
    LiDARKalmanModel,
    assess_collision_risk,
    astar_3d,
    choose_best_path,
    obstacle_surface_clearance,
)


def add(a, b): return tuple(a[i] + b[i] for i in range(3))
def sub(a, b): return tuple(a[i] - b[i] for i in range(3))
def mul(a, s): return tuple(v * s for v in a)
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def length(v): return math.sqrt(dot(v, v))
def unit(v):
    n = length(v)
    return mul(v, 1.0/n) if n > 1e-9 else (0.0, 0.0, 0.0)
def distance(a, b): return length(sub(a, b))
def lerp(a, b, t): return add(a, mul(sub(b, a), t))
def clamp(x, lo, hi): return max(lo, min(hi, x))
def rgb(c, alpha=1.0): glColor4f(c[0], c[1], c[2], alpha)


def draw_box(center, half, colour):
    x, y, z = center
    hx, hy, hz = half
    faces = [
        ((.73,.77,.83),((-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz))),
        ((.82,.85,.89),((-hx,-hy,-hz),(-hx,hy,-hz),(hx,hy,-hz),(hx,-hy,-hz))),
        ((.90,.92,.95),((-hx,-hy,-hz),(hx,-hy,-hz),(hx,-hy,hz),(-hx,-hy,hz))),
        ((.66,.71,.78),((-hx,hy,-hz),(-hx,hy,hz),(hx,hy,hz),(hx,hy,-hz))),
        ((.77,.81,.86),((-hx,-hy,-hz),(-hx,-hy,hz),(-hx,hy,hz),(-hx,hy,-hz))),
        ((.85,.88,.92),((hx,-hy,-hz),(hx,hy,-hz),(hx,hy,hz),(hx,-hy,hz))),
    ]
    glBegin(GL_QUADS)
    for shade, verts in faces:
        rgb(tuple(colour[i]*shade[i] for i in range(3)))
        for vx, vy, vz in verts: glVertex3f(x+vx, y+vy, z+vz)
    glEnd()


def draw_line(a, b, colour, width=1.0, alpha=1.0):
    glLineWidth(width); rgb(colour, alpha); glBegin(GL_LINES)
    glVertex3f(*a); glVertex3f(*b); glEnd(); glLineWidth(1.0)


def draw_sphere(center, radius, colour, slices=16, stacks=9, alpha=1.0):
    cx, cy, cz = center
    rgb(colour, alpha)
    for lat in range(stacks):
        a0 = -math.pi/2 + math.pi*lat/stacks
        a1 = -math.pi/2 + math.pi*(lat+1)/stacks
        glBegin(GL_QUAD_STRIP)
        for i in range(slices+1):
            lon = 2*math.pi*i/slices
            for la in (a0, a1):
                glVertex3f(cx+radius*math.cos(la)*math.cos(lon),
                           cy+radius*math.cos(la)*math.sin(lon),
                           cz+radius*math.sin(la))
        glEnd()


def draw_cylinder(radius, height, colour, slices=12):
    glBegin(GL_QUAD_STRIP)
    for i in range(slices+1):
        a=2*math.pi*i/slices; x,y=math.cos(a)*radius,math.sin(a)*radius
        rgb(colour); glVertex3f(x,y,0); glVertex3f(x,y,height)
    glEnd()
    glBegin(GL_TRIANGLE_FAN); rgb(colour); glVertex3f(0,0,height)
    for i in range(slices+1):
        a=2*math.pi*i/slices; glVertex3f(math.cos(a)*radius,math.sin(a)*radius,height)
    glEnd()


def draw_tube(a, b, radius, colour):
    dvec=sub(b,a); span=length(dvec)
    if span < 1e-6: return
    d=mul(dvec,1/span); axis=cross((0,0,1),d)
    alen=length(axis); angle=math.degrees(math.acos(clamp(d[2],-1,1)))
    glPushMatrix(); glTranslatef(*a)
    if alen>1e-6: glRotatef(angle,axis[0]/alen,axis[1]/alen,axis[2]/alen)
    elif d[2]<0: glRotatef(180,1,0,0)
    draw_cylinder(radius,span,colour); glPopMatrix()


def draw_circle_xy(center, radius, colour, width=1.2, alpha=1.0, segments=56):
    glLineWidth(width); rgb(colour,alpha); glBegin(GL_LINE_LOOP)
    for i in range(segments):
        a=2*math.pi*i/segments
        glVertex3f(center[0]+math.cos(a)*radius,center[1]+math.sin(a)*radius,center[2])
    glEnd(); glLineWidth(1.0)


class Obstacle:
    def __init__(self,name,kind,center,size,colour,demo=False):
        self.name=name; self.kind=kind; self.center=center; self.size=size
        self.colour=colour; self.demo=demo; self.contacted=False



class SkyRoute3D(pyglet.window.Window):
    SIDEBAR=322
    DRONE_RADIUS=.34
    GRID_STEP=.65
    WORLD_MIN=(-10.0,-8.0,.65)
    WORLD_MAX=(10.0,8.0,7.1)
    START=(-8.1,0.0,3.25)
    GOAL=(8.0,0.0,3.25)

    def __init__(self):
        config=Config(major_version=2,minor_version=1,depth_size=24,double_buffer=True)
        super().__init__(width=1440,height=900,caption="SkyRoute 3D | UAV Path Planning",
                         resizable=True,config=config,vsync=True)
        self.set_minimum_size(1120,700)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(.965,.978,.991,1.0)
        self.time=0.0; self.drone=self.START; self.heading=0.0; self.velocity=0.0
        self.mode="idle"; self.status_msg="READY FOR FLIGHT"; self.status_timer=0.0
        self.contacts=0; self.route=[]; self.route_index=0; self.direct_target=None
        self.impact_obstacle=None; self.impact_timer=0.0; self.impact_age=0.0
        self.contact_flash=0.0; self.paused=False
        self.camera_yaw=24.0; self.camera_elevation=31.0; self.camera_distance=32.0
        self.obstacles=self.make_obstacles()
        self.make_labels()
        pyglet.clock.schedule_interval(self.update,1.0/60.0)

    def make_obstacles(self):
        return [
            Obstacle("Demo block","box",(0.0,0.0,3.25),(.62,.62,.62),(.97,.48,.22),True),
            Obstacle("North building","box",(-3.7,4.15,1.1),(1.5,1.25,1.1),(.76,.82,.89)),
            Obstacle("East building","box",(4.6,4.8,1.5),(1.15,1.05,1.5),(.74,.85,.83)),
            Obstacle("South building","box",(3.6,-4.1,.9),(1.8,1.1,.9),(.87,.79,.70)),
            Obstacle("Sensor mast","sphere",(-4.8,-4.3,2.35),.82,(.46,.67,.88)),
        ]

    def make_labels(self):
        def label(text,x,y,size=10,color=(35,55,79,255),bold=False,anchor="left"):
            return pyglet.text.Label(text,x=x,y=y,font_name="Arial",font_size=size,
                                     weight="bold" if bold else "normal",color=color,anchor_x=anchor)
        self.labels={
            "brand":label("SKYROUTE",26,self.height-52,22,(25,45,68,255),True),
            "tag":label("AUTONOMOUS FLIGHT LAB",27,self.height-74,8,(105,124,147,255),True),
            "live":label("LIVE",266,self.height-54,9,(26,130,101,255),True),
            "section":label("FLIGHT CONTROL",26,self.height-123,9,(105,124,147,255),True),
            "status":label("READY FOR FLIGHT",49,self.height-164,13,(35,55,79,255),True),
            "mission_head":label("MISSION",26,self.height-224,9,(105,124,147,255),True),
            "mission":label("START  →  LANDING PAD",26,self.height-250,12,(35,55,79,255),True),
            "mission_sub":label("3D waypoint navigation",26,self.height-270,10,(119,136,157,255)),
            "button_contact":label("TEST CONTACT + REROUTE",161,self.height-349,10,(255,255,255,255),True,"center"),
            "button_auto":label("AUTONOMOUS ROUTE",161,self.height-406,10,(38,74,105,255),True,"center"),
            "button_reset":label("RESET SIMULATION",161,self.height-456,9,(79,96,117,255),True,"center"),
            "readout_head":label("LIVE TELEMETRY",26,151,9,(105,124,147,255),True),
            "alt_head":label("ALTITUDE",26,122,8,(119,136,157,255),True),
            "speed_head":label("SPEED",170,122,8,(119,136,157,255),True),
            "alt":label("3.25 m",26,94,18,(35,55,79,255),True),
            "speed":label("0.0 m/s",170,94,18,(35,55,79,255),True),
            "contact_head":label("OBJECT CONTACTS",26,52,8,(119,136,157,255),True),
            "contact_count":label("0",154,49,15,(35,55,79,255),True),
            "controls":pyglet.text.Label("C  Contact demo      A  Safe route\\nR  Reset     Space  Pause     Right-drag  Orbit",
                    x=26,y=24,font_name="Arial",font_size=8,color=(118,135,155,255),multiline=True,width=286),
            "world_title":label("3D FLIGHT ZONE",0,0,9,(91,112,135,255),True),
            "world_sub":label("White campus  ·  3D planning  ·  collision response",0,0,9,(112,130,151,255)),
            "goal_tag":label("LANDING PAD",0,0,8,(34,128,102,255),True),
            "impact_tag":label("",0,0,11,(191,73,52,255),True),
            "metrics":pyglet.text.Label("",x=self.SIDEBAR+34,y=0,font_name="Arial",font_size=8,
                    color=(42,62,84,255),multiline=True,width=500),
        }

    def set_status(self,message,seconds=2.0):
        self.status_msg=message; self.status_timer=seconds

    def reset(self):
        self.drone=self.START; self.heading=0.0; self.velocity=0.0
        self.mode="idle"; self.route=[]; self.route_index=0; self.contacts=0
        self.impact_obstacle=None; self.impact_timer=0.0; self.impact_age=0.0
        self.contact_flash=0.0; self.paused=False
        for obstacle in self.obstacles: obstacle.contacted=False
        self.set_status("READY FOR FLIGHT",2.0)

    def run_contact_demo(self):
        self.reset(); self.mode="direct"; self.direct_target=self.GOAL
        self.route=[self.START,self.GOAL]
        self.set_status("CONTACT TEST · APPROACHING OBJECT",4.0)

    def run_safe_route(self):
        self.reset(); path=self.plan_path(self.START,self.GOAL)
        if not path: self.set_status("NO SAFE ROUTE FOUND",3.0); return
        self.route=path; self.route_index=1 if len(path)>1 else 0
        self.mode="follow"; self.set_status("SAFE 3D ROUTE · IN FLIGHT",2.5)

    def collides(self,point,clearance=0.0):
        px,py,pz=point
        for obstacle in self.obstacles:
            cx,cy,cz=obstacle.center
            if obstacle.kind=="box":
                hx,hy,hz=obstacle.size
                if abs(px-cx)<=hx+clearance and abs(py-cy)<=hy+clearance and abs(pz-cz)<=hz+clearance:
                    return obstacle
            else:
                radius=obstacle.size+clearance
                if (px-cx)**2+(py-cy)**2+(pz-cz)**2<=radius*radius: return obstacle
        return None

    def segment_clear(self,a,b,clearance=.55):
        steps=max(2,int(distance(a,b)/.22))
        return all(not self.collides(lerp(a,b,i/steps),clearance) for i in range(1,steps))

    def plan_path(self,start,goal):
        return astar_3d(start,goal,self.WORLD_MIN,self.WORLD_MAX,self.GRID_STEP,
                        self.collides,self.segment_clear)

    def begin_contact(self,obstacle,previous):
        self.drone=previous; self.velocity=0.0; self.contacts+=1
        obstacle.contacted=True; self.impact_obstacle=obstacle
        self.mode="impact"; self.impact_timer=.72; self.impact_age=0.0
        self.contact_flash=.72; self.set_status("CONTACT DETECTED · BRAKING",2.0)

    def replan_after_contact(self):
        obstacle=self.impact_obstacle
        away=unit(sub(self.drone,obstacle.center))
        if length(away)<.1: away=(-1.0,0.0,0.0)
        backoff=add(self.drone,mul(away,.72)); self.drone=backoff
        self.mode="replan"; self.set_status("CONTACT · REPLANNING AROUND OBJECT",2.8)
        self.route=self.plan_path(backoff,self.GOAL); self.route_index=1 if len(self.route)>1 else 0
        if self.route: self.mode="follow"
        else: self.mode="idle"; self.set_status("REPLAN FAILED · RESET TO TRY AGAIN",3.0)

    def update(self,dt):
        dt=min(dt,.05); self.time+=dt
        if self.status_timer>0: self.status_timer-=dt
        if self.contact_flash>0:
            self.contact_flash=max(0.0,self.contact_flash-dt); self.impact_age+=dt
        if self.paused: self.velocity=0.0; return
        if self.mode=="impact":
            self.impact_timer-=dt
            if self.impact_timer<=0: self.replan_after_contact()
            return
        if self.mode=="direct" and self.direct_target:
            target=self.direct_target; remain=distance(self.drone,target)
            if remain<=.05:
                self.drone=target; self.mode="complete"; self.velocity=0.0
                self.set_status("MISSION COMPLETE",4.0); return
            previous=self.drone; move=min(remain,3.25*dt)
            candidate=add(previous,mul(unit(sub(target,previous)),move))
            obstacle=self.collides(candidate,self.DRONE_RADIUS)
            if obstacle: self.begin_contact(obstacle,previous); return
            self.drone=candidate; self.velocity=move/max(dt,1e-5)
            self.heading=math.degrees(math.atan2(target[1]-previous[1],target[0]-previous[0]))
        elif self.mode=="follow" and self.route:
            while self.route_index<len(self.route) and distance(self.drone,self.route[self.route_index])<.12:
                self.route_index+=1
            if self.route_index>=len(self.route):
                self.drone=self.GOAL; self.mode="complete"; self.velocity=0.0
                self.set_status("REROUTE COMPLETE · SAFE LANDING" if self.contacts else "MISSION COMPLETE · SAFE LANDING",5.0)
                return
            target=self.route[self.route_index]; previous=self.drone
            remain=distance(previous,target); move=min(remain,3.15*dt)
            candidate=add(previous,mul(unit(sub(target,previous)),move))
            obstacle=self.collides(candidate,self.DRONE_RADIUS)
            if obstacle: self.begin_contact(obstacle,previous); return
            self.drone=candidate; self.velocity=move/max(dt,1e-5)
            self.heading=math.degrees(math.atan2(target[1]-previous[1],target[0]-previous[0]))
        else: self.velocity=0.0

    def on_draw(self):
        self.clear(); glViewport(0,0,self.width,self.height)
        self.draw_world(); self.draw_ui()

    def draw_world(self):
        vx=self.SIDEBAR; vw=max(100,self.width-vx)
        glViewport(vx,0,vw,self.height); glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        near=.1; top=near*math.tan(math.radians(48)/2); aspect=vw/max(1.0,float(self.height))
        glFrustum(-top*aspect,top*aspect,-top,top,near,120.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        target=(0.0,0.0,1.7); yaw=math.radians(self.camera_yaw); elev=math.radians(self.camera_elevation)
        horiz=self.camera_distance*math.cos(elev)
        eye=(target[0]+math.sin(yaw)*horiz,target[1]-math.cos(yaw)*horiz,target[2]+self.camera_distance*math.sin(elev))
        forward=unit(sub(target,eye)); side=unit(cross(forward,(0,0,1))); up=cross(side,forward)
        view=(side[0],up[0],-forward[0],0.0,side[1],up[1],-forward[1],0.0,
              side[2],up[2],-forward[2],0.0,-dot(side,eye),-dot(up,eye),dot(forward,eye),1.0)
        glLoadMatrixf((GLfloat*16)(*view))
        self.draw_floor(); self.draw_world_route(); self.draw_environment()
        self.draw_goal(); self.draw_start(); self.draw_drone()
        glViewport(0,0,self.width,self.height)

    def draw_floor(self):
        glDisable(GL_DEPTH_TEST); glColor4f(.985,.991,.997,1)
        glBegin(GL_QUADS)
        for v in ((-12,-10,-.03),(12,-10,-.03),(12,10,-.03),(-12,10,-.03)): glVertex3f(*v)
        glEnd(); glEnable(GL_DEPTH_TEST)
        glColor4f(.83,.88,.93,.7); glBegin(GL_LINES)
        for x in range(-10,11): glVertex3f(x,-8,0); glVertex3f(x,8,0)
        for y in range(-8,9): glVertex3f(-10,y,0); glVertex3f(10,y,0)
        glEnd()
        draw_circle_xy((self.START[0],self.START[1],.035),.76,(.24,.66,.78),2,.7)
        draw_circle_xy((self.GOAL[0],self.GOAL[1],.04),.95,(.18,.63,.49),2,.78)

    def draw_world_route(self):
        if len(self.route)<2: return
        if self.mode=="direct" or getattr(self,"route_planner",None)=="D* LITE":
            for a,b in zip(self.route,self.route[1:]):
                span=distance(a,b); count=max(1,int(span/.42))
                for i in range(0,count,2):
                    p0=lerp(a,b,i/count); p1=lerp(a,b,min(1,(i+1)/count))
                    draw_line((p0[0],p0[1],p0[2]+.10),(p1[0],p1[1],p1[2]+.10),(.96,.53,.20),2.1,.9)
            return
        for a,b in zip(self.route,self.route[1:]):
            draw_line((a[0],a[1],a[2]+.10),(b[0],b[1],b[2]+.10),(.10,.66,.70),3,.9)
        for p in self.route[1:-1]: draw_sphere((p[0],p[1],p[2]+.10),.08,(.10,.66,.70),10,5)

    def draw_environment(self):
        for obstacle in self.obstacles:
            if obstacle.kind=="box":
                cx,cy,cz=obstacle.center; hx,hy,hz=obstacle.size
                draw_circle_xy((cx,cy,.035),max(hx,hy)*.9,(.48,.55,.64),1.1,.19)
                if obstacle.demo:
                    base=(.91,.30,.19) if obstacle.contacted else obstacle.colour
                    draw_box(obstacle.center,obstacle.size,base)
                    draw_box((cx,cy,cz+hz+.045),(hx*.84,hy*.84,.045),(.99,.69,.36) if not obstacle.contacted else (.93,.39,.28))
                    draw_sphere((cx,cy,cz+hz+.20),.11,(.98,.30,.20) if obstacle.contacted else (.98,.65,.22),12,7)
                    if obstacle.contacted or self.mode=="direct":
                        pulse=.7+.16*math.sin(self.time*5)
                        draw_circle_xy((cx,cy,.07),1.05*pulse,(.94,.40,.22),1.6,.7)
                else:
                    draw_box(obstacle.center,obstacle.size,obstacle.colour)
                    for sign in (-1,1):
                        for floor in range(max(1,int(hz*1.3))):
                            z=max(.35,cz-hz+.45+floor*.62)
                            for row in (-1,0,1):
                                wy=cy+row*min(hy*.52,.65)
                                draw_box((cx+sign*(hx+.012),wy,z),(.015,.17,.15),(.70,.85,.91))
            else:
                draw_circle_xy((obstacle.center[0],obstacle.center[1],.04),obstacle.size*1.2,(.52,.62,.70),1.2,.45)
                draw_sphere(obstacle.center,obstacle.size,obstacle.colour,20,11)
        if self.contact_flash>0 and self.impact_obstacle:
            o=self.impact_obstacle; radius=.35+(.72-self.contact_flash)*2.4
            c=o.center; alpha=self.contact_flash/.72
            draw_circle_xy(c,radius,(.94,.29,.19),2.5,alpha)
            draw_circle_xy((c[0],c[1],c[2]+.12),radius*.72,(.98,.55,.24),1.7,alpha)

    def draw_goal(self):
        gx,gy,_=self.GOAL; glColor4f(.16,.64,.49,.15); glBegin(GL_QUADS)
        for v in ((gx-.85,gy-.85,.02),(gx+.85,gy-.85,.02),(gx+.85,gy+.85,.02),(gx-.85,gy+.85,.02)): glVertex3f(*v)
        glEnd(); draw_circle_xy((gx,gy,.08),.62,(.16,.60,.46),2,.9)
        draw_circle_xy((gx,gy,.09),.34,(.16,.60,.46),1.5,.75)
        draw_line((gx-.18,gy,.10),(gx-.04,gy-.14,.10),(.15,.58,.44),2)
        draw_line((gx-.04,gy-.14,.10),(gx+.20,gy+.16,.10),(.15,.58,.44),2)

    def draw_start(self):
        draw_circle_xy((self.START[0],self.START[1],.08),.43,(.24,.65,.76),1.4,.75)

    def draw_drone(self):
        x,y,z=self.drone
        shadow_radius=.7+min(z*.055,24.0)
        shadow_alpha=clamp(.19-z*.0007,.035,.19)
        draw_circle_xy((x,y,.055),shadow_radius,(.33,.39,.46),1.2,shadow_alpha)
        bob=0.0 if self.mode in ("idle","complete") else .07*math.sin(self.time*5)
        glPushMatrix(); glTranslatef(x,y,z+bob); glRotatef(self.heading,0,0,1)
        arms=((.72,.72,0),(.72,-.72,0),(-.72,.72,0),(-.72,-.72,0))
        for p in arms: draw_tube((0,0,0),p,.074,(.24,.30,.38))
        draw_box((0,0,.02),(.39,.30,.15),(.91,.94,.97))
        draw_box((.02,0,.17),(.22,.20,.055),(.30,.45,.58))
        draw_sphere((.40,0,0),.10,(.92,.26,.22),12,7)
        draw_sphere((.34,0,-.03),.07,(.19,.34,.48),12,7)
        for px,py,_ in arms:
            glPushMatrix(); glTranslatef(px,py,.02)
            draw_cylinder(.145,.14,(.22,.27,.34),16)
            draw_sphere((0,0,.15),.09,(.16,.21,.28),12,7)
            glPushMatrix(); glTranslatef(0,0,.205)
            glRotatef((self.time*1100 if self.mode not in ("idle","complete") else 0.0)+px*300+py*240,0,0,1)
            draw_box((0,0,0),(.50,.035,.018),(.16,.22,.29))
            glPopMatrix(); glPopMatrix()
        for side in (-1,1):
            draw_tube((.08,side*.21,-.10),(.10,side*.34,-.40),.035,(.26,.32,.39))
            draw_tube((-.08,side*.21,-.10),(-.10,side*.34,-.40),.035,(.26,.32,.39))
            draw_tube((-.38,side*.34,-.40),(.38,side*.34,-.40),.028,(.22,.28,.35))
        draw_sphere((.70,0,.12),.06,(.95,.18,.16),10,6)
        draw_sphere((-.70,-.72,.12),.05,(.13,.65,.41),10,6)
        draw_sphere((-.70,.72,.12),.05,(.13,.65,.41),10,6)
        glPopMatrix()

    def ui_rect(self,x,y,w,h,colour):
        glColor4f(*colour); glBegin(GL_QUADS)
        glVertex2f(x,y); glVertex2f(x+w,y); glVertex2f(x+w,y+h); glVertex2f(x,y+h); glEnd()

    def draw_ui(self):
        glViewport(0,0,self.width,self.height); glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,self.width,0,self.height,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity(); h=self.height
        self.ui_rect(0,0,self.SIDEBAR,h,(.995,.998,1,1))
        self.ui_rect(self.SIDEBAR-1,0,1,h,(.84,.88,.92,1))
        self.ui_rect(20,h-191,282,48,(.94,.97,.99,1))
        state=(.13,.60,.45,1) if self.mode in ("idle","complete") else ((.89,.28,.19,1) if self.mode=="impact" else (.10,.58,.70,1))
        self.ui_rect(27,h-175,9,9,state)
        self.ui_rect(20,h-293,282,65,(.975,.984,.992,1))
        self.ui_rect(20,h-373,282,48,(.10,.47,.59,1))
        self.ui_rect(20,h-426,282,40,(.91,.95,.97,1))
        self.ui_rect(20,h-474,282,35,(.965,.974,.983,1))
        self.ui_rect(20,31,282,132,(.965,.976,.986,1))
        self.ui_rect(157,53,1,77,(.87,.90,.94,1))
        self.ui_rect(26,77,4,4,(.11,.62,.48,1)); self.ui_rect(26,39,7,7,(.96,.48,.21,1))
        self.ui_rect(self.SIDEBAR+22,h-38,150,24,(.99,.995,1,.88))
        self.ui_rect(self.SIDEBAR+22,h-67,430,20,(.99,.995,1,.88))
        l=self.labels
        for name,y in (("brand",h-52),("tag",h-74),("live",h-54),("section",h-123),
                       ("status",h-164),("mission_head",h-224),("mission",h-250),
                       ("mission_sub",h-270),("button_contact",h-349),("button_auto",h-406),
                       ("button_reset",h-456)):
            l[name].y=y
        l["status"].text=self.status_msg if self.status_timer>0 else self.status_for_mode()
        l["status"].font_size=10 if len(l["status"].text)>22 else 13
        l["status"].color=(178,65,47,255) if self.mode=="impact" else (35,55,79,255)
        l["alt"].text=f"{self.drone[2]:.2f} m"; l["speed"].text=f"{self.velocity:.1f} m/s"
        l["contact_count"].text=str(self.contacts)
        l["world_title"].x=self.SIDEBAR+32; l["world_title"].y=h-31
        l["world_sub"].x=self.SIDEBAR+32; l["world_sub"].y=h-57
        l["goal_tag"].x=self.width-170; l["goal_tag"].y=45
        l["impact_tag"].x=self.SIDEBAR+28; l["impact_tag"].y=h-91
        if self.contacts:
            note="ROUTE UPDATED" if self.mode=="complete" else ("BRAKING" if self.mode=="impact" else "REPLANNING ACTIVE")
            l["impact_tag"].text=f"CONTACT {self.contacts}  ·  {note}"
        else:
            l["impact_tag"].text=""
        for label in l.values(): label.draw()
        glEnable(GL_DEPTH_TEST)

    def status_for_mode(self):
        if self.paused: return "SIMULATION PAUSED"
        return {"idle":"READY FOR FLIGHT","direct":"CONTACT TEST · IN PROGRESS",
                "impact":"CONTACT DETECTED · BRAKING","replan":"REPLANNING AROUND OBJECT",
                "follow":"FOLLOWING UPDATED ROUTE" if self.contacts else "AUTONOMOUS ROUTE · IN FLIGHT",
                "complete":"REROUTE COMPLETE" if self.contacts else "MISSION COMPLETE"}.get(self.mode,"READY FOR FLIGHT")

    def on_resize(self,width,height):
        glViewport(0,0,width,height)

    def on_mouse_press(self,x,y,button,modifiers):
        h=self.height
        if button==mouse.LEFT and x<self.SIDEBAR:
            if h-373<=y<=h-325: self.run_contact_demo()
            elif h-426<=y<=h-386: self.run_safe_route()
            elif h-474<=y<=h-439: self.reset()
        elif button==mouse.RIGHT: self.last_mouse=(x,y)

    def on_mouse_drag(self,x,y,dx,dy,buttons,modifiers):
        if buttons & mouse.RIGHT:
            self.camera_yaw+=dx*.34
            self.camera_elevation=clamp(self.camera_elevation+dy*.24,12,68)

    def on_mouse_scroll(self,x,y,scroll_x,scroll_y):
        self.camera_distance=clamp(self.camera_distance-scroll_y*1.7,20,48)

    def on_key_press(self,symbol,modifiers):
        if symbol==key.C: self.run_contact_demo()
        elif symbol==key.A: self.run_safe_route()
        elif symbol==key.R: self.reset()
        elif symbol==key.SPACE:
            self.paused=not self.paused
            self.set_status("SIMULATION PAUSED" if self.paused else "SIMULATION RESUMED",1.8)
        elif symbol==key.ESCAPE: self.close()



class UrbanObstacle(Obstacle):
    """City object with optional smooth, multi-axis motion."""
    def __init__(self, name, kind, center, size, colour, motion=None):
        super().__init__(name, kind, center, size, colour, False)
        self.motion = motion
        self.home = center
        self.heading = 0.0



class SkyRouteCity(SkyRoute3D):
    """Urban sensor-avoidance scenario with moving traffic and camera presets."""
    # 4.2 km x 1.05 km = 4.41 km², with a long axis like Central Park.
    MAP_HALF_X = 2100.0
    MAP_HALF_Y = 525.0
    BLOCK_SPACING = 210.0
    START = (-735.0, -105.0, 150.0)
    GOAL = (735.0, 105.0, 150.0)
    WORLD_MIN = (-2100.0, -525.0, 0.75)
    WORLD_MAX = (2100.0, 525.0, 225.0)
    GRID_STEP = 50.0
    SENSOR_RANGE = 180.0
    SENSOR_SAFETY_CLEARANCE = 3.0
    SENSOR_REROUTE_DISTANCE = 2.0
    COURSE_HAZARD_COUNT = 12
    HAZARD_SPEED_SCALE = 1.0
    CITY_CAR_COUNT = 64
    LIDAR_BEAM_ANGLES = (-32,-24,-16,-8,0,8,16,24,32)
    LIDAR_BEAM_HALF_ANGLE = 5.0
    LIDAR_MEASUREMENT_STD = 1.5
    def __init__(self):
        self.camera_mode = "overview"
        self.avoidances = 0
        self.sensor_target = None
        self.avoidance_focus = None
        self.avoidance_cooldowns = {}
        self.detected_course_hazards = set()
        self._planning_obstacle = None
        self._planning_obstacle_center = None
        self._planning_mode = False
        self._sensor_prediction = None
        self._dstar_planner = None
        self.ai_model=LiDARKalmanModel(
            beam_angles=self.LIDAR_BEAM_ANGLES,
            beam_half_angle=self.LIDAR_BEAM_HALF_ANGLE,
            max_range=self.SENSOR_RANGE,
            measurement_std=self.LIDAR_MEASUREMENT_STD,
            seed=41218,
        )
        self._kalman_tracks=self.ai_model.tracks
        self._lidar_detections=self.ai_model.return_points
        super().__init__()
        self.camera_mode = "overview"
        self.camera_distance = 4600.0
        self.camera_elevation = 56.0
        self.set_caption("SkyRoute Urban 3D | 4.41 km² Downtown Flight Region")

    @classmethod
    def course_route_point(cls,fraction):
        return lerp(cls.START,cls.GOAL,clamp(fraction,0.0,1.0))

    def make_obstacles(self):
        rng = random.Random(41216)
        buildings = []
        tree_rng = random.Random(8127)
        foliage_palette = [(.22,.43,.30),(.27,.50,.33),(.34,.55,.36),(.40,.59,.38)]
        self.trees=[]
        for x in range(-1890,1891,210):
            for y in range(-315,316,210):
                for side in (-1,1):
                    self.trees.append((
                        x+side*25.0,y,
                        tree_rng.uniform(9.5,14.0),
                        tree_rng.uniform(4.6,6.2),
                        foliage_palette[tree_rng.randrange(len(foliage_palette))],
                        foliage_palette[tree_rng.randrange(len(foliage_palette))],
                        tree_rng.uniform(.4,1.4),
                    ))
        facade_palette = [
            (.72,.76,.80), (.84,.81,.75), (.76,.83,.85),
            (.83,.78,.72), (.70,.75,.82), (.88,.86,.81),
        ]
        for row, cy in enumerate(range(-315, 316, 210)):
            for col, cx in enumerate(range(-1995, 1996, 210)):
                # Leave a small civic plaza at the launch pad and reserve a low
                # rooftop heliport at the far end of the route.
                if (cx, cy) == (-735, -105):
                    continue
                if (cx, cy) == (735, 105):
                    height = 36.0
                    name = "Rooftop heliport tower"
                else:
                    radial = math.hypot(cx, cy)
                    height = clamp(128.0 - radial*.058 + rng.uniform(-18.0, 24.0), 28.0, 145.0)
                    name = f"City block {row+1:02d}-{col+1:02d}"
                hx = rng.uniform(68.0, 78.0)
                hy = rng.uniform(64.0, 74.0)
                colour = facade_palette[rng.randrange(len(facade_palette))]
                buildings.append(UrbanObstacle(name, "box", (cx,cy,height/2),
                                                (hx,hy,height/2), colour))

        # Cars follow the avenues in both directions. Their low altitude keeps
        # ground traffic visible without making it a false UAV collision risk.
        car_palette = [(.17,.42,.64), (.79,.32,.22), (.84,.82,.74),
                       (.25,.38,.34), (.78,.57,.24), (.31,.34,.40)]
        avenue_x = list(range(-2100, 2101, 210))
        avenue_y = list(range(-420, 421, 210))
        for i in range(self.CITY_CAR_COUNT):
            axis = i % 2
            lane = -8.0 if (i//2) % 2 == 0 else 8.0
            speed = rng.uniform(.011, .018)
            phase = rng.uniform(-math.pi, math.pi)
            colour = car_palette[i % len(car_palette)]
            if axis == 0:
                road_y = avenue_y[(i*3) % len(avenue_y)] + lane
                motion = (0, -2040.0, 2040.0, speed, phase)
                home = (0.0, road_y, 1.0)
            else:
                road_x = avenue_x[(i*5) % len(avenue_x)] + lane
                motion = (1, -490.0, 490.0, speed, phase)
                home = (road_x, 0.0, 1.0)
            buildings.append(UrbanObstacle(f"City traffic {i+1:02d}", "vehicle", home,
                                            (4.5,2.0,.85), colour, motion=motion))

        # A twelve-object aerial course crosses the route at varied headings.
        # A seeded random stream keeps each run repeatable while giving every
        # object its own axis, direction, speed and crossing distance.
        takeoff_time = self.START[2] / 9.0
        cruise_speed = 35.0
        course_rng = random.Random(41217)

        def course_motion(fraction,lateral_range,angular_speed,axis,direction,
                          cross_amplitude,cross_speed,cross_direction,
                          vertical_amplitude,vertical_speed):
            route_point=self.course_route_point(fraction)
            point=route_point
            encounter_time=takeoff_time+distance(self.START,route_point)/cruise_speed
            # Each hazard crosses the nominal route at its scheduled encounter,
            # while a second lateral oscillation and altitude bob make its
            # approach visibly three-dimensional and non-identical.
            phase=-angular_speed*encounter_time+(math.pi if direction<0 else 0.0)
            cross_phase=-cross_speed*encounter_time+(math.pi if cross_direction<0 else 0.0)
            vertical_phase=-vertical_speed*encounter_time
            coordinate=point[axis]
            cross_axis=1-axis
            motion=(axis,coordinate-lateral_range,coordinate+lateral_range,angular_speed,phase,
                    cross_axis,cross_amplitude,cross_speed,cross_phase,
                    vertical_amplitude,vertical_speed,vertical_phase)
            return point,motion

        hazard_specs = [
            *((fraction,"aircraft",1.2,(.91,.34,.22),"crossing UAV")
              for fraction in (.12,.27,.42,.58,.73,.88)),
            *((fraction,"aircraft",1.45,(.95,.67,.18),"inspection UAV")
              for fraction in (.19,.50,.81)),
            *((fraction,"box",(5.5,3.5,2.5),(.82,.30,.22),"work platform")
              for fraction in (.34,.66,.94)),
        ]
        hazard_specs=hazard_specs[:max(1,min(int(self.COURSE_HAZARD_COUNT),len(hazard_specs)))]
        for number,(fraction,kind,size,colour,description) in enumerate(hazard_specs,1):
            axis=course_rng.choice((0,1))
            direction=course_rng.choice((-1,1))
            cross_direction=course_rng.choice((-1,1))
            lateral_range=course_rng.uniform(60.0,105.0)
            is_aircraft=kind=="aircraft"
            angular_speed=course_rng.uniform(.10,.18) if is_aircraft else course_rng.uniform(.065,.12)
            angular_speed*=self.HAZARD_SPEED_SCALE
            cross_amplitude=course_rng.uniform(12.0,32.0)
            cross_speed=course_rng.uniform(.62,.98)*angular_speed
            vertical_amplitude=course_rng.uniform(5.0,11.0) if is_aircraft else course_rng.uniform(2.0,5.0)
            vertical_speed=course_rng.uniform(.68,1.12)*angular_speed
            point,motion=course_motion(fraction,lateral_range,angular_speed,axis,direction,
                                       cross_amplitude,cross_speed,cross_direction,
                                       vertical_amplitude,vertical_speed)
            buildings.append(UrbanObstacle(f"Course hazard {number:02d} · {description}",
                                            kind,point,size,colour,motion=motion))

        # Small moving pedestrians add scale cues to the sidewalks.
        for i in range(14):
            axis = i % 2
            lane = 29.0 if i % 3 else -29.0
            if axis == 0:
                road_y = avenue_y[(i*2) % len(avenue_y)]
                home = (0.0, road_y+lane, .85)
                motion = (0, -2040.0, 2040.0, .006, i*.9)
            else:
                road_x = avenue_x[(i*3) % len(avenue_x)]
                home = (road_x+lane, 0.0, .85)
                motion = (1, -490.0, 490.0, .006, i*.9)
            buildings.append(UrbanObstacle(f"Pedestrian {i+1:02d}", "sphere", home,
                                            1.05, (.84,.56,.30), motion=motion))
        return buildings

    def segment_clear(self, a, b, clearance=None):
        if clearance is None:
            clearance=self.DRONE_RADIUS+self.SENSOR_SAFETY_CLEARANCE
        steps = max(2, int(math.ceil(distance(a,b)/8.0)))
        return all(not self.collides(lerp(a,b,i/steps),clearance)
                   for i in range(1,steps))

    def make_labels(self):
        super().make_labels()
        l=self.labels
        l["button_contact"].text="START SENSOR MISSION"
        l["button_contact"].font_size=9
        l["button_auto"].text="START SAFE MISSION"
        l["mission"].text="START PLAZA  →  ROOFTOP PAD"
        l["mission_sub"].text="1.5 km route · 12 hazards · rooftop landing"
        l["contact_head"].text="SENSOR AVOIDANCES"
        l["world_title"].text="CENTRAL PARK-SCALE DOWNTOWN"
        l["world_sub"].text="4.2 km long  ·  4.41 km²  ·  80 blocks  ·  12 course hazards"
        l["controls"].text="C sensor mission  A safe route  R reset  Space pause\n1 orbit  2 chase  3 pilot  4 top  5 city view"
        l["impact_tag"].color=(177,104,29,255)
        l["camera"]=pyglet.text.Label("CAMERA  1 ORBIT  ·  2 CHASE  ·  3 PILOT  ·  4 TOP  ·  5 CITY",
            x=0,y=0,font_name="Arial",font_size=8,weight="bold",color=(81,101,124,255))
        l["sensor_value"]=pyglet.text.Label("RANGE 180m",x=186,y=49,font_name="Arial",font_size=8,weight="bold",color=(81,101,124,255))

    def reset(self):
        super().reset()
        self.avoidances=0
        self.sensor_reading=self.SENSOR_RANGE
        self.sensor_target=None
        self.avoidance_focus=None
        self.avoidance_cooldowns.clear()
        self.detected_course_hazards.clear()
        self._dstar_planner=None
        self._planning_obstacle=None
        self._planning_obstacle_center=None
        self._planning_mode=False
        self._sensor_prediction=None
        self.ai_model.reset()
        self._kalman_tracks=self.ai_model.tracks
        self._lidar_detections=self.ai_model.return_points
        self.camera_mode="overview"
        self.camera_distance=4600.0
        self.camera_elevation=56.0

    def camera_focus(self):
        return (0.0,0.0,0.0) if self.camera_mode=="overview" else self.drone

    def run_contact_demo(self):
        self.reset()
        self.mode="direct"
        self.direct_target=self.GOAL
        self.route=[self.START,self.GOAL]
        self.set_status("SENSOR TEST · SCANNING FOR MOVING TRAFFIC",4.0)

    def run_safe_route(self):
        self.reset()
        path=self.plan_path(self.START,self.GOAL)
        if not path:
            self.set_status("NO SAFE ROUTE FOUND",3.0)
            return
        self.route=path
        self.route_index=1 if len(path)>1 else 0
        self.mode="follow"
        self.set_status("SAFE 3D ROUTE · LIVE SENSORS ENABLED",2.5)

    def predicted_center(self,obstacle,horizon=1.25):
        if not obstacle.motion:
            return obstacle.center
        center,_=self.obstacle_motion_state(obstacle,self.time+horizon)
        return center

    @staticmethod
    def obstacle_motion_state(obstacle,at_time):
        """Evaluate a moving city's position and velocity at a given time."""
        motion=obstacle.motion
        axis,lower,upper,speed,phase=motion[:5]
        center=list(obstacle.home)
        velocity=[0.0,0.0,0.0]
        angle=at_time*speed+phase
        span=upper-lower
        center[axis]=lower+(math.sin(angle)+1.0)*.5*span
        velocity[axis]=.5*span*speed*math.cos(angle)
        if len(motion)>=12:
            cross_axis,cross_amplitude,cross_speed,cross_phase,vertical_amplitude,vertical_speed,vertical_phase=motion[5:12]
            cross_angle=at_time*cross_speed+cross_phase
            vertical_angle=at_time*vertical_speed+vertical_phase
            center[cross_axis]=obstacle.home[cross_axis]+cross_amplitude*math.sin(cross_angle)
            center[2]=obstacle.home[2]+vertical_amplitude*math.sin(vertical_angle)
            velocity[cross_axis]=cross_amplitude*cross_speed*math.cos(cross_angle)
            velocity[2]=vertical_amplitude*vertical_speed*math.cos(vertical_angle)
        return tuple(center),tuple(velocity)

    def collides(self,point,clearance=0.0):
        px,py,pz=point
        for obstacle in self.obstacles:
            # In sensor mode, replan around the hazard currently in view.
            # Future course hazards remain unknown so the forward sensor can
            # encounter and report them one at a time during the mission.
            if (self._planning_mode and getattr(self,"mission_type",None)=="sensor"
                    and obstacle.name.startswith("Course hazard")
                    and obstacle is not self._planning_obstacle):
                continue
            if (self._planning_mode and obstacle is self._planning_obstacle
                    and self._planning_obstacle_center is not None):
                center=self._planning_obstacle_center
            else:
                center=self.predicted_center(obstacle) if self._planning_mode else obstacle.center
            cx,cy,cz=center
            dynamic_pad=.82 if self._planning_mode and obstacle.motion else 0.0
            pad=clearance+dynamic_pad
            if obstacle.kind in ("sphere","aircraft"):
                radius=obstacle.size+pad
                if (px-cx)**2+(py-cy)**2+(pz-cz)**2<=radius*radius:
                    return obstacle
            else:
                hx,hy,hz=obstacle.size
                if abs(px-cx)<=hx+pad and abs(py-cy)<=hy+pad and abs(pz-cz)<=hz+pad:
                    return obstacle
        return None

    def plan_path(self,start,goal):
        self._planning_mode=True
        try:
            return super().plan_path(start,goal)
        finally:
            self._planning_mode=False

    def plan_path_dstar(self,start,goal):
        step=self.GRID_STEP
        lo,hi=self.WORLD_MIN,self.WORLD_MAX
        dims=tuple(int(round((hi[i]-lo[i])/step))+1 for i in range(3))
        def to_cell(point):
            return tuple(int(round((point[i]-lo[i])/step)) for i in range(3))
        def to_point(cell):
            return tuple(lo[i]+cell[i]*step for i in range(3))
        start_cell,goal_cell=to_cell(start),to_cell(goal)
        old_planning_mode=self._planning_mode
        self._planning_mode=True
        try:
            if self.collides(start,.25) or self.collides(goal,.25):
                return []
            blocked=set()
            target=self._planning_obstacle
            target_center=(self._planning_obstacle_center
                           if target is not None and self._planning_obstacle_center is not None
                           else self.predicted_center(target) if target else None)
            target_pad=step*.55
            for x in range(dims[0]):
                for y in range(dims[1]):
                    for z in range(dims[2]):
                        cell=(x,y,z)
                        point=to_point(cell)
                        blocked_here=self.collides(point,.53)
                        if not blocked_here and target is not None:
                            px,py,pz=point
                            cx,cy,cz=target_center
                            if target.kind in ("box","vehicle"):
                                hx,hy,hz=target.size
                                blocked_here=(abs(px-cx)<=hx+target_pad and
                                              abs(py-cy)<=hy+target_pad and
                                              abs(pz-cz)<=hz+target_pad)
                            else:
                                radius=target.size+target_pad
                                blocked_here=((px-cx)**2+(py-cy)**2+(pz-cz)**2<=radius*radius)
                        if blocked_here:
                            blocked.add(cell)
            planner=self._dstar_planner
            if (planner is None or planner.dims!=dims or planner.step!=step or
                    planner.goal!=goal_cell):
                planner=DStarLitePlanner(dims,step,start_cell,goal_cell)
                self._dstar_planner=planner
            else:
                planner.move_start(start_cell)
            planner.update_blocked(blocked)
            cells=planner.extract_path()
            if not cells:
                return []
            raw=[start]+[to_point(cell) for cell in cells[1:-1]]+[goal]
            smooth=[raw[0]]
            anchor=0
            while anchor<len(raw)-1:
                furthest=anchor+1
                for candidate in range(len(raw)-1,anchor,-1):
                    if self.segment_clear(raw[anchor],raw[candidate]):
                        furthest=candidate
                        break
                smooth.append(raw[furthest])
                anchor=furthest
            return smooth
        finally:
            self._planning_mode=old_planning_mode

    def animate_city(self):
        for obstacle in self.obstacles:
            if not obstacle.motion:
                continue
            center,velocity=self.obstacle_motion_state(obstacle,self.time)
            obstacle.center=center
            if math.hypot(velocity[0],velocity[1])>1e-5:
                obstacle.heading=math.degrees(math.atan2(velocity[1],velocity[0]))

    def route_aim(self):
        if self.mode=="direct":
            return self.direct_target or self.GOAL
        if self.mode=="follow" and self.route:
            while self.route_index<len(self.route) and distance(self.drone,self.route[self.route_index])<.12:
                self.route_index+=1
            if self.route_index<len(self.route):
                return self.route[self.route_index]
        return self.GOAL

    def scan_lidar(self,forward):
        """Run the standalone AI model's simulated LiDAR/Kalman pipeline."""
        detections=self.ai_model.scan(self.obstacles,self.drone,forward,self.time)
        self._lidar_detections=self.ai_model.return_points
        for obstacle,_ in detections:
            if obstacle.name.startswith("Course hazard"):
                self.detected_course_hazards.add(obstacle.name)
        return detections
    def detect_sensor_hazard(self,dt=1.0/60.0):
        aim=self.route_aim()
        forward=unit(sub(aim,self.drone))
        if length(forward)<1e-8:
            heading=math.radians(self.heading)
            forward=(math.cos(heading),math.sin(heading),0.0)
        detections=self.scan_lidar(forward)
        self._sensor_prediction=assess_collision_risk(
            detections,self.drone,forward,getattr(self,"CRUISE_SPEED",35.0),
            self.SENSOR_RANGE,self.DRONE_RADIUS,self.SENSOR_SAFETY_CLEARANCE,
            cooldowns=self.avoidance_cooldowns,time_seconds=self.time,
            current_trigger_distance=self.SENSOR_REROUTE_DISTANCE)
        return self._sensor_prediction[0] if self._sensor_prediction else None
    def begin_sensor_avoidance(self,obstacle,emergency=False):
        self.avoidances+=1
        self.replan_count=getattr(self,"replan_count",0)+1
        if emergency:
            self.collision_risk_events=getattr(self,"collision_risk_events",0)+1
        self.avoidance_focus=obstacle
        self.sensor_target=obstacle
        track=self._kalman_tracks.get(obstacle)
        measured_position=track.position if track else obstacle.center
        self.sensor_reading=obstacle_surface_clearance(
            obstacle,self.drone,measured_position,self.DRONE_RADIUS)
        self.avoidance_cooldowns[obstacle]=self.time+6.0
        if obstacle.name.startswith("Course hazard"):
            self.detected_course_hazards.add(obstacle.name)
        self.velocity=0.0
        label="EMERGENCY STOP" if emergency else "2M SENSOR ALERT"
        planner_label=("QUICK D* REROUTE" if getattr(self,"mission_type",None)=="sensor"
                       else "A* REPLAN")
        self.set_status(
            f"{label} · {obstacle.name.upper()} · {self.sensor_reading:.1f}M GAP · {planner_label}",3.0)
        self.direct_target=None
        self.mode="replan"
        self._planning_obstacle=obstacle
        prediction=self._sensor_prediction
        self._planning_obstacle_center=(prediction[1] if prediction and prediction[0] is obstacle
                                        else measured_position)
        planning_started=perf_counter()
        try:
            if getattr(self,"mission_type",None)=="sensor":
                impact=(prediction[3] if prediction and prediction[0] is obstacle
                        else measured_position)
                rejoin_goal,baseline_suffix=self.sensor_rejoin_after(impact,60.0)
                path=self.plan_path_dstar(self.drone,rejoin_goal)
                if path and baseline_suffix:
                    if distance(path[-1],baseline_suffix[0])<1e-5:
                        baseline_suffix=baseline_suffix[1:]
                    path=path+baseline_suffix
                planner_name="D* LITE"
            else:
                path=self.plan_path(self.drone,self.GOAL)
                planner_name="A*"
        finally:
            planning_seconds=perf_counter()-planning_started
            self.replan_compute_total=getattr(self,"replan_compute_total",0.0)+planning_seconds
            self.replan_compute_samples=getattr(self,"replan_compute_samples",0)+1
            self._planning_obstacle=None
            self._planning_obstacle_center=None
        if path:
            self.route=path
            self.route_index=1 if len(path)>1 else 0
            self.route_planner=planner_name
            self.mode="follow"
            if planner_name=="D* LITE":
                self.resultant_route=list(path)
                self.set_status("SENSOR MODE · D* RESULTANT PATH ACTIVE",3.0)
            else:
                self.set_status(f"COLLISION RISK · {planner_name} ROUTE ACTIVE",3.0)
        else:
            self.route=[]
            self.mode="idle"
            self.mission_success=False
            self.set_status(f"{planner_name} ROUTE BLOCKED · HOLDING",3.0)

    def update(self,dt):
        dt=min(dt,.05)
        self.time+=dt
        self.animate_city()
        if self.status_timer>0:
            self.status_timer-=dt
        if self.contact_flash>0:
            self.contact_flash=max(0.0,self.contact_flash-dt)
        if self.paused:
            self.velocity=0.0
            return
        if self.mode in ("direct","follow"):
            obstacle=self.detect_sensor_hazard(dt)
            if obstacle and self.avoidance_cooldowns.get(obstacle,0.0)<=self.time:
                self.begin_sensor_avoidance(obstacle)
                return
        if self.mode=="direct" and self.direct_target:
            target=self.direct_target
            remain=distance(self.drone,target)
            if remain<=.05:
                self.drone=target
                self.mode="complete"
                self.velocity=0.0
                self.set_status("MISSION COMPLETE · SAFE LANDING",4.0)
                return
            previous=self.drone
            move=min(remain,3.15*dt)
            candidate=add(previous,mul(unit(sub(target,previous)),move))
            obstacle=self.collides(candidate,self.DRONE_RADIUS)
            if obstacle:
                self.begin_sensor_avoidance(obstacle,True)
                return
            self.drone=candidate
            self.velocity=move/max(dt,1e-5)
            self.heading=math.degrees(math.atan2(target[1]-previous[1],target[0]-previous[0]))
        elif self.mode=="follow" and self.route:
            while self.route_index<len(self.route) and distance(self.drone,self.route[self.route_index])<.12:
                self.route_index+=1
            if self.route_index>=len(self.route):
                self.drone=self.GOAL
                self.mode="complete"
                self.velocity=0.0
                self.set_status("SAFE REROUTE COMPLETE · LANDED",5.0)
                return
            target=self.route[self.route_index]
            previous=self.drone
            remain=distance(previous,target)
            move=min(remain,3.0*dt)
            candidate=add(previous,mul(unit(sub(target,previous)),move))
            obstacle=self.collides(candidate,self.DRONE_RADIUS)
            if obstacle:
                self.begin_sensor_avoidance(obstacle,True)
                return
            self.drone=candidate
            self.velocity=move/max(dt,1e-5)
            self.heading=math.degrees(math.atan2(target[1]-previous[1],target[0]-previous[0]))
        else:
            self.velocity=0.0
        if self.sensor_target and distance(self.drone,self.sensor_target.center)>self.SENSOR_RANGE+2.0:
            self.sensor_target=None

    def draw_world(self):
        vx=self.SIDEBAR
        vw=max(100,self.width-vx)
        glViewport(vx,0,vw,self.height)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        near=.1
        top=near*math.tan(math.radians(50)/2)
        aspect=vw/max(1.0,float(self.height))
        glFrustum(-top*aspect,top*aspect,-top,top,near,8500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        forward=(math.cos(math.radians(self.heading)),math.sin(math.radians(self.heading)),0.0)
        if self.camera_mode=="chase":
            target=add(self.drone,(forward[0]*5.0,forward[1]*5.0,-1.0))
            eye=add(sub(self.drone,mul(forward,22.0)),(0,0,8.0))
            up_ref=(0,0,1)
        elif self.camera_mode=="pilot":
            eye=add(self.drone,(forward[0]*1.4,forward[1]*1.4,.15))
            target=add(self.drone,(forward[0]*150.0,forward[1]*150.0,-.35))
            up_ref=(0,0,1)
        elif self.camera_mode=="top":
            target=(self.drone[0],self.drone[1],0.0)
            eye=(self.drone[0],self.drone[1],max(80.0,self.camera_distance*.85))
            up_ref=(0,1,0)
        elif self.camera_mode=="overview":
            target=(0.0,0.0,28.0)
            yaw=math.radians(self.camera_yaw)
            elev=math.radians(self.camera_elevation)
            horizontal=self.camera_distance*math.cos(elev)
            eye=(target[0]+math.sin(yaw)*horizontal,
                 target[1]-math.cos(yaw)*horizontal,
                 target[2]+self.camera_distance*math.sin(elev))
            up_ref=(0,0,1)
        else:
            target=(self.drone[0],self.drone[1],self.drone[2]-12.0)
            yaw=math.radians(self.camera_yaw)
            elev=math.radians(self.camera_elevation)
            horizontal=self.camera_distance*math.cos(elev)
            eye=(target[0]+math.sin(yaw)*horizontal,
                 target[1]-math.cos(yaw)*horizontal,
                 target[2]+self.camera_distance*math.sin(elev))
            up_ref=(0,0,1)
        f=unit(sub(target,eye))
        side=unit(cross(f,up_ref))
        up=cross(side,f)
        view=(side[0],up[0],-f[0],0.0,
              side[1],up[1],-f[1],0.0,
              side[2],up[2],-f[2],0.0,
              -dot(side,eye),-dot(up,eye),dot(f,eye),1.0)
        glLoadMatrixf((GLfloat*16)(*view))
        self.draw_floor()
        self.draw_world_route()
        self.draw_environment()
        self.draw_goal()
        self.draw_start()
        self.draw_sensor_rays()
        self.draw_drone()
        glViewport(0,0,self.width,self.height)

    def flat_quad(self,x1,y1,x2,y2,z,colour):
        rgb(colour)
        glBegin(GL_QUADS)
        glVertex3f(x1,y1,z)
        glVertex3f(x2,y1,z)
        glVertex3f(x2,y2,z)
        glVertex3f(x1,y2,z)
        glEnd()

    def draw_floor(self):
        glDisable(GL_DEPTH_TEST)
        glColor4f(.93,.945,.96,1.0)
        glBegin(GL_QUADS)
        for v in ((-2120,-545,-.05),(2120,-545,-.05),(2120,545,-.05),(-2120,545,-.05)):
            glVertex3f(*v)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        road_x=range(-2100,2101,210)
        road_y=range(-420,421,210)
        # Asphalt and raised sidewalk bands are batched for a smooth overview.
        glColor4f(.25,.285,.33,1.0)
        glBegin(GL_QUADS)
        for x in road_x:
            for v in ((x-18,-525,.005),(x+18,-525,.005),(x+18,525,.005),(x-18,525,.005)):
                glVertex3f(*v)
        for y in road_y:
            for v in ((-2100,y-18,.006),(2100,y-18,.006),(2100,y+18,.006),(-2100,y+18,.006)):
                glVertex3f(*v)
        glEnd()
        glColor4f(.70,.73,.76,1.0)
        glBegin(GL_QUADS)
        for x in road_x:
            for side in (-1,1):
                xa=x+side*22
                glVertex3f(xa-4,-525,.02); glVertex3f(xa+4,-525,.02)
                glVertex3f(xa+4,525,.02); glVertex3f(xa-4,525,.02)
        for y in road_y:
            for side in (-1,1):
                ya=y+side*22
                glVertex3f(-2100,ya-4,.021); glVertex3f(2100,ya-4,.021)
                glVertex3f(2100,ya+4,.021); glVertex3f(-2100,ya+4,.021)
        glEnd()
        # Kerbs, lane guides and dashed centre lines.
        glLineWidth(1.6)
        glColor4f(.88,.88,.84,.95)
        glBegin(GL_LINES)
        for x in road_x:
            for side in (-1,1):
                xx=x+side*18
                glVertex3f(xx,-525,.055); glVertex3f(xx,525,.055)
        for y in road_y:
            for side in (-1,1):
                yy=y+side*18
                glVertex3f(-2100,yy,.056); glVertex3f(2100,yy,.056)
        glEnd()
        glLineWidth(2.0)
        glColor4f(.94,.73,.27,.95)
        glBegin(GL_LINES)
        for x in road_x:
            for y in range(-495,496,42):
                glVertex3f(x,y,.065); glVertex3f(x,min(525,y+12),.065)
        for y in road_y:
            for x in range(-2070,2071,42):
                glVertex3f(x,y,.066); glVertex3f(min(2100,x+12),y,.066)
        glEnd()
        glLineWidth(1.3)
        glColor4f(.91,.93,.94,.88)
        glBegin(GL_LINES)
        for x in road_x:
            for lane in (-8,8):
                for y in range(-495,496,48):
                    glVertex3f(x+lane,y,.067); glVertex3f(x+lane,min(525,y+10),.067)
        for y in road_y:
            for lane in (-8,8):
                for x in range(-2070,2071,48):
                    glVertex3f(x,y+lane,.068); glVertex3f(min(2100,x+10),y+lane,.068)
        glEnd()
        # Large zebra crossings at the central junctions.
        glColor4f(.91,.93,.94,.94)
        glBegin(GL_QUADS)
        for x in (-1260,0,1260):
            for y in (-210,0,210):
                for stripe in range(-3,4):
                    offset=stripe*5.0
                    for side in (-1,1):
                        yy=y+side*24
                        glVertex3f(x-17,yy+offset-1,.075); glVertex3f(x+17,yy+offset-1,.075)
                        glVertex3f(x+17,yy+offset+1,.075); glVertex3f(x-17,yy+offset+1,.075)
                        xx=x+side*24
                        glVertex3f(xx+offset-1,y-17,.076); glVertex3f(xx+offset+1,y-17,.076)
                        glVertex3f(xx+offset+1,y+17,.076); glVertex3f(xx+offset-1,y+17,.076)
        glEnd()

    def draw_building_windows(self, obstacle, distance_from_camera):
        cx,cy,cz=obstacle.center
        hx,hy,hz=obstacle.size
        if distance_from_camera < 260.0:
            spacing=8.5
        elif distance_from_camera < 850.0:
            spacing=17.0
        else:
            spacing=31.0
        floors=max(2,min(18,int((2*hz)/spacing)))
        columns=max(3,min(18,int((2*hx)/spacing)))
        row_step=(2*hz)/(floors+1)
        col_step=(2*hx)/(columns+1)
        frame_w=min(3.3,col_step*.55)
        frame_h=min(4.4,row_step*.60)
        seed=sum(ord(c) for c in obstacle.name)
        glBegin(GL_QUADS)
        for floor in range(floors):
            z=cz-hz+(floor+1)*row_step
            for col in range(columns):
                wx=cx-hx+(col+1)*col_step
                wy=cy-hy+(col+1)*((2*hy)/(columns+1))
                tint=(.29,.48,.60) if (floor+col+seed)%7 else (.68,.63,.48)
                # Dark frames, reflective glass, and a narrow highlight on
                # each window give the facades more depth than flat grids.
                for side in (-1,1):
                    y=cy+side*(hy+.035)
                    glColor4f(.16,.21,.27,1.0)
                    glVertex3f(wx-frame_w/2,y,z-frame_h/2)
                    glVertex3f(wx+frame_w/2,y,z-frame_h/2)
                    glVertex3f(wx+frame_w/2,y,z+frame_h/2)
                    glVertex3f(wx-frame_w/2,y,z+frame_h/2)
                    y_glass=cy+side*(hy+.055)
                    glColor4f(tint[0],tint[1],tint[2],1.0)
                    glVertex3f(wx-frame_w*.39,y_glass,z-frame_h*.39)
                    glVertex3f(wx+frame_w*.39,y_glass,z-frame_h*.39)
                    glVertex3f(wx+frame_w*.39,y_glass,z+frame_h*.39)
                    glVertex3f(wx-frame_w*.39,y_glass,z+frame_h*.39)
                    glColor4f(.72,.82,.86,.78)
                    glVertex3f(wx-frame_w*.30,y_glass+side*.012,z+frame_h*.22)
                    glVertex3f(wx-frame_w*.05,y_glass+side*.012,z+frame_h*.22)
                    glVertex3f(wx-frame_w*.05,y_glass+side*.012,z+frame_h*.30)
                    glVertex3f(wx-frame_w*.30,y_glass+side*.012,z+frame_h*.30)

                    x=cx+side*(hx+.035)
                    glColor4f(.16,.21,.27,1.0)
                    glVertex3f(x,wy-frame_w/2,z-frame_h/2)
                    glVertex3f(x,wy+frame_w/2,z-frame_h/2)
                    glVertex3f(x,wy+frame_w/2,z+frame_h/2)
                    glVertex3f(x,wy-frame_w/2,z+frame_h/2)
                    x_glass=cx+side*(hx+.055)
                    glColor4f(tint[0],tint[1],tint[2],1.0)
                    glVertex3f(x_glass,wy-frame_w*.39,z-frame_h*.39)
                    glVertex3f(x_glass,wy+frame_w*.39,z-frame_h*.39)
                    glVertex3f(x_glass,wy+frame_w*.39,z+frame_h*.39)
                    glVertex3f(x_glass,wy-frame_w*.39,z+frame_h*.39)
        glEnd()

    def draw_rooftop_detail(self, obstacle, distance_from_camera):
        cx,cy,cz=obstacle.center
        hx,hy,hz=obstacle.size
        roof=cz+hz
        if distance_from_camera>900.0:
            return
        # Low parapet strips leave a clear landing surface in the heliport block.
        rail_h=.65
        draw_box((cx,cy-hy+1.0,roof+rail_h),(hx,1.0,rail_h),(.58,.64,.70))
        draw_box((cx,cy+hy-1.0,roof+rail_h),(hx,1.0,rail_h),(.58,.64,.70))
        draw_box((cx-hx+1.0,cy,roof+rail_h),(1.0,hy,rail_h),(.58,.64,.70))
        draw_box((cx+hx-1.0,cy,roof+rail_h),(1.0,hy,rail_h),(.58,.64,.70))
        if obstacle.name=="Rooftop heliport tower":
            draw_box((cx-34,cy+34,roof+5.0),(11,10,4.5),(.47,.54,.61))
        elif distance_from_camera<450.0:
            unit_x=min(14.0,hx*.22); unit_y=min(12.0,hy*.22)
            draw_box((cx-hx*.34,cy-hy*.31,roof+3.0),(unit_x,unit_y,2.6),(.54,.59,.64))
            draw_box((cx+hx*.30,cy+hy*.27,roof+2.2),(unit_x*.72,unit_y*.8,1.8),(.59,.63,.67))

    def draw_vehicle(self,obstacle):
        cx,cy,cz=obstacle.center
        hx,hy,hz=obstacle.size
        glPushMatrix()
        glTranslatef(cx,cy,cz)
        glRotatef(obstacle.heading,0,0,1)
        draw_box((0,0,0),(hx,hy,hz),obstacle.colour)
        draw_box((-.10,0,hz*.66),(hx*.47,hy*.72,hz*.42),(.64,.78,.84))
        # Windshield and rear glass.
        draw_box((hx*.36,0,hz*.68),(.035,hy*.66,hz*.29),(.44,.64,.74))
        draw_box((-hx*.51,0,hz*.68),(.035,hy*.66,hz*.29),(.44,.64,.74))
        for wx in (-hx*.58,hx*.58):
            for sy in (-1,1):
                draw_sphere((wx,sy*(hy+.01),-hz*.52),.17,(.16,.19,.23),10,6)
        draw_box((hx*.93,0,0),(.045,hy*.40,.12),(.96,.88,.68))
        glPopMatrix()

    def draw_aircraft_obstacle(self,obstacle):
        cx,cy,cz=obstacle.center
        arm=obstacle.size*.82
        glPushMatrix()
        glTranslatef(cx,cy,cz+.06*math.sin(self.time*4.0))
        glRotatef(obstacle.heading,0,0,1)
        arms=((arm,arm,0),(arm,-arm,0),(-arm,arm,0),(-arm,-arm,0))
        for p in arms:
            draw_tube((0,0,0),p,.065,(.28,.32,.37))
        draw_box((0,0,0),(.34,.27,.13),obstacle.colour)
        for px,py,_ in arms:
            draw_sphere((px,py,.05),.12,(.18,.23,.29),12,7)
            glPushMatrix()
            glTranslatef(px,py,.08)
            glRotatef(self.time*850+px*200+py*140,0,0,1)
            draw_box((0,0,0),(.43,.032,.014),(.20,.25,.31))
            glPopMatrix()
        glPopMatrix()

    def draw_street_furniture(self):
        focus=self.camera_focus()
        visibility=350.0+min(self.camera_distance*.25,700.0)
        # Larger, varied roadside trees make the city greenery visible from
        # the chase camera as well as from the full-city view.
        for tx,ty,height,crown,leaf_a,leaf_b,branch in self.trees:
            if math.hypot(tx-focus[0],ty-focus[1])>visibility:
                continue
            trunk_top=height*.73
            draw_tube((tx,ty,.10),(tx,ty,trunk_top),.56,(.34,.24,.15))
            draw_tube((tx,ty,height*.52),(tx+branch,ty+.35,height*.83),.25,(.36,.26,.16))
            draw_sphere((tx,ty,height*.82),crown*.72,leaf_a,16,10)
            draw_sphere((tx+branch,ty+.3,height*.98),crown*.54,leaf_b,14,9)
            draw_sphere((tx-branch*.65,ty-.45,height*.94),crown*.50,leaf_a,14,9)
        # Tall modern streetlights at regular intervals along the avenues.
        for x in range(-1890,1891,420):
            for y in range(-315,316,210):
                if math.hypot(x-focus[0],y-focus[1])>visibility:
                    continue
                draw_tube((x+27,y,.10),(x+27,y,9.0),.28,(.28,.34,.40))
                draw_tube((x+27,y,9.0),(x+32,y,10.0),.23,(.28,.34,.40))
                draw_box((x+34,y,10.0),(2.0,.65,.18),(.96,.84,.57))

    def draw_environment(self):
        focus=self.camera_focus()
        visibility=550.0+self.camera_distance*1.55
        for obstacle in self.obstacles:
            cx,cy,cz=obstacle.center
            d=math.hypot(cx-focus[0],cy-focus[1])
            if d>visibility:
                continue
            if obstacle.kind=="box":
                hx,hy,hz=obstacle.size
                is_building=obstacle.name.startswith("City block") or obstacle.name=="Rooftop heliport tower"
                if is_building:
                    if d<450.0:
                        draw_circle_xy((cx,cy,.06),max(hx,hy)*.92,(.30,.37,.44),1.5,.19)
                    draw_box(obstacle.center,obstacle.size,obstacle.colour)
                    self.draw_building_windows(obstacle,d)
                    self.draw_rooftop_detail(obstacle,d)
                else:
                    draw_box(obstacle.center,obstacle.size,obstacle.colour)
                    # High-contrast warning panels identify the suspended
                    # construction platforms as course hazards.
                    for side,colour in ((-1,(.16,.20,.24)),(1,(.98,.62,.13))):
                        draw_box((cx+side*hx*.55,cy-hy-.05,cz),(.42,.08,hz*.80),colour)
                    for side in (-1,1):
                        draw_sphere((cx+side*hx*.70,cy,cz+hz+.45),.58,(.98,.68,.17),10,6)
                    if obstacle is self.sensor_target:
                        draw_circle_xy((cx,cy,cz-hz-1.2),max(hx,hy)+4.0,(.98,.49,.17),2.0,.8)
            elif obstacle.kind=="vehicle":
                if d<1050.0:
                    self.draw_vehicle(obstacle)
                else:
                    draw_box(obstacle.center,obstacle.size,obstacle.colour)
            elif obstacle.kind=="aircraft":
                draw_circle_xy((obstacle.center[0],obstacle.center[1],obstacle.center[2]-1.5),obstacle.size+3.0,
                               (.93,.48,.23) if obstacle is self.sensor_target else (.34,.68,.78),1.5,.60)
                self.draw_aircraft_obstacle(obstacle)
            else:
                if obstacle.name.startswith("Course hazard"):
                    radius=obstacle.size+3.5
                    draw_circle_xy((cx,cy,cz-obstacle.size-.15),radius,
                                   (.98,.49,.17) if obstacle is self.sensor_target else (.90,.66,.22),
                                   1.6,.55)
                draw_sphere(obstacle.center,obstacle.size,obstacle.colour,14,8)
        self.draw_street_furniture()
        # Signal gantries at the principal downtown junctions.
        for ix in (-1260.0,0.0,1260.0):
            for iy in (-210.0,0.0,210.0):
                if math.hypot(ix-focus[0],iy-focus[1])>visibility:
                    continue
                pole_x=ix+23.0
                pole_y=iy+23.0
                draw_tube((pole_x,pole_y,.1),(pole_x,pole_y,11.0),.35,(.27,.33,.39))
                draw_tube((pole_x,pole_y,10.0),(pole_x-14.0,pole_y,10.0),.28,(.27,.33,.39))
                draw_box((pole_x-15.0,pole_y,9.0),(.9,.8,2.4),(.16,.20,.25))
                for offset,colour in ((10.0,(.88,.22,.18)),(9.0,(.91,.69,.18)),(8.0,(.19,.67,.39))):
                    draw_sphere((pole_x-15.0,pole_y-.86,offset),.36,colour,10,6)

    def draw_world_route(self):
        if len(self.route)<2:
            return
        if getattr(self,"mission_type",None)=="sensor":
            glLineWidth(1.4)
            glColor4f(.24,.57,.68,.20)
            glBegin(GL_LINES)
            for name,path in getattr(self,"path_candidates",{}).items():
                if name==getattr(self,"selected_path_name",None) or len(path)<2:
                    continue
                for a,b in zip(path,path[1:]):
                    glVertex3f(a[0],a[1],a[2]+1.2)
                    glVertex3f(b[0],b[1],b[2]+1.2)
            glEnd()
        dstar_route=getattr(self,"route_planner",None)=="D* LITE"
        # Keep the original sensor-mode A* route visible as a faint reference
        # after a risk is detected. The active orange route is the D* result
        # the drone is currently following.
        baseline=getattr(self,"baseline_route",[])
        if dstar_route and self.mode not in ("landing","complete") and len(baseline)>1:
            glLineWidth(2.0)
            glColor4f(.08,.63,.70,.30)
            glBegin(GL_LINES)
            for a,b in zip(baseline,baseline[1:]):
                glVertex3f(a[0],a[1],a[2]+1.0)
                glVertex3f(b[0],b[1],b[2]+1.0)
            glEnd()
        glLineWidth(4.0 if self.mode=="direct" or dstar_route else 3.2)
        if self.mode=="direct" or dstar_route:
            glColor4f(.96,.51,.17,.94)
            glBegin(GL_LINES)
            for a,b in zip(self.route,self.route[1:]):
                span=distance(a,b)
                dash=26.0
                period=52.0
                for start in range(0,int(span),int(period)):
                    p0=lerp(a,b,min(1.0,start/max(span,1e-6)))
                    p1=lerp(a,b,min(1.0,(start+dash)/max(span,1e-6)))
                    glVertex3f(p0[0],p0[1],p0[2]+2.0)
                    glVertex3f(p1[0],p1[1],p1[2]+2.0)
            glEnd()
        else:
            glColor4f(.08,.63,.70,.94)
            glBegin(GL_LINES)
            for a,b in zip(self.route,self.route[1:]):
                glVertex3f(a[0],a[1],a[2]+2.0)
                glVertex3f(b[0],b[1],b[2]+2.0)
            glEnd()
        glLineWidth(1.0)

    def draw_sensor_rays(self):
        if self.mode not in ("direct","follow","replan"):
            return
        x,y,z=self.drone
        heading=math.radians(self.heading)
        origin=(x+math.cos(heading)*.48,y+math.sin(heading)*.48,z)
        for offset in self.LIDAR_BEAM_ANGLES:
            angle=heading+math.radians(offset)
            end=(origin[0]+math.cos(angle)*self.SENSOR_RANGE,
                 origin[1]+math.sin(angle)*self.SENSOR_RANGE,
                 origin[2]+.03*math.cos(offset))
            colour=(.94,.54,.20) if offset==0 and self.sensor_target else (.17,.60,.73)
            alpha=.78 if offset==0 else .31
            draw_line(origin,end,colour,1.8 if offset==0 else 1.0,alpha)
        for _,point in self._lidar_detections:
            marker=2.0
            draw_line((point[0]-marker,point[1],point[2]),
                      (point[0]+marker,point[1],point[2]),(.10,.76,.83),1.5,.86)
            draw_line((point[0],point[1]-marker,point[2]),
                      (point[0],point[1]+marker,point[2]),(.10,.76,.83),1.5,.86)
        if self.sensor_target:
            track=self._kalman_tracks.get(self.sensor_target)
            target=track.position if track else self.sensor_target.center
            draw_line(origin,target,(.96,.48,.18),2.3,.95)
            if self.sensor_target.kind=="aircraft":
                draw_circle_xy((target[0],target[1],target[2]+.05),self.sensor_target.size+.18,
                               (.97,.49,.20),2.2,.9)

    def draw_ui(self):
        glViewport(0,0,self.width,self.height)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,self.width,0,self.height,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        h=self.height
        self.ui_rect(0,0,self.SIDEBAR,h,(.995,.998,1,1))
        self.ui_rect(self.SIDEBAR-1,0,1,h,(.84,.88,.92,1))
        self.ui_rect(20,h-191,282,48,(.94,.97,.99,1))
        state=(.13,.60,.45,1) if self.mode in ("idle","complete") else ((.92,.55,.16,1) if self.mode=="replan" else (.10,.58,.70,1))
        self.ui_rect(27,h-175,9,9,state)
        self.ui_rect(20,h-293,282,65,(.975,.984,.992,1))
        self.ui_rect(20,h-373,282,48,(.10,.47,.59,1))
        self.ui_rect(20,h-426,282,40,(.91,.95,.97,1))
        self.ui_rect(20,h-474,282,35,(.965,.974,.983,1))
        self.ui_rect(20,31,282,132,(.965,.976,.986,1))
        self.ui_rect(157,53,1,77,(.87,.90,.94,1))
        self.ui_rect(26,77,4,4,(.11,.62,.48,1))
        self.ui_rect(26,39,7,7,(.96,.48,.21,1))
        self.ui_rect(self.SIDEBAR+22,h-38,180,24,(.99,.995,1,.88))
        self.ui_rect(self.SIDEBAR+22,h-67,490,20,(.99,.995,1,.88))
        self.ui_rect(self.SIDEBAR+22,18,500,43,(.99,.995,1,.88))
        self.ui_rect(max(self.SIDEBAR+420,self.width-485),h-42,455,22,(.99,.995,1,.86))
        l=self.labels
        for name,y in (("brand",h-52),("tag",h-74),("live",h-54),("section",h-123),
                       ("status",h-164),("mission_head",h-224),("mission",h-250),
                       ("mission_sub",h-270),("button_contact",h-349),("button_auto",h-406),
                       ("button_reset",h-456)):
            l[name].y=y
        l["status"].text=self.status_msg if self.status_timer>0 else self.status_for_mode()
        l["status"].font_size=10 if len(l["status"].text)>22 else 13
        l["status"].color=(177,104,29,255) if self.mode=="replan" else (35,55,79,255)
        l["alt"].y=94; l["alt"].text=f"{self.drone[2]:.2f} m"
        l["speed"].y=94; l["speed"].text=f"{self.velocity:.1f} m/s"
        l["contact_head"].y=52; l["contact_head"].text="COURSE HAZARDS SENSED"
        l["contact_count"].x=143; l["contact_count"].y=49
        l["contact_count"].text=f"{len(self.detected_course_hazards):02d}/{self.COURSE_HAZARD_COUNT}"
        l["sensor_value"].x=196; l["sensor_value"].y=49
        l["sensor_value"].text=f"LAST DETECTION {self.sensor_reading:.1f}m" if self.avoidances else f"SENSOR RANGE {self.SENSOR_RANGE:.1f}m"
        l["controls"].y=24
        path_km=getattr(self,"flight_distance",0.0)/1000.0
        collision_count=getattr(self,"actual_collisions",0)
        collision_rate=collision_count/max(path_km,1e-6) if path_km>0 else 0.0
        replan_samples=getattr(self,"replan_compute_samples",0)
        average_replan_ms=(1000.0*self.replan_compute_total/replan_samples
                           if replan_samples else 0.0)
        mission_success=getattr(self,"mission_success",None)
        success_text="--" if mission_success is None else ("YES" if mission_success else "NO")
        l["metrics"].x=self.SIDEBAR+34; l["metrics"].y=32
        l["metrics"].text=(f"PATH {path_km:.2f} km  ·  COLLISIONS {collision_count}  ·  RATE {collision_rate:.2f}/km\n"
                           f"REPLANS {getattr(self,'replan_count',0)}  ·  AVG {average_replan_ms:.1f} ms  ·  MISSION {success_text}")
        l["world_title"].x=self.SIDEBAR+32; l["world_title"].y=h-31
        l["world_sub"].x=self.SIDEBAR+32; l["world_sub"].y=h-57
        l["goal_tag"].x=self.width-170; l["goal_tag"].y=45
        l["impact_tag"].x=self.SIDEBAR+28; l["impact_tag"].y=h-91
        if self.sensor_target and self.avoidances:
            l["impact_tag"].text=f"RANGE SENSOR  ·  {self.sensor_target.name.upper()}  ·  {self.sensor_reading:.1f}M  ·  AVOIDED"
        elif self.avoidances:
            l["impact_tag"].text=f"{self.avoidances} SENSOR AVOIDANCE EVENT(S)"
        else:
            l["impact_tag"].text=""
        view_names={"orbit":"1 ORBIT","chase":"2 CHASE","pilot":"3 PILOT",
                    "top":"4 TOP","overview":"5 CITY OVERVIEW"}
        l["camera"].x=max(self.SIDEBAR+430,self.width-470)
        l["camera"].y=h-31
        l["camera"].text="CAMERA  "+view_names[self.camera_mode]+"   ·   1–5 TO SWITCH"
        for label in l.values():
            label.draw()
        glEnable(GL_DEPTH_TEST)

    def status_for_mode(self):
        if self.paused:
            return "SIMULATION PAUSED"
        return {
            "idle":"READY FOR FLIGHT",
            "direct":"SENSOR DEMO · APPROACHING TRAFFIC",
            "replan":"RANGE SENSOR · REPLANNING",
            "follow":"FOLLOWING SENSOR-SAFE ROUTE",
            "complete":"SAFE ROUTE COMPLETE",
        }.get(self.mode,"READY FOR FLIGHT")

    def on_mouse_press(self,x,y,button,modifiers):
        if button==mouse.LEFT and x>self.SIDEBAR and y>self.height-70:
            self.cycle_camera()
            return
        super().on_mouse_press(x,y,button,modifiers)

    def on_mouse_drag(self,x,y,dx,dy,buttons,modifiers):
        if buttons & mouse.RIGHT:
            if self.camera_mode!="orbit":
                self.camera_mode="orbit"
                if self.camera_distance>1000.0:
                    self.camera_distance=200.0
                    self.camera_elevation=32.0
            self.camera_yaw+=dx*.34
            self.camera_elevation=clamp(self.camera_elevation+dy*.24,12,72)

    def on_mouse_scroll(self,x,y,scroll_x,scroll_y):
        self.camera_distance=clamp(self.camera_distance*(.88**scroll_y),80.0,6000.0)

    def cycle_camera(self):
        modes=("orbit","chase","pilot","top","overview")
        previous=self.camera_mode
        self.camera_mode=modes[(modes.index(self.camera_mode)+1)%len(modes)]
        if self.camera_mode=="overview":
            self.camera_distance=4600.0
            self.camera_elevation=56.0
        elif previous=="overview":
            self.camera_distance=200.0
            self.camera_elevation=32.0
        self.set_status("CAMERA VIEW · "+self.camera_mode.upper(),1.6)

    def on_key_press(self,symbol,modifiers):
        if symbol==key.C:
            self.run_contact_demo()
        elif symbol==key.A:
            self.run_safe_route()
        elif symbol==key.R:
            self.reset()
        elif symbol==key.SPACE:
            self.paused=not self.paused
            self.set_status("SIMULATION PAUSED" if self.paused else "SIMULATION RESUMED",1.8)
        elif symbol==key._1:
            if self.camera_distance>1000.0: self.camera_distance=200.0
            self.camera_elevation=32.0
            self.camera_mode="orbit"; self.set_status("CAMERA VIEW · ORBIT",1.5)
        elif symbol==key._2:
            if self.camera_distance>1000.0: self.camera_distance=220.0
            self.camera_mode="chase"; self.set_status("CAMERA VIEW · CHASE",1.5)
        elif symbol==key._3:
            if self.camera_distance>1000.0: self.camera_distance=220.0
            self.camera_mode="pilot"; self.set_status("CAMERA VIEW · PILOT",1.5)
        elif symbol==key._4:
            if self.camera_distance>1000.0: self.camera_distance=500.0
            self.camera_mode="top"; self.set_status("CAMERA VIEW · TOP-DOWN",1.5)
        elif symbol==key._5:
            self.camera_mode="overview"; self.camera_distance=4600.0
            self.camera_elevation=56.0
            self.set_status("CAMERA VIEW · FULL CITY OVERVIEW",1.8)
        elif symbol==key.ESCAPE:
            self.close()




class SkyRouteMission(SkyRouteCity):
    """Adds a complete takeoff, cruise, and landing sequence."""
    START_PAD = (-735.0, -105.0, .44)
    LANDING_PAD = (735.0, 105.0, 36.45)
    TAKEOFF_SPEED = 9.0
    CRUISE_SPEED = 35.0
    LANDING_SPEED = 8.0
    def __init__(self):
        super().__init__()
        self.mission_type="sensor"
        self.reset()
        self.set_caption("SkyRoute Urban 3D | Central Park-Scale City · UAV Mission")

    def make_labels(self):
        super().make_labels()
        self.labels["button_contact"].text="START SENSOR MISSION"
        self.labels["button_auto"].text="START SAFE MISSION"
        self.labels["mission"].text="PLAZA LAUNCH  →  ROOFTOP HELIPAD"
        self.labels["mission_sub"].text="AI A*: distance + turns + climb · 4 candidates"
        self.labels["world_sub"].text="4.41 km² · tree-lined streets · 64 moving cars · 12 aerial hazards"
        self.labels["controls"].text="C sensor mode  A safe mode  R reset  Space pause\n1 orbit  2 chase  3 pilot  4 top  5 city view · right-drag"

    def reset(self):
        super().reset()
        self.time=0.0
        self.drone=self.START_PAD
        self.heading=0.0
        self.velocity=0.0
        self.mode="idle"
        self.route=[]
        self.route_index=0
        self.baseline_route=[]
        self.path_candidates={}
        self.path_candidate_scores={}
        self.selected_path_name=None
        self.selected_path_cost=math.inf
        self.resultant_route=[]
        self.direct_target=None
        self.mission_type="sensor"
        self.route_planner="A*"
        self.sensor_reading=self.SENSOR_RANGE
        self.sensor_target=None
        self.avoidances=0
        self.mission_started_at=None
        self.mission_elapsed=0.0
        self.mission_success=None
        self.actual_collisions=0
        self.collision_risk_events=0
        self.replan_count=0
        self.replan_compute_total=0.0
        self.replan_compute_samples=0
        self.flight_distance=0.0
        self._collision_latched=False
        self.animate_city()
        self.camera_mode="chase"
        self.camera_distance=220.0
        self.camera_elevation=32.0
        self.set_status("READY · ON LAUNCH PAD",2.5)

    def run_contact_demo(self):
        self.reset()
        self.mission_type="sensor"
        self.camera_mode="chase"
        self.camera_distance=220.0
        self.camera_elevation=32.0
        self.mode="takeoff"
        self.mission_started_at=self.time
        self.set_status("SENSOR MODE · A* RESULTANT PATH · TAKING OFF",2.2)

    def run_safe_route(self):
        self.reset()
        self.mission_type="safe"
        self.camera_mode="chase"
        self.camera_distance=220.0
        self.mode="takeoff"
        self.mission_started_at=self.time
        self.set_status("SAFE MODE · A* ROUTE · TAKING OFF",2.2)

    def plan_sensor_resultant_path(self,start,goal):
        """Compare A* corridor candidates and keep the lowest-cost resultant path."""
        self.path_candidates={}
        self.path_candidate_scores={}
        self.selected_path_name=None
        self.selected_path_cost=math.inf

        dx=float(goal[0])-float(start[0])
        dy=float(goal[1])-float(start[1])
        horizontal=math.hypot(dx,dy)
        normal=(-dy/horizontal,dx/horizontal) if horizontal>1e-8 else (0.0,1.0)
        midpoint=lerp(start,goal,.5)

        candidates={}
        direct=self.plan_path(start,goal)
        if direct:
            candidates["DIRECT"]=direct

        def via(waypoint):
            first=self.plan_path(start,waypoint)
            second=self.plan_path(waypoint,goal)
            if not first or not second:
                return []
            return first+second[1:] if distance(first[-1],second[0])<1e-4 else first+second

        for name,offset in (("NORTH",110.0),("SOUTH",-110.0)):
            waypoint=(midpoint[0]+normal[0]*offset,
                      midpoint[1]+normal[1]*offset,midpoint[2])
            route=via(waypoint)
            if route:
                candidates[name]=route

        high_z=min(self.WORLD_MAX[2]-10.0,midpoint[2]+45.0)
        high_waypoint=(midpoint[0],midpoint[1],high_z)
        high_route=via(high_waypoint)
        if high_route:
            candidates["HIGH"]=high_route

        selected,path,scores=choose_best_path(candidates)
        self.path_candidates={name:candidates[name] for name in scores}
        self.path_candidate_scores=scores
        self.selected_path_name=selected
        self.selected_path_cost=scores[selected] if selected else math.inf
        return path

    def sensor_rejoin_after(self,impact,lead_distance):
        """Choose a point ahead on the original A* path and its remaining suffix."""
        baseline=self.baseline_route or [self.START,self.GOAL]
        if len(baseline)<2:
            return self.GOAL,[]

        def project_progress(point):
            best_distance=math.inf
            best_progress=0.0
            travelled=0.0
            for a,b in zip(baseline,baseline[1:]):
                segment=sub(b,a)
                segment_length=length(segment)
                if segment_length<1e-8:
                    continue
                fraction=clamp(dot(sub(point,a),segment)/(segment_length*segment_length),0.0,1.0)
                projection=lerp(a,b,fraction)
                error=distance(point,projection)
                if error<best_distance:
                    best_distance=error
                    best_progress=travelled+fraction*segment_length
                travelled+=segment_length
            return best_progress,travelled

        impact_progress,total_length=project_progress(impact)
        drone_progress,_=project_progress(self.drone)
        target_progress=min(total_length,max(impact_progress,drone_progress)+max(0.0,lead_distance))
        travelled=0.0
        for index,(a,b) in enumerate(zip(baseline,baseline[1:])):
            segment_length=distance(a,b)
            if segment_length<1e-8:
                continue
            if target_progress<=travelled+segment_length or index==len(baseline)-2:
                fraction=clamp((target_progress-travelled)/segment_length,0.0,1.0)
                rejoin=lerp(a,b,fraction)
                suffix=list(baseline[index+1:])
                if suffix and distance(rejoin,suffix[0])<1e-5:
                    suffix=suffix[1:]
                return rejoin,suffix
            travelled+=segment_length
        return self.GOAL,[]

    def begin_landing(self):
        self.drone=(self.GOAL[0],self.GOAL[1],max(self.GOAL[2],self.drone[2]))
        self.mode="landing"
        self.direct_target=self.LANDING_PAD
        self.route=[self.drone,self.LANDING_PAD]
        self.route_index=1
        self.heading=180.0
        self.set_status("LANDING APPROACH · DESCENDING TO PAD",3.2)

    def finish_landing(self):
        self.drone=self.LANDING_PAD
        self.velocity=0.0
        self.mode="complete"
        self.mission_success=True
        self.mission_elapsed=(self.time-self.mission_started_at
                              if self.mission_started_at is not None else 0.0)
        self.route=[self.GOAL,self.LANDING_PAD]
        self.route_index=len(self.route)
        self.set_status("LANDED · MOTORS IDLE",5.0)

    def move_drone(self,target,speed,dt):
        previous=self.drone
        remaining=distance(previous,target)
        if remaining<=.045:
            self.drone=target
            self.flight_distance+=remaining
            self.velocity=0.0
            return True
        move=min(remaining,speed*dt)
        candidate=add(previous,mul(unit(sub(target,previous)),move))
        obstacle=self.collides(candidate,self.DRONE_RADIUS)
        if obstacle:
            if self.mode in ("direct","follow"):
                self.begin_sensor_avoidance(obstacle,True)
            else:
                # Stop on the safe side and hold for a clear landing area.
                self.velocity=0.0
                self.set_status("LANDING ZONE BLOCKED · HOLDING",2.0)
            return False
        self.drone=candidate
        self.flight_distance+=move
        self.velocity=move/max(dt,1e-5)
        horizontal=sub(target,previous)
        if abs(horizontal[0])+abs(horizontal[1])>.001:
            self.heading=math.degrees(math.atan2(horizontal[1],horizontal[0]))
        return distance(self.drone,target)<=.045

    def update(self,dt):
        dt=min(dt,.05)
        if self.paused:
            if self.status_timer>0:
                self.status_timer=max(0.0,self.status_timer-dt)
            self.velocity=0.0
            return
        self.time+=dt
        self.animate_city()
        if self.status_timer>0:
            self.status_timer-=dt
        if self.contact_flash>0:
            self.contact_flash=max(0.0,self.contact_flash-dt)

        if self.mode in ("takeoff","follow","landing"):
            contact=self.collides(self.drone,self.DRONE_RADIUS)
            if contact:
                if not self._collision_latched:
                    self.actual_collisions+=1
                self._collision_latched=True
                if (self.mode=="follow" and self.mission_type=="sensor" and
                        self.avoidance_cooldowns.get(contact,0.0)<=self.time):
                    self.begin_sensor_avoidance(contact,True)
                    return
            else:
                self._collision_latched=False

        if self.mode=="takeoff":
            if self.move_drone(self.START,self.TAKEOFF_SPEED,dt):
                if self.mission_type=="sensor":
                    path=self.plan_sensor_resultant_path(self.START,self.GOAL)
                else:
                    path=self.plan_path(self.START,self.GOAL)
                planner_name="A*"
                if path:
                    self.route=path
                    self.route_index=1 if len(path)>1 else 0
                    self.route_planner=planner_name
                    self.baseline_route=list(path) if self.mission_type=="sensor" else []
                    self.resultant_route=[]
                    self.mode="follow"
                    if self.mission_type=="sensor":
                        self.set_status(
                            f"A* COST PICK · {self.selected_path_name} · "
                            f"{self.selected_path_cost:.0f} · 2M SENSOR LIVE",3.0)
                    else:
                        self.set_status("SAFE MODE · A* ROUTE ACTIVE",2.5)
                else:
                    self.mode="idle"
                    self.drone=self.START_PAD
                    self.mission_success=False
                    self.set_status("NO A* RESULTANT ROUTE · RETURNED TO PAD",3.0)
            return

        if self.mode in ("direct","follow"):
            obstacle=self.detect_sensor_hazard(dt)
            if obstacle and self.avoidance_cooldowns.get(obstacle,0.0)<=self.time:
                self.begin_sensor_avoidance(obstacle)
                return

        if self.mode=="direct" and self.direct_target:
            if distance(self.drone,self.direct_target)<=.05:
                self.drone=self.direct_target
                self.begin_landing()
                return
            self.move_drone(self.direct_target,self.CRUISE_SPEED,dt)
        elif self.mode=="follow" and self.route:
            while self.route_index<len(self.route) and distance(self.drone,self.route[self.route_index])<.12:
                self.route_index+=1
            if self.route_index>=len(self.route):
                self.begin_landing()
                return
            self.move_drone(self.route[self.route_index],self.CRUISE_SPEED,dt)
        elif self.mode=="landing":
            if self.move_drone(self.LANDING_PAD,self.LANDING_SPEED,dt):
                self.finish_landing()
                return
        else:
            self.velocity=0.0

        if self.sensor_target and distance(self.drone,self.sensor_target.center)>self.SENSOR_RANGE+2.0:
            self.sensor_target=None

    def status_for_mode(self):
        if self.paused:
            return "SIMULATION PAUSED"
        if self.mode=="follow" and self.mission_type=="safe":
            return "FOLLOWING SAFE A* ROUTE"
        if self.mode=="follow" and self.mission_type=="sensor":
            return ("FOLLOWING D* RESULTANT PATH · SENSORS ACTIVE"
                    if self.route_planner=="D* LITE"
                    else "FOLLOWING A* RESULTANT PATH · SENSORS ACTIVE")
        return {
            "idle":"READY · ON LAUNCH PAD",
            "takeoff":"TAKING OFF · CLIMBING TO ROUTE ALTITUDE",
            "direct":"CRUISING · SENSORS ACTIVE",
            "replan":"RANGE SENSOR · REPLANNING",
            "follow":"FOLLOWING ROUTE",
            "landing":"LANDING · DESCENDING TO PAD",
            "complete":"LANDED · MOTORS IDLE",
        }.get(self.mode,"READY · ON LAUNCH PAD")

    def draw_pad(self, center, colour, rooftop=False):
        x,y,z=center
        pad_z=z+.045
        half=10.0
        glColor4f(.16,.20,.24,.92)
        glBegin(GL_QUADS)
        glVertex3f(x-half,y-half,pad_z); glVertex3f(x+half,y-half,pad_z)
        glVertex3f(x+half,y+half,pad_z); glVertex3f(x-half,y+half,pad_z)
        glEnd()
        draw_circle_xy((x,y,pad_z+.02),8.0,colour,2.8,.98)
        draw_circle_xy((x,y,pad_z+.025),6.2,colour,1.2,.78)
        # Large H marking remains visible from the city overview.
        draw_line((x-3.0,y-3.2,pad_z+.04),(x-3.0,y+3.2,pad_z+.04),(.97,.98,.98),2.6)
        draw_line((x+3.0,y-3.2,pad_z+.04),(x+3.0,y+3.2,pad_z+.04),(.97,.98,.98),2.6)
        draw_line((x-3.0,y,pad_z+.04),(x+3.0,y,pad_z+.04),(.97,.98,.98),2.6)
        if rooftop:
            draw_line((x-8.0,y-8.0,pad_z+.04),(x+8.0,y-8.0,pad_z+.04),colour,1.3,.85)
            draw_line((x-8.0,y+8.0,pad_z+.04),(x+8.0,y+8.0,pad_z+.04),colour,1.3,.85)

    def draw_start(self):
        self.draw_pad(self.START_PAD,(.23,.75,.83))

    def draw_goal(self):
        self.draw_pad(self.LANDING_PAD,(.18,.78,.51),rooftop=True)



if __name__=="__main__":
    app=SkyRouteMission()
    pyglet.app.run()












