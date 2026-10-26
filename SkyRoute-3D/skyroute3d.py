"""SkyRoute 3D — native OpenGL UAV path-planning simulation."""
from __future__ import annotations
import heapq
import math
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "vendor"))
import pyglet
pyglet.options["shadow_window"] = False
pyglet.options["debug_gl"] = False
from pyglet.gl import *
from pyglet.gl.gl_compat import *
from pyglet.window import key, mouse


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
        step=self.GRID_STEP; lo,hi=self.WORLD_MIN,self.WORLD_MAX
        dims=tuple(int(round((hi[i]-lo[i])/step))+1 for i in range(3))
        def to_cell(p): return tuple(int(round((p[i]-lo[i])/step)) for i in range(3))
        def to_point(c): return tuple(lo[i]+c[i]*step for i in range(3))
        start_cell,goal_cell=to_cell(start),to_cell(goal)
        if self.collides(start,.25) or self.collides(goal,.25): return []
        offsets=[(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)!=(0,0,0)]
        opened=[(0.0,0.0,start_cell)]; came={}; costs={start_cell:0.0}; closed=set()
        while opened:
            _,current_cost,current=heapq.heappop(opened)
            if current in closed: continue
            if current==goal_cell:
                cells=[current]
                while current in came:
                    current=came[current]; cells.append(current)
                cells.reverse()
                raw=[start]+[to_point(c) for c in cells[1:-1]]+[goal]
                smooth=[raw[0]]; anchor=0
                while anchor<len(raw)-1:
                    furthest=anchor+1
                    for candidate in range(len(raw)-1,anchor,-1):
                        if self.segment_clear(raw[anchor],raw[candidate]):
                            furthest=candidate; break
                    smooth.append(raw[furthest]); anchor=furthest
                return smooth
            closed.add(current)
            for off in offsets:
                nxt=tuple(current[i]+off[i] for i in range(3))
                if any(nxt[i]<0 or nxt[i]>=dims[i] for i in range(3)) or nxt in closed: continue
                point=to_point(nxt)
                if self.collides(point,.53): continue
                edge=math.sqrt(sum(v*v for v in off))*step
                new_cost=current_cost+edge
                if new_cost>=costs.get(nxt,float("inf")): continue
                costs[nxt]=new_cost; came[nxt]=current
                heapq.heappush(opened,(new_cost+distance(point,goal),new_cost,nxt))
        return []

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
        if self.mode=="direct":
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
        draw_circle_xy((x,y,.055),.53,(.42,.51,.60),1,.25)
        glPushMatrix(); glTranslatef(x,y,z+.07*math.sin(self.time*5)); glRotatef(self.heading,0,0,1)
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
            glRotatef(self.time*1100+px*300+py*240,0,0,1)
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
    """City object with optional smooth back-and-forth motion."""
    def __init__(self, name, kind, center, size, colour, motion=None):
        super().__init__(name, kind, center, size, colour, False)
        self.motion = motion
        self.home = center
        self.heading = 0.0


class SkyRouteCity(SkyRoute3D):
    """Urban sensor-avoidance scenario with moving traffic and camera presets."""
    START = (-13.0, 0.0, 3.3)
    GOAL = (13.0, 0.0, 3.3)
    WORLD_MIN = (-15.0, -12.0, 0.75)
    WORLD_MAX = (15.0, 12.0, 10.5)
    GRID_STEP = 0.82
    SENSOR_RANGE = 7.4

    def __init__(self):
        self.camera_mode = "orbit"
        self.avoidances = 0
        self.sensor_target = None
        self.avoidance_focus = None
        self.avoidance_cooldowns = {}
        self._planning_mode = False
        super().__init__()
        self.set_caption("SkyRoute Urban 3D | Sensor-Based UAV Avoidance")

    def make_obstacles(self):
        buildings = [
            UrbanObstacle("West tower", "box", (-10.2, 8.0, 3.2), (2.25, 2.35, 3.2), (.72,.78,.84)),
            UrbanObstacle("North apartments", "box", (-4.8, 8.4, 4.4), (2.0, 2.05, 4.4), (.82,.77,.70)),
            UrbanObstacle("North hotel", "box", (5.7, 8.3, 5.1), (2.15, 2.2, 5.1), (.72,.83,.84)),
            UrbanObstacle("East tower", "box", (11.0, 7.7, 4.0), (2.2, 2.4, 4.0), (.79,.80,.87)),
            UrbanObstacle("South offices", "box", (-10.4, -8.2, 4.1), (2.25, 2.3, 4.1), (.82,.83,.85)),
            UrbanObstacle("South market", "box", (-4.7, -8.4, 2.8), (2.0, 2.0, 2.8), (.87,.78,.68)),
            UrbanObstacle("Riverfront block", "box", (5.7, -8.1, 3.8), (2.1, 2.25, 3.8), (.71,.79,.83)),
            UrbanObstacle("South apartments", "box", (11.0, -8.0, 4.7), (2.2, 2.4, 4.7), (.83,.77,.73)),
            UrbanObstacle("Opposing UAV", "aircraft", (0.8, 0.0, 3.3), .67, (.91,.36,.24),
                          motion=(1,-1.35,1.35,.62,math.pi/2)),
            UrbanObstacle("Westbound taxi", "vehicle", (-7.0,-1.35,.64), (.92,.43,.46), (.22,.48,.70),
                          motion=(0,-11.0,11.0,.20,0.0)),
            UrbanObstacle("Eastbound van", "vehicle", (7.0,1.35,.64), (1.05,.48,.50), (.82,.53,.25),
                          motion=(0,-11.0,11.0,.17,math.pi)),
            UrbanObstacle("Cross-street car", "vehicle", (1.1,-7.0,.64), (.92,.43,.46), (.38,.61,.48),
                          motion=(1,-8.7,8.7,.20,1.0)),
            UrbanObstacle("Sidewalk robot", "sphere", (-8.0,3.35,.50), .32, (.93,.63,.25),
                          motion=(0,-11.0,11.0,.15,2.0)),
        ]
        return buildings

    def make_labels(self):
        super().make_labels()
        l=self.labels
        l["button_contact"].text="RUN SENSOR AVOIDANCE"
        l["button_contact"].font_size=9
        l["mission"].text="START  →  SAFE LANDING"
        l["mission_sub"].text="Live downtown traffic course"
        l["contact_head"].text="SENSOR AVOIDANCES"
        l["world_title"].text="DOWNTOWN FLIGHT CORRIDOR"
        l["world_sub"].text="Moving traffic  ·  forward range sensors  ·  predictive rerouting"
        l["controls"].text="C sensor demo  A safe route  R reset  Space pause\n1 orbit  2 chase  3 pilot  4 top  ·  right-drag to orbit"
        l["impact_tag"].color=(177,104,29,255)
        l["camera"]=pyglet.text.Label("CAMERA  1 ORBIT  ·  2 CHASE  ·  3 PILOT  ·  4 TOP",
            x=0,y=0,font_name="Arial",font_size=8,weight="bold",color=(81,101,124,255))
        l["sensor_value"]=pyglet.text.Label("RANGE 7.4m",x=186,y=49,font_name="Arial",font_size=8,weight="bold",color=(81,101,124,255))

    def reset(self):
        super().reset()
        self.avoidances=0
        self.sensor_reading=self.SENSOR_RANGE
        self.sensor_target=None
        self.avoidance_focus=None
        self.avoidance_cooldowns.clear()

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
        axis,lower,upper,speed,phase=obstacle.motion
        center=list(obstacle.home)
        u=(math.sin((self.time+horizon)*speed+phase)+1.0)*.5
        center[axis]=lower+u*(upper-lower)
        return tuple(center)

    def collides(self,point,clearance=0.0):
        px,py,pz=point
        for obstacle in self.obstacles:
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

    def animate_city(self):
        for obstacle in self.obstacles:
            if not obstacle.motion:
                continue
            axis,lower,upper,speed,phase=obstacle.motion
            angle=self.time*speed+phase
            u=(math.sin(angle)+1.0)*.5
            center=list(obstacle.home)
            center[axis]=lower+u*(upper-lower)
            obstacle.center=tuple(center)
            direction=math.cos(angle)
            if axis==0:
                obstacle.heading=0.0 if direction>=0 else 180.0
            else:
                obstacle.heading=90.0 if direction>=0 else -90.0

    def route_aim(self):
        if self.mode=="direct":
            return self.direct_target or self.GOAL
        if self.mode=="follow" and self.route:
            while self.route_index<len(self.route) and distance(self.drone,self.route[self.route_index])<.12:
                self.route_index+=1
            if self.route_index<len(self.route):
                return self.route[self.route_index]
        return self.GOAL

    def detect_sensor_hazard(self):
        aim=self.route_aim()
        forward=unit(sub(aim,self.drone))
        nearest=None
        nearest_distance=self.SENSOR_RANGE+1
        for obstacle in self.obstacles:
            center=self.predicted_center(obstacle,.25)
            delta=sub(center,self.drone)
            along=dot(delta,forward)
            if along<=0 or along>self.SENSOR_RANGE:
                continue
            side=length(sub(delta,mul(forward,along)))
            if obstacle.kind=="aircraft":
                vertical_extent=obstacle.size+1.0
                lateral_extent=obstacle.size+1.25
            elif obstacle.kind=="vehicle":
                vertical_extent=obstacle.size[2]+.9
                lateral_extent=max(obstacle.size[0],obstacle.size[1])+1.0
            elif obstacle.kind=="box":
                vertical_extent=obstacle.size[2]+.55
                lateral_extent=max(obstacle.size[0],obstacle.size[1])+.75
            else:
                vertical_extent=obstacle.size+.8
                lateral_extent=obstacle.size+.9
            if abs(delta[2])>vertical_extent or side>lateral_extent:
                continue
            d=length(delta)
            if d<nearest_distance:
                nearest,nearest_distance=obstacle,d
        return nearest

    def begin_sensor_avoidance(self,obstacle,emergency=False):
        self.drone=self.drone
        self.avoidances+=1
        self.avoidance_focus=obstacle
        self.sensor_target=obstacle
        self.sensor_reading=distance(self.drone,self.predicted_center(obstacle,.25))
        self.avoidance_cooldowns[obstacle]=self.time+4.5
        self.velocity=0.0
        label="PROXIMITY WARNING" if emergency else "RANGE SENSOR DETECTED"
        self.set_status(f"{label} · {obstacle.name.upper()} AT {self.sensor_reading:.1f}M · REROUTING",3.0)
        self.direct_target=None
        self.mode="replan"
        path=self.plan_path(self.drone,self.GOAL)
        if path:
            self.route=path
            self.route_index=1 if len(path)>1 else 0
            self.mode="follow"
        else:
            self.route=[]
            self.mode="idle"
            self.set_status("ROUTE BLOCKED · RESET TO TRY AGAIN",3.0)

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
            obstacle=self.detect_sensor_hazard()
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
        glFrustum(-top*aspect,top*aspect,-top,top,near,150.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        forward=(math.cos(math.radians(self.heading)),math.sin(math.radians(self.heading)),0.0)
        if self.camera_mode=="chase":
            target=add(self.drone,(forward[0]*2.8,forward[1]*2.8,.75))
            eye=add(sub(self.drone,mul(forward,7.5)),(0,0,4.0))
            up_ref=(0,0,1)
        elif self.camera_mode=="pilot":
            eye=add(self.drone,(forward[0]*.48,forward[1]*.48,.10))
            target=add(self.drone,(forward[0]*15.0,forward[1]*15.0,-.20))
            up_ref=(0,0,1)
        elif self.camera_mode=="top":
            target=(self.drone[0],self.drone[1],0.0)
            eye=(self.drone[0],self.drone[1],max(18.0,self.camera_distance*.85))
            up_ref=(0,1,0)
        else:
            target=(0.0,0.0,2.3)
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
        glColor4f(.94,.955,.972,1.0)
        glBegin(GL_QUADS)
        for v in ((-16,-14,-.05),(16,-14,-.05),(16,14,-.05),(-16,14,-.05)):
            glVertex3f(*v)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        # Avenues and sidewalks create a recognisable downtown street grid.
        self.flat_quad(-15.5,-2.65,15.5,2.65,.005,(.31,.35,.40))
        self.flat_quad(-1.45,-13,3.65,13,.012,(.34,.38,.42))
        self.flat_quad(-15.5,-3.45,15.5,-2.65,.025,(.76,.79,.82))
        self.flat_quad(-15.5,2.65,15.5,3.45,.025,(.76,.79,.82))
        self.flat_quad(-2.25,-13,-1.45,13,.028,(.75,.79,.82))
        self.flat_quad(3.65,-13,4.45,13,.028,(.75,.79,.82))
        # Kerb edges.
        for y in (-2.68,2.62):
            draw_line((-15.5,y,.06),(15.5,y,.06),(.91,.92,.92),1.4,.9)
        for x in (-1.48,3.62):
            draw_line((x,-13,.06),(x,13,.06),(.91,.92,.92),1.4,.9)
        # Painted yellow centerlines and white lane dashes.
        glLineWidth(2.0)
        glColor4f(.95,.79,.32,.95)
        glBegin(GL_LINES)
        for x in range(-15,16,2):
            glVertex3f(x,0,.055)
            glVertex3f(min(15,x+1.05),0,.055)
        glEnd()
        glLineWidth(1.3)
        glColor4f(.93,.95,.96,.82)
        glBegin(GL_LINES)
        for y in (-1.35,1.35):
            for x in range(-15,16,3):
                glVertex3f(x,y,.056)
                glVertex3f(min(15,x+1.45),y,.056)
        for y in range(-12,13,3):
            glVertex3f(1.1,y,.058)
            glVertex3f(1.1,min(13,y+1.4),.058)
        glEnd()
        # Crosswalks at the main intersection.
        glColor4f(.91,.93,.94,.92)
        glBegin(GL_QUADS)
        for i in range(7):
            x=-2.9+i*.63
            glVertex3f(x,-2.35,.06); glVertex3f(x+.32,-2.35,.06)
            glVertex3f(x+.32,-1.70,.06); glVertex3f(x,-1.70,.06)
            glVertex3f(x,1.70,.06); glVertex3f(x+.32,1.70,.06)
            glVertex3f(x+.32,2.35,.06); glVertex3f(x,2.35,.06)
        glEnd()
        # Faint paving joints.
        glColor4f(.68,.73,.78,.38)
        glBegin(GL_LINES)
        for x in range(-15,16,2):
            for y in (-3.35,3.35):
                glVertex3f(x,y,.031); glVertex3f(x+1.5,y,.031)
        for y in range(-12,13,2):
            for x in (-2.15,4.35):
                glVertex3f(x,y,.032); glVertex3f(x,y+1.5,.032)
        glEnd()

    def draw_building_windows(self,obstacle):
        cx,cy,cz=obstacle.center
        hx,hy,hz=obstacle.size
        floors=max(2,int((2*hz)/.78))
        columns=max(3,int((2*hx)/.72))
        glColor4f(.39,.61,.72,.86)
        glBegin(GL_QUADS)
        for floor in range(floors):
            z=cz-hz+.42+floor*.76
            if z>cz+hz-.20:
                continue
            for col in range(columns):
                wx=cx-hx+.35+col*((2*hx-.7)/max(1,columns-1))
                ww=.25
                wh=.27
                # Street-facing front and rear façades.
                for y in (cy-hy-.012,cy+hy+.012):
                    glVertex3f(wx-ww,y,z-wh); glVertex3f(wx+ww,y,z-wh)
                    glVertex3f(wx+ww,y,z+wh); glVertex3f(wx-ww,y,z+wh)
                # Side façades.
                wy=cy-hy+.38+col*((2*hy-.76)/max(1,columns-1))
                for x in (cx-hx-.012,cx+hx+.012):
                    glVertex3f(x,wy-.20,z-wh); glVertex3f(x,wy+.20,z-wh)
                    glVertex3f(x,wy+.20,z+wh); glVertex3f(x,wy-.20,z+wh)
        glEnd()
        # A simple lit crown gives taller buildings a skyline silhouette.
        if hz>3.5:
            draw_box((cx,cy,cz+hz+.08),(hx*.96,hy*.96,.08),(.68,.73,.80))

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
        glPushMatrix()
        glTranslatef(cx,cy,cz+.06*math.sin(self.time*4.0))
        glRotatef(obstacle.heading,0,0,1)
        arms=((.55,.55,0),(.55,-.55,0),(-.55,.55,0),(-.55,-.55,0))
        for p in arms:
            draw_tube((0,0,0),p,.045,(.28,.32,.37))
        draw_box((0,0,0),(.25,.20,.10),obstacle.colour)
        for px,py,_ in arms:
            draw_sphere((px,py,.04),.09,(.18,.23,.29),10,6)
            glPushMatrix()
            glTranslatef(px,py,.08)
            glRotatef(self.time*850+px*200+py*140,0,0,1)
            draw_box((0,0,0),(.31,.025,.012),(.20,.25,.31))
            glPopMatrix()
        glPopMatrix()

    def draw_street_furniture(self):
        # Street trees on the pavements.
        for x in (-13,-9,-5,7,11,14):
            for side in (-1,1):
                y=side*3.12
                draw_tube((x,y,.08),(x,y,1.55),.075,(.42,.34,.25))
                draw_sphere((x,y,2.02),.72,(.51,.65,.49),14,8)
                draw_sphere((x+.28,y-.08,2.22),.43,(.57,.70,.55),12,7)
        # Modern street lamps and curved-looking arms.
        for x in (-12,-7,-2,5,10,14):
            for side in (-1,1):
                y=side*3.18
                draw_tube((x,y,.06),(x,y,3.25),.045,(.34,.40,.46))
                draw_tube((x,y,3.25),(x+.45,y,3.45),.04,(.34,.40,.46))
                draw_box((x+.50,y,3.44),(.26,.10,.055),(.96,.91,.68))

    def draw_environment(self):
        for obstacle in self.obstacles:
            if obstacle.kind=="box":
                cx,cy,cz=obstacle.center
                hx,hy,hz=obstacle.size
                draw_circle_xy((cx,cy,.04),max(hx,hy)*.86,(.39,.46,.53),1.0,.14)
                draw_box(obstacle.center,obstacle.size,obstacle.colour)
                self.draw_building_windows(obstacle)
            elif obstacle.kind=="vehicle":
                self.draw_vehicle(obstacle)
            elif obstacle.kind=="aircraft":
                draw_circle_xy((obstacle.center[0],obstacle.center[1],obstacle.center[2]-.7),obstacle.size+1.0,
                               (.93,.48,.23) if obstacle is self.sensor_target else (.34,.68,.78),1.5,.60)
                self.draw_aircraft_obstacle(obstacle)
            else:
                draw_sphere(obstacle.center,obstacle.size,obstacle.colour,14,8)
        self.draw_street_furniture()
        # Traffic signal assembly at the centre of the cross-street.
        draw_tube((-2.75,3.12,.08),(-2.75,3.12,3.05),.065,(.30,.36,.42))
        draw_tube((-2.75,3.12,3.0),(-1.55,3.12,3.0),.05,(.30,.36,.42))
        draw_box((-1.50,3.12,2.82),(.18,.18,.48),(.20,.24,.29))
        draw_sphere((-1.50,3.12,3.12),.085,(.91,.30,.24),10,6)
        draw_sphere((-1.50,3.12,2.53),.085,(.22,.68,.43),10,6)

    def draw_sensor_rays(self):
        if self.mode not in ("direct","follow","replan"):
            return
        x,y,z=self.drone
        heading=math.radians(self.heading)
        origin=(x+math.cos(heading)*.48,y+math.sin(heading)*.48,z)
        for offset in (-32,-18,0,18,32):
            angle=heading+math.radians(offset)
            end=(origin[0]+math.cos(angle)*self.SENSOR_RANGE,
                 origin[1]+math.sin(angle)*self.SENSOR_RANGE,
                 origin[2]+.03*math.cos(offset))
            colour=(.94,.54,.20) if offset==0 and self.sensor_target else (.17,.60,.73)
            alpha=.78 if offset==0 else .31
            draw_line(origin,end,colour,1.8 if offset==0 else 1.0,alpha)
        if self.sensor_target:
            target=self.sensor_target.center
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
        l["contact_head"].y=52; l["contact_head"].text="SENSOR AVOIDANCES"
        l["contact_count"].y=49; l["contact_count"].text=str(self.avoidances)
        l["sensor_value"].x=186; l["sensor_value"].y=49
        l["sensor_value"].text=f"LAST DETECTION {self.sensor_reading:.1f}m" if self.avoidances else f"SENSOR RANGE {self.SENSOR_RANGE:.1f}m"
        l["controls"].y=24
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
        view_names={"orbit":"1 ORBIT","chase":"2 CHASE","pilot":"3 PILOT","top":"4 TOP"}
        l["camera"].x=max(self.SIDEBAR+430,self.width-470)
        l["camera"].y=h-31
        l["camera"].text="CAMERA  "+view_names[self.camera_mode]+"   ·   1–4 TO SWITCH"
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
            self.camera_yaw+=dx*.34
            self.camera_elevation=clamp(self.camera_elevation+dy*.24,12,72)

    def on_mouse_scroll(self,x,y,scroll_x,scroll_y):
        self.camera_distance=clamp(self.camera_distance-scroll_y*1.7,17.0,48.0)

    def cycle_camera(self):
        modes=("orbit","chase","pilot","top")
        self.camera_mode=modes[(modes.index(self.camera_mode)+1)%len(modes)]
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
            self.camera_mode="orbit"; self.set_status("CAMERA VIEW · ORBIT",1.5)
        elif symbol==key._2:
            self.camera_mode="chase"; self.set_status("CAMERA VIEW · CHASE",1.5)
        elif symbol==key._3:
            self.camera_mode="pilot"; self.set_status("CAMERA VIEW · PILOT",1.5)
        elif symbol==key._4:
            self.camera_mode="top"; self.set_status("CAMERA VIEW · TOP-DOWN",1.5)
        elif symbol==key.ESCAPE:
            self.close()



if __name__=="__main__":
    app=SkyRouteCity()
    pyglet.app.run()










