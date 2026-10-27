"""
Smart Traffic Signal Simulator
==============================
DSA project in Python. The window is built with Tkinter, which comes with
Python, so nothing needs to be installed.

Data structures and algorithms used
-----------------------------------
1. QUEUE (collections.deque)
   Normal cars waiting on each road. First come, first served (FIFO).
2. PRIORITY QUEUE (heapq, a min-heap)
   Ambulances and fire trucks on each road. The most urgent vehicle comes
   out first, even if normal cars arrived earlier.
3. STACK (a Python list with append / pop)
   Signal log. The latest signal change is on top (LIFO).
4. SEARCHING (max with a score function)
   The controller searches for the road that needs the green light most.
5. GREEDY ALGORITHM
   Green time is given according to how many vehicles are waiting.

Run:
    python traffic_simulator.py             -> opens the simulator window
    python traffic_simulator.py --compare   -> prints Smart vs Fixed results
"""

import heapq
import random
import sys
from collections import deque

try:
    import tkinter as tk
except ImportError:          # the text comparison (--compare) works without it
    tk = None

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
ROADS = ["North", "East", "South", "West"]

TPS = 20                    # simulation ticks per second
TICK_MS = 1000 // TPS       # milliseconds between ticks (50 ms)

YELLOW_TICKS = 30           # 1.5 s yellow
ALL_RED_TICKS = 20          # 1.0 s all red (junction clears)
MIN_GREEN = 80              # 4 s minimum green (smart mode)
MAX_GREEN = 300             # 15 s maximum green (smart mode)
PER_CAR = 20                # +1 s of green for every waiting vehicle
FIXED_GREEN = 200           # 10 s green for every road (fixed mode)
RELEASE_GAP = 14            # ticks between two vehicles leaving
GAP_OUT = 30                # end green if the road stays empty for 1.5 s
MIN_SERVE = 30              # a green must run at least 1.5 s before cut short
MAX_QUEUE = 25              # maximum vehicles waiting on one road
SPEED = 6                   # pixels moved per tick
EMERGENCY_CHANCE = 0.02     # 2% of random arrivals are emergency vehicles

# ---------------------------------------------------------------------------
# DRAWING GEOMETRY (left-hand traffic, like India)
# ---------------------------------------------------------------------------
CANVAS = 640
CENTER = CANVAS // 2
HALF = 50                   # half the road width
CAR_W, CAR_L = 22, 36
SLOT = 44                   # distance between queued vehicles
FIRST_SLOT = 24             # centre of the first vehicle from the stop line
DISP_START = CENTER - HALF + 40     # where vehicles enter the screen
DISP_EXIT = -(2 * HALF + CENTER - HALF + 40)

# Stop-line point of each approach lane, and the direction in which the queue
# grows (away from the junction).
STOP = {
    "North": (CENTER + HALF // 2, CENTER - HALF),
    "South": (CENTER - HALF // 2, CENTER + HALF),
    "East": (CENTER + HALF, CENTER + HALF // 2),
    "West": (CENTER - HALF, CENTER - HALF // 2),
}
BACK = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}

LIGHT_POS = {
    "North": (CENTER + 75, CENTER - 75),
    "East": (CENTER + 75, CENTER + 75),
    "South": (CENTER - 75, CENTER + 75),
    "West": (CENTER - 75, CENTER - 75),
}
TEXT_POS = {
    "North": (CENTER - HALF - 60, 18),
    "South": (CENTER + HALF + 60, CANVAS - 18),
    "East": (CANVAS - 60, CENTER - HALF - 18),
    "West": (60, CENTER + HALF + 18),
}

CAR_COLORS = ["#e53935", "#1e88e5", "#fdd835", "#8e24aa",
              "#fb8c00", "#00acc1", "#6d4c41", "#eceff1"]
PRIORITY = {"ambulance": 1, "firetruck": 2}      # smaller number = more urgent


# ---------------------------------------------------------------------------
# DATA STRUCTURE CLASSES
# ---------------------------------------------------------------------------
class Vehicle:
    def __init__(self, serial, kind, arrival_tick, color):
        self.serial = serial
        self.kind = kind                        # car / ambulance / firetruck
        self.priority = PRIORITY.get(kind)      # None for normal cars
        self.arrival = arrival_tick
        self.color = color
        self.disp = DISP_START                  # drawing distance from stop line


class Road:
    """One approach road with a QUEUE for cars and a HEAP for emergencies."""

    def __init__(self, name):
        self.name = name
        self.normal = deque()       # Queue  : normal cars (FIFO)
        self.emergency = []         # Heap   : (priority, serial, vehicle)
        self.last_green = 0

    def size(self):
        return len(self.normal) + len(self.emergency)

    def add(self, vehicle):
        if vehicle.priority is None:
            self.normal.append(vehicle)                                   # enqueue
        else:
            heapq.heappush(self.emergency,
                           (vehicle.priority, vehicle.serial, vehicle))   # heap push

    def peek(self):
        if self.emergency:
            return self.emergency[0][2]
        if self.normal:
            return self.normal[0]
        return None

    def next_vehicle(self):
        """Emergency vehicles always leave first, then normal cars in order."""
        if self.emergency:
            return heapq.heappop(self.emergency)[2]                       # heap pop
        if self.normal:
            return self.normal.popleft()                                  # dequeue
        return None

    def waiting_order(self):
        """Vehicles in the order they will leave (used for drawing)."""
        return [t[2] for t in sorted(self.emergency)] + list(self.normal)


# ---------------------------------------------------------------------------
# SIMULATION (all the logic, no GUI code)
# ---------------------------------------------------------------------------
class Simulation:
    def __init__(self, mode="smart", density=40, seed=None):
        self.mode = mode            # "smart" or "fixed"
        self.density = density      # 5..100
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.tick = 0
        self.serial = 0
        self.roads = {name: Road(name) for name in ROADS}
        self.crossing = []          # (road name, vehicle) currently crossing
        self.history = []           # STACK of signal changes
        self.phase = "ALL_RED"      # GREEN / YELLOW / ALL_RED
        self.phase_left = 10
        self.green_road = None
        self.green_elapsed = 0
        self.idle_ticks = 0
        self.release_cd = 0
        self.rr_index = -1
        self.passed = 0
        self.wait_total = 0
        self.em_passed = 0
        self.em_wait_total = 0
        self.queue_sum = 0

    # ---------------- statistics ----------------
    def avg_wait(self):
        return self.wait_total / TPS / self.passed if self.passed else 0.0

    def avg_em_wait(self):
        return self.em_wait_total / TPS / self.em_passed if self.em_passed else 0.0

    def avg_waiting(self):
        return self.queue_sum / self.tick if self.tick else 0.0

    # ---------------- one tick of time ----------------
    def step(self):
        self.tick += 1
        self._random_arrivals()
        self._update_signal()
        self._release_vehicle()
        self._move_vehicles()
        self.queue_sum += sum(r.size() for r in self.roads.values())

    # ---------------- arrivals ----------------
    def _random_arrivals(self):
        p = self.density / 100 * 0.025
        for name in ROADS:
            # always draw 3 numbers so both modes get identical traffic
            a, b, c = self.rng.random(), self.rng.random(), self.rng.random()
            if a < p:
                if b < EMERGENCY_CHANCE:
                    kind = "ambulance" if c < 0.6 else "firetruck"
                    self.spawn(name, kind)
                else:
                    self.spawn(name, "car", CAR_COLORS[int(c * len(CAR_COLORS))])

    def spawn(self, road_name, kind="car", color="#e53935"):
        road = self.roads[road_name]
        if road.size() >= MAX_QUEUE:
            return False
        self.serial += 1
        v = Vehicle(self.serial, kind, self.tick, color)
        order = road.waiting_order()
        if order and v.priority is None:
            v.disp = max(v.disp, order[-1].disp + SLOT)   # join behind the last car
        road.add(v)
        return True

    # ---------------- signal controller ----------------
    def _emergency_elsewhere(self):
        return any(r.emergency for n, r in self.roads.items() if n != self.green_road)

    def _update_signal(self):
        if self.phase == "GREEN":
            road = self.roads[self.green_road]
            self.green_elapsed += 1
            self.phase_left -= 1
            self.idle_ticks = self.idle_ticks + 1 if road.size() == 0 else 0
            stop = self.phase_left <= 0
            if self.mode == "smart" and self.green_elapsed >= MIN_SERVE:
                if self.idle_ticks >= GAP_OUT:                    # road is empty
                    stop = True
                elif not road.emergency and self._emergency_elsewhere():
                    stop = True                                   # emergency preemption
            if stop:
                road.last_green = self.tick
                self.phase = "YELLOW"
                self.phase_left = YELLOW_TICKS
        elif self.phase == "YELLOW":
            self.phase_left -= 1
            if self.phase_left <= 0:
                self.phase = "ALL_RED"
                self.phase_left = ALL_RED_TICKS
                self.green_road = None
        else:                                                     # ALL_RED
            self.phase_left -= 1
            if self.phase_left <= 0:
                self._start_green()

    def _start_green(self):
        name, duration, reason = self._choose_next()
        self.green_road = name
        self.phase = "GREEN"
        self.phase_left = duration
        self.green_elapsed = 0
        self.idle_ticks = 0
        self.release_cd = 0
        self.history.append((self.tick / TPS, name, duration / TPS, reason))  # STACK push
        if len(self.history) > 60:
            del self.history[0]

    def _choose_next(self):
        """Decide which road gets green and for how long."""
        if self.mode == "fixed":
            self.rr_index = (self.rr_index + 1) % len(ROADS)
            return ROADS[self.rr_index], FIXED_GREEN, "fixed timer"

        if all(r.size() == 0 for r in self.roads.values()):
            self.rr_index = (self.rr_index + 1) % len(ROADS)
            return ROADS[self.rr_index], MIN_GREEN, "no traffic"

        def score(name):
            road = self.roads[name]
            waited = (self.tick - road.last_green) / 40 if road.size() else 0
            return 100 * len(road.emergency) + road.size() + waited

        best = max(ROADS, key=score)                              # SEARCH
        road = self.roads[best]
        duration = max(MIN_GREEN, min(MAX_GREEN, MIN_GREEN + PER_CAR * road.size()))
        if road.emergency:
            reason = "emergency first"
        elif road.size() == max(r.size() for r in self.roads.values()):
            reason = "busiest road"
        else:
            reason = "long wait"
        return best, duration, reason

    def signal_state(self, name):
        if self.green_road == name:
            return "green" if self.phase == "GREEN" else "yellow"
        return "red"

    # ---------------- vehicles ----------------
    def _release_vehicle(self):
        if self.phase != "GREEN":
            return
        if self.release_cd > 0:
            self.release_cd -= 1
            return
        road = self.roads[self.green_road]
        front = road.peek()
        if front is None or front.disp > FIRST_SLOT + SLOT:
            return                                   # nobody close to the stop line
        v = road.next_vehicle()
        wait = self.tick - v.arrival
        self.passed += 1
        self.wait_total += wait
        if v.priority is not None:
            self.em_passed += 1
            self.em_wait_total += wait
        self.crossing.append((self.green_road, v))
        self.release_cd = RELEASE_GAP

    def _move_vehicles(self):
        for road in self.roads.values():
            for i, v in enumerate(road.waiting_order()):
                target = FIRST_SLOT + i * SLOT
                if v.disp > target:
                    v.disp = max(target, v.disp - SPEED)
                elif v.disp < target:
                    v.disp = min(target, v.disp + SPEED)
        for _, v in self.crossing:
            v.disp -= SPEED
        self.crossing = [(n, v) for n, v in self.crossing if v.disp > DISP_EXIT]


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
LIGHT_ON = {"red": "#ff3b30", "yellow": "#ffcc00", "green": "#34c759"}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Smart Traffic Signal Simulator - DSA Project")
        root.resizable(False, False)
        self.sim = Simulation()
        self.paused = False
        self._build_ui()
        self._draw_static()
        self._loop()

    # ---------- layout ----------
    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=CANVAS, height=CANVAS,
                                highlightthickness=0, bg="#3a7d44")
        self.canvas.grid(row=0, column=0)

        panel = tk.Frame(self.root, padx=12, pady=8)
        panel.grid(row=0, column=1, sticky="n")

        tk.Label(panel, text="Smart Traffic Signal Simulator",
                 font=("Helvetica", 14, "bold")).pack(pady=(0, 6))

        mode_box = tk.LabelFrame(panel, text="Signal mode", padx=6, pady=2)
        mode_box.pack(fill="x")
        self.mode_var = tk.StringVar(value="smart")
        tk.Radiobutton(mode_box, text="Smart (adaptive)", variable=self.mode_var,
                       value="smart", command=self.on_mode).pack(anchor="w")
        tk.Radiobutton(mode_box, text="Fixed timer (old style)", variable=self.mode_var,
                       value="fixed", command=self.on_mode).pack(anchor="w")

        self.density = tk.Scale(panel, from_=5, to=100, orient="horizontal",
                                label="Traffic density", length=250,
                                command=self.on_density)
        self.density.set(40)
        self.density.pack()

        self.speed = tk.Scale(panel, from_=1, to=5, orient="horizontal",
                              label="Simulation speed", length=250)
        self.speed.set(1)
        self.speed.pack()

        row = tk.Frame(panel)
        row.pack(pady=4)
        self.pause_btn = tk.Button(row, text="Pause", width=10, command=self.on_pause)
        self.pause_btn.pack(side="left", padx=3)
        tk.Button(row, text="Reset", width=10, command=self.sim.reset).pack(side="left", padx=3)

        em = tk.LabelFrame(panel, text="Send emergency vehicle", padx=4, pady=4)
        em.pack(fill="x")
        for i, name in enumerate(ROADS):
            tk.Button(em, text=f"Ambulance {name}", width=14,
                      command=lambda n=name: self.sim.spawn(n, "ambulance")
                      ).grid(row=i // 2, column=i % 2, padx=2, pady=1)
        tk.Button(em, text="Fire truck (random road)",
                  command=lambda: self.sim.spawn(random.choice(ROADS), "firetruck")
                  ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        self.status = tk.Label(panel, text="", font=("Helvetica", 12, "bold"))
        self.status.pack(pady=(8, 2))

        self.stats = tk.Label(panel, text="", font=("Courier", 11), justify="left")
        self.stats.pack(anchor="w")

        tk.Label(panel, text="SIGNAL LOG (stack, newest on top)",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.log = tk.Label(panel, text="", font=("Courier", 10), justify="left")
        self.log.pack(anchor="w")

    # ---------- control callbacks ----------
    def on_mode(self):
        self.sim.mode = self.mode_var.get()
        self.sim.reset()

    def on_density(self, value):
        self.sim.density = int(float(value))

    def on_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    # ---------- drawing ----------
    def _draw_static(self):
        c = self.canvas
        c.create_rectangle(CENTER - HALF, 0, CENTER + HALF, CANVAS, fill="#3c3c3c", outline="")
        c.create_rectangle(0, CENTER - HALF, CANVAS, CENTER + HALF, fill="#3c3c3c", outline="")
        # dashed centre lines
        c.create_line(CENTER, 0, CENTER, CENTER - HALF, fill="#f5c542", dash=(10, 8), width=2)
        c.create_line(CENTER, CENTER + HALF, CENTER, CANVAS, fill="#f5c542", dash=(10, 8), width=2)
        c.create_line(0, CENTER, CENTER - HALF, CENTER, fill="#f5c542", dash=(10, 8), width=2)
        c.create_line(CENTER + HALF, CENTER, CANVAS, CENTER, fill="#f5c542", dash=(10, 8), width=2)
        # stop lines
        c.create_line(CENTER, CENTER - HALF, CENTER + HALF, CENTER - HALF, fill="white", width=4)
        c.create_line(CENTER - HALF, CENTER + HALF, CENTER, CENTER + HALF, fill="white", width=4)
        c.create_line(CENTER + HALF, CENTER, CENTER + HALF, CENTER + HALF, fill="white", width=4)
        c.create_line(CENTER - HALF, CENTER - HALF, CENTER - HALF, CENTER, fill="white", width=4)
        # signal poles
        for name, (x, y) in LIGHT_POS.items():
            c.create_rectangle(x - 11, y - 28, x + 11, y + 28, fill="#111", outline="#555")
            side = 24 if x > CENTER else -24
            c.create_text(x + side, y, text=name[0], fill="white", font=("Helvetica", 11, "bold"))

    def _draw_vehicle(self, road, v):
        c = self.canvas
        sx, sy = STOP[road]
        bx, by = BACK[road]
        cx, cy = sx + bx * v.disp, sy + by * v.disp
        vertical = bx == 0
        w, h = (CAR_W, CAR_L) if vertical else (CAR_L, CAR_W)

        body = v.color
        if v.kind == "ambulance":
            body = "#ffffff"
        elif v.kind == "firetruck":
            body = "#d32f2f"
        c.create_rectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                           fill=body, outline="#111", tags="dyn")

        # windshield at the front
        fx, fy = -bx, -by
        wx, wy = cx + fx * 8, cy + fy * 8
        sw, sh = (CAR_W - 6, 7) if vertical else (7, CAR_W - 6)
        c.create_rectangle(wx - sw / 2, wy - sh / 2, wx + sw / 2, wy + sh / 2,
                           fill="#263238", outline="", tags="dyn")

        if v.priority is not None:                    # flashing siren
            flash = "#ff1744" if (self.sim.tick // 5) % 2 == 0 else "#2979ff"
            c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=flash, outline="", tags="dyn")

    def draw(self):
        c = self.canvas
        sim = self.sim
        c.delete("dyn")

        for name, (x, y) in LIGHT_POS.items():
            state = sim.signal_state(name)
            for k, colour in enumerate(("red", "yellow", "green")):
                fill = LIGHT_ON[colour] if state == colour else "#2a2a2a"
                cy = y - 17 + k * 17
                c.create_oval(x - 7, cy - 7, x + 7, cy + 7, fill=fill, outline="", tags="dyn")

        for name, road in sim.roads.items():
            for v in road.waiting_order():
                self._draw_vehicle(name, v)
            x, y = TEXT_POS[name]
            c.create_text(x, y, text=f"{name.upper()} ({road.size()})", fill="white",
                          font=("Helvetica", 11, "bold"), tags="dyn")
        for name, v in sim.crossing:
            self._draw_vehicle(name, v)

    def update_panel(self):
        sim = self.sim
        if sim.phase == "GREEN":
            text = f"GREEN: {sim.green_road.upper()}  ({sim.phase_left / TPS:.1f}s)"
        elif sim.phase == "YELLOW":
            text = f"YELLOW: {sim.green_road.upper()}"
        else:
            text = "ALL RED"
        self.status.config(text=text)

        waiting = " ".join(f"{n[0]}:{sim.roads[n].size()}" for n in ROADS)
        self.stats.config(text=(
            f"Time            : {sim.tick / TPS:6.1f} s\n"
            f"Vehicles passed : {sim.passed}\n"
            f"Avg wait        : {sim.avg_wait():5.1f} s\n"
            f"Emergency passed: {sim.em_passed}\n"
            f"Emergency wait  : {sim.avg_em_wait():5.1f} s\n"
            f"Avg waiting     : {sim.avg_waiting():5.1f} cars\n"
            f"Waiting now     : {waiting}"))

        lines = []
        for t, road, dur, reason in reversed(sim.history[-6:]):   # top of stack first
            lines.append(f"{t:6.1f}s {road:<5} {dur:4.1f}s {reason}")
        self.log.config(text="\n".join(lines))

    def _loop(self):
        if not self.paused:
            for _ in range(int(self.speed.get())):
                self.sim.step()
        self.draw()
        self.update_panel()
        self.root.after(TICK_MS, self._loop)


# ---------------------------------------------------------------------------
# TEXT COMPARISON (no window): python traffic_simulator.py --compare
# ---------------------------------------------------------------------------
def compare(minutes=10, density=40, seed=7):
    print(f"Same traffic for both modes: {minutes} minutes, density {density}\n")
    print(f"{'Mode':<7}{'Passed':>8}{'Avg wait':>11}{'Emerg. wait':>13}{'Avg waiting':>13}")
    for mode in ("fixed", "smart"):
        sim = Simulation(mode, density, seed)
        for _ in range(minutes * 60 * TPS):
            sim.step()
        print(f"{mode:<7}{sim.passed:>8}{sim.avg_wait():>10.1f}s"
              f"{sim.avg_em_wait():>12.1f}s{sim.avg_waiting():>13.1f}")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    elif tk is None:
        print("Tkinter is not installed. Install Python from python.org "
              "(or: sudo apt install python3-tk on Linux) and run again.")
    else:
        window = tk.Tk()
        App(window)
        window.mainloop()