import heapq
import tkinter as tk
from tkinter import ttk, messagebox


def _astar(n, start, goal, weights):
    
    def h(p):
        dx = abs(p[0] - goal[0])
        dy = abs(p[1] - goal[1])
        return max(dx, dy)

    open_heap = [(h(start) + weights.get(start, 0), 0, 0, start)]
    g = {start: (0,0)}
    parent = {}
    closed = set()

    while open_heap:
        _, dcur, gcur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:                              
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)

        for dx in (-1, 0, 1):                        
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n) or nb in closed:
                    continue
                
                is_diag = 1 if (dx != 0 and dy != 0) else 0
                step = max(abs(dx), abs(dy))         
                new_g = gcur + step + weights.get(nb, 0)
                new_d = dcur + is_diag
                if (new_g, new_d) < g.get(nb, (float("inf"), float("inf"))):
                    g[nb] = (new_g, new_d)
                    parent[nb] = cur
                    f_steps = new_g + h(nb)
                    heapq.heappush(open_heap, (f_steps, new_d, new_g, nb))
    return None


def mstar(n, start, goal, broken=(), edge_weight=1.0, edge_margin=1, penalty=100.0, max_steps=2000):

    broken = set(map(tuple, broken))
    start, goal = tuple(start), tuple(goal)

    # boundary weights: discourage planning along the chip edge
    weights = {}
    for x in range(n):
        for y in range(n):
            if min(x, y, n - 1 - x, n - 1 - y) < edge_margin:
                weights[(x, y)] = edge_weight
    weights[start] = weights[goal] = 0

    path = _astar(n, start, goal, weights)
    result = {"initial_path": path or [], "executed": [start], "stuck_at": [],
              "blocked": [], "replans": 0, "success": False}
    if path is None:
        return result

    pos = start
    steps = 0
    while pos != goal and steps < max_steps:
        steps += 1
        nxt = path[path.index(pos) + 1]

        sensed = pos if nxt in broken else nxt

        if sensed == pos:
            result["stuck_at"].append(pos)
            result["blocked"].append(nxt)
            weights[nxt] = weights.get(nxt, 0) + penalty
            result["replans"] += 1
            path = _astar(n, pos, goal, weights)
            if path is None:
                return result
        else:
            pos = sensed
            result["executed"].append(pos)
    result["success"] = (pos == goal)
    return result


COL_EMPTY, COL_GRID = "#f4f7fb", "#b8c4d6"
COL_START, COL_END, COL_BROKEN = "#2e7de0", "#8e44ad", "#d9362b"
COL_YELLOW, COL_GREEN = "#ffe45c", "#4cc96a"


class App:
    def __init__(self, root):
        self.root = root
        root.title("M* droplet router")
        self.n = 10
        self.start = self.end = None
        self.broken = set()
        self.yellow, self.green = set(), set()
        self.job = None

        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")
        ttk.Label(bar, text="Grid size (n):").pack(side="left")
        self.size_var = tk.IntVar(value=self.n)
        ttk.Spinbox(bar, from_=3, to=40, width=4, textvariable=self.size_var).pack(side="left", padx=4)
        ttk.Button(bar, text="Build grid", command=self.build).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear paths", command=self.clear_paths).pack(side="left")

        self.mode = tk.StringVar(value="start")
        modes = ttk.Frame(root, padding=(8, 0))
        modes.pack(fill="x")
        for text, val in (("Start", "start"), ("End", "end"), ("Broken electrode", "broken")):
            ttk.Radiobutton(modes, text=text, value=val, variable=self.mode).pack(side="left", padx=6)

        self.canvas = tk.Canvas(root, width=600, height=600, bg="white", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = ttk.Label(root, text="", padding=(8, 0, 8, 8))
        self.status.pack(fill="x")
        legend = ttk.Label(root, padding=(8, 0, 8, 8),
                           text="Blue = start | Purple = end | Red = broken | "
                                "Yellow = approximate path | Green = recalculated path")
        legend.pack(fill="x")
        self.build()

    # ---- grid -------------------------------------------------------------
    def build(self):
        try:
            n = int(self.size_var.get())
        except (tk.TclError, ValueError):
            n = 10
        self.n = max(3, min(40, n))
        self.start = self.end = None
        self.broken.clear()
        self.clear_paths(redraw=False)
        self.cell = max(10, 600 // self.n)
        side = self.cell * self.n
        self.canvas.config(width=side, height=side)
        self.status.config(text="Pick Start, then End, then add broken electrodes.")
        self.draw()

    def on_click(self, e):
        x, y = e.x // self.cell, e.y // self.cell
        if not (0 <= x < self.n and 0 <= y < self.n):
            return
        c, m = (x, y), self.mode.get()
        if m == "start":
            if c in self.broken or c == self.end:
                return
            self.start = c
        elif m == "end":
            if c in self.broken or c == self.start:
                return
            self.end = c
        else:
            if c in (self.start, self.end):
                return
            self.broken ^= {c}
        self.run()

    # ---- run M* + animate -------------------------------------------------
    def clear_paths(self, redraw=True):
        if self.job:
            self.root.after_cancel(self.job)
            self.job = None
        self.yellow, self.green = set(), set()
        if redraw:
            self.draw()

    def run(self):
        self.clear_paths(redraw=False)
        if not (self.start and self.end):
            self.draw()
            return
        res = mstar(self.n, self.start, self.end, self.broken)
        if not res["initial_path"]:
            self.draw()
            messagebox.showinfo("M*", "No path found.")
            return

        plan = res["initial_path"]
        # yellow: planned route, stopping at the first problem electrode
        yellow = []
        for c in plan:
            if c in self.broken:
                break
            yellow.append(c)
        # green: what the droplet really did after the first stall
        if res["stuck_at"]:
            first = res["executed"].index(res["stuck_at"][0])
            green = res["executed"][first:]
        else:
            green = res["executed"]

        self.yellow = set(yellow)
        self.draw()
        if res["stuck_at"]:
            msg = (f"Droplet stalled at {res['stuck_at'][0]} (broken electrode "
                   f"{res['blocked'][0]}). Re-planned {res['replans']}x.")
        else:
            msg = "No broken electrode on the planned path."
        msg += "  Goal reached." if res["success"] else "  Goal NOT reachable."
        self.status.config(text=msg)

        def show_green():
            self.green = set(green)
            self.draw()
        self.job = self.root.after(800, show_green)

    # ---- drawing ----------------------------------------------------------
    def draw(self):
        cv, s = self.canvas, self.cell
        cv.delete("all")
        for x in range(self.n):
            for y in range(self.n):
                c = (x, y)
                col = COL_EMPTY
                if c in self.green:
                    col = COL_GREEN
                elif c in self.yellow:
                    col = COL_YELLOW
                if c in self.broken:
                    col = COL_BROKEN
                if c == self.start:
                    col = COL_START
                if c == self.end:
                    col = COL_END
                cv.create_rectangle(x * s, y * s, (x + 1) * s, (y + 1) * s,
                                    fill=col, outline=COL_GRID)
                if c == self.start:
                    cv.create_text(x * s + s / 2, y * s + s / 2, text="S", fill="white")
                elif c == self.end:
                    cv.create_text(x * s + s / 2, y * s + s / 2, text="T", fill="white")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()