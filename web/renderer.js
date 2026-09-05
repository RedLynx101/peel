/** Original Canvas illustration. The simulator owns rules; this module only draws states. */
const colors = {
  floor: ["#b7ba92", "#bfc29a"],
  top: "#7b896a",
  left: "#435842",
  right: "#53674c",
  yellow: "#f3cd54",
  ink: "#213b2e",
};
function polygon(ctx, points, fill, stroke) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }
}
function ellipse(ctx, x, y, rx, ry, color) {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}
function line(ctx, points, color, width = 1) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}
function banana(ctx, x, y, s, t = 0) {
  ctx.save();
  ctx.translate(x, y + Math.sin(t) * 1.3);
  ctx.scale(s, s);
  ctx.rotate(-0.2);
  ctx.beginPath();
  ctx.moveTo(-11, -14);
  ctx.bezierCurveTo(-16, 5, -2, 20, 16, -6);
  ctx.bezierCurveTo(15, 11, 1, 21, -10, 10);
  ctx.bezierCurveTo(-18, 2, -17, -8, -13, -15);
  ctx.closePath();
  ctx.fillStyle = colors.yellow;
  ctx.fill();
  ctx.strokeStyle = "#826a27";
  ctx.lineWidth = 1;
  ctx.stroke();
  line(
    ctx,
    [
      [-12, -15],
      [-10, -18],
    ],
    "#443e26",
    2.3,
  );
  line(
    ctx,
    [
      [15, -6],
      [18, -8],
    ],
    "#443e26",
    2,
  );
  ctx.restore();
}
function thief(ctx, x, y, s, dir = 0, hasBanana = false, t = 0) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s, s);
  ellipse(ctx, 0, 1, 15, 5, "#24382535");
  ellipse(ctx, -6, -4, 5, 3, colors.ink);
  ellipse(ctx, 7, -4, 5, 3, colors.ink);
  ctx.fillStyle = "#293e31";
  ctx.beginPath();
  ctx.roundRect(-12, -27, 25, 25, 7);
  ctx.fill();
  ctx.fillStyle = "#f0cc55";
  ctx.beginPath();
  ctx.moveTo(-13, -25);
  ctx.lineTo(12, -25);
  ctx.lineTo(9, -20);
  ctx.lineTo(-8, -18);
  ctx.lineTo(-14, -9);
  ctx.closePath();
  ctx.fill();
  ellipse(ctx, 0, -37, 15, 14, "#e2d9b6");
  ctx.fillStyle = "#203a2d";
  ctx.beginPath();
  ctx.roundRect(-15, -40, 30, 10, 4);
  ctx.fill();
  const look = dir === 0 ? 2 : dir === 2 ? -2 : 0;
  ellipse(ctx, -6 + look, -35, 3.5, 4, "#fff8db");
  ellipse(ctx, 6 + look, -35, 3.5, 4, "#fff8db");
  ellipse(ctx, -5 + look, -34, 1.8, 2, colors.ink);
  ellipse(ctx, 7 + look, -34, 1.8, 2, colors.ink);
  ellipse(ctx, -1, -47, 14, 9, "#2b4331");
  ctx.fillStyle = "#1e3629";
  ctx.beginPath();
  ctx.roundRect(-18, -46, 36, 5, 2);
  ctx.fill();
  if (hasBanana) banana(ctx, 17, -16, 0.55, t);
  ctx.restore();
}
export class MuseumRenderer {
  constructor(canvas, { decorations = true } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.decorations = decorations;
    this.state = null;
    this.previous = null;
    this.changedAt = 0;
    this.agentView = false;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.width = r.width;
    this.height = r.height;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  setState(state, animate = true) {
    this.previous = animate ? this.state : null;
    this.state = state;
    this.changedAt = performance.now();
  }
  draw(time = 0) {
    const c = this.ctx,
      w = this.width,
      h = this.height;
    if (!w || !h) return;
    c.clearRect(0, 0, w, h);
    if (!this.state) return;
    const state = this.state,
      grid = state.grid,
      n = grid.length;
    const tw = Math.min(w / (n + 0.7), h / (n * 0.47 + 2.3), 76),
      th = tw * 0.46;
    const ox = w * 0.52,
      oy = (h - n * th) / 2 + 15;
    const point = (x, y, z = 0) => [
      ox + ((x - y) * tw) / 2,
      oy + ((x + y) * th) / 2 - z,
    ];
    // Subtle architectural registration marks and the plinth below the entire museum.
    if (this.decorations) {
      c.strokeStyle = "#93a88812";
      c.lineWidth = 0.5;
      for (let k = -12; k < 15; k++) {
        line(
          c,
          [
            [w / 2 + k * 55, 0],
            [w / 2 + k * 55 - h * 2, h],
          ],
          "#9eaf8b08",
        );
        line(
          c,
          [
            [w / 2 + k * 55, 0],
            [w / 2 + k * 55 + h * 2, h],
          ],
          "#9eaf8b08",
        );
      }
    }
    const a = point(-0.55, -0.55, -7),
      b = point(n - 0.45, -0.55, -7),
      d = point(-0.55, n - 0.45, -7),
      e = point(n - 0.45, n - 0.45, -7);
    polygon(
      c,
      [
        [a[0], a[1] + 10],
        [b[0], b[1] + 10],
        [e[0], e[1] + 16],
        [d[0], d[1] + 10],
      ],
      "#0d291f50",
    );
    polygon(c, [a, b, e, d], "#6c7a56");
    polygon(c, [d, e, [e[0], e[1] + 9], [d[0], d[1] + 9]], "#354f36");
    polygon(c, [b, e, [e[0], e[1] + 9], [b[0], b[1] + 9]], "#425d3f");
    const visible = new Set(
      (state.visible || []).map((p) => `${p[0]},${p[1]}`),
    );
    const beam = new Set();
    if (
      state.camera &&
      (!this.agentView || visible.has(`${state.camera.x},${state.camera.y}`))
    ) {
      const { x, y, dir } = state.camera;
      const [dx, dy] = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ][dir % 4];
      for (
        let bx = x + dx, by = y + dy;
        bx >= 0 && by >= 0 && bx < n && by < n;
        bx += dx, by += dy
      ) {
        if ("#D".includes(grid[by][bx])) break;
        beam.add(`${bx},${by}`);
      }
    }
    let ax = state.agent.x,
      ay = state.agent.y;
    const mix = this.reduced
      ? 1
      : Math.min(1, (performance.now() - this.changedAt) / 170);
    if (
      this.previous &&
      Math.abs(this.previous.agent.x - ax) +
        Math.abs(this.previous.agent.y - ay) <=
        1
    ) {
      const s = mix * mix * (3 - 2 * mix);
      ax = this.previous.agent.x + (ax - this.previous.agent.x) * s;
      ay = this.previous.agent.y + (ay - this.previous.agent.y) * s;
    }
    for (let depth = 0; depth < 2 * n; depth++)
      for (let x = 0; x < n; x++) {
        const y = depth - x;
        if (y < 0 || y >= n) continue;
        const tile = grid[y][x],
          p = point(x, y);
        const diamond = [
          [p[0], p[1] - th / 2],
          [p[0] + tw / 2, p[1]],
          [p[0], p[1] + th / 2],
          [p[0] - tw / 2, p[1]],
        ];
        const unseen = this.agentView && !visible.has(`${x},${y}`);
        polygon(
          c,
          diamond,
          unseen ? "#344c39" : colors.floor[(x + y) % 2],
          "#233c2830",
        );
        if (unseen) continue;
        if (beam.has(`${x},${y}`)) polygon(c, diamond, "#f5d36355");
        if (tile === "#") {
          const height = x === 0 || y === 0 ? tw * 0.46 : tw * 0.23;
          const top = diamond.map(([px, py]) => [px, py - height]);
          polygon(
            c,
            [top[3], top[2], diamond[2], diamond[3]],
            unseen ? "#293f30" : colors.left,
          );
          polygon(
            c,
            [top[1], top[2], diamond[2], diamond[1]],
            unseen ? "#2d4333" : colors.right,
          );
          polygon(c, top, unseen ? "#425641" : colors.top, "#b8c19535");
          if (x === 0 || y === 0) {
            line(
              c,
              [
                [p[0] - tw * 0.2, p[1] - height + th * 0.08],
                [p[0] + tw * 0.2, p[1] - height + th * 0.08],
              ],
              "#bdc49730",
            );
          }
        }
        if (!unseen) {
          const scale = tw / 65;
          if (tile === "E") {
            polygon(
              c,
              diamond.map(([px, py]) => [
                p[0] + (px - p[0]) * 0.76,
                p[1] + (py - p[1]) * 0.76,
              ]),
              "#657f51",
              "#dbe3be",
            );
            line(
              c,
              [
                [p[0] - 7 * scale, p[1]],
                [p[0] + 7 * scale, p[1]],
              ],
              "#e7edcf",
              2,
            );
            line(
              c,
              [
                [p[0] + 2 * scale, p[1] - 4 * scale],
                [p[0] + 7 * scale, p[1]],
                [p[0] + 2 * scale, p[1] + 4 * scale],
              ],
              "#e7edcf",
              2,
            );
          }
          if (tile === "B") {
            ellipse(c, p[0], p[1] + 2, tw * 0.21, th * 0.19, "#48513925");
            ellipse(c, p[0], p[1] - 1, tw * 0.19, th * 0.17, "#ece0aa");
            banana(
              c,
              p[0],
              p[1] - 11 * scale,
              scale * 0.82,
              this.reduced ? 0 : time * 0.0018,
            );
          }
          if (tile === "K") {
            ellipse(c, p[0], p[1] + 1, tw * 0.15, th * 0.14, "#364b2825");
            c.strokeStyle = "#f5d367";
            c.lineWidth = 3 * scale;
            c.beginPath();
            c.arc(
              p[0] - 4 * scale,
              p[1] - 6 * scale,
              4 * scale,
              0,
              Math.PI * 2,
            );
            c.stroke();
            line(
              c,
              [
                [p[0], p[1] - 4 * scale],
                [p[0] + 10 * scale, p[1] + 1 * scale],
                [p[0] + 10 * scale, p[1] - 3 * scale],
              ],
              "#f5d367",
              3 * scale,
            );
          }
          if (tile === "D" || tile === "d") {
            const dh = tile === "D" ? tw * 0.55 : tw * 0.14;
            polygon(
              c,
              [
                [p[0] - tw * 0.22, p[1] - dh],
                [p[0] + tw * 0.22, p[1] - dh - th * 0.3],
                [p[0] + tw * 0.22, p[1] - th * 0.3],
                [p[0] - tw * 0.22, p[1]],
              ],
              tile === "D" ? "#9e773e" : "#71835b",
              "#d6b26a",
            );
            if (tile === "D")
              ellipse(c, p[0] + tw * 0.09, p[1] - dh * 0.45, 2, 3, "#f1ce58");
          }
          if (tile === "C") {
            const cam = state.camera || { x, y, dir: 0 };
            ellipse(c, p[0], p[1], tw * 0.13, th * 0.16, "#566847");
            line(
              c,
              [
                [p[0], p[1]],
                [p[0], p[1] - tw * 0.45],
              ],
              "#516146",
              4 * scale,
            );
            c.save();
            c.translate(p[0], p[1] - tw * 0.45);
            c.rotate((((cam.dir || 0) * Math.PI) / 2) * 0.2);
            c.fillStyle = "#d1d2af";
            c.fillRect(-9 * scale, -5 * scale, 20 * scale, 10 * scale);
            c.fillStyle = "#e7c864";
            c.fillRect(8 * scale, -3 * scale, 4 * scale, 6 * scale);
            c.restore();
          }
        }
        // Draw the actor in depth order so front walls correctly occlude its feet.
        if (x === Math.round(ax) && y === Math.round(ay)) {
          const q = point(ax, ay);
          thief(
            c,
            q[0],
            q[1],
            tw / 65,
            state.agent.dir,
            state.inventory?.banana,
            time * 0.0018,
          );
        }
      }
  }
  destroy() {
    this.resizeObserver.disconnect();
  }
}
