import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ALL_DIRS_ORDERED = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const EAST_DIRS = ["NE", "E", "SE"];

function loadImage(url) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = url;
    });
}

function urlFor(entry) {
    const params = new URLSearchParams({
        filename: entry.filename,
        subfolder: entry.subfolder || "",
        type: entry.type || "temp",
    });
    return api.apiURL(`/view?${params}`);
}

function createInspector(node) {
    const wrap = document.createElement("div");
    wrap.style.cssText = `
        display:flex; flex-direction:column; gap:8px; padding:8px;
        background:#111; color:#fff; font-family: system-ui, sans-serif; font-size:12px;
        border-radius:6px; box-sizing:border-box; width:100%;
    `;

    const canvasRow = document.createElement("div");
    canvasRow.style.cssText = "display:flex; align-items:center; justify-content:center; gap:8px;";

    const btnLeft = document.createElement("button");
    btnLeft.textContent = "◄";
    const btnRight = document.createElement("button");
    btnRight.textContent = "►";
    for (const b of [btnLeft, btnRight]) {
        b.style.cssText = `
            background:#1a1a1a; color:#fff; border:1px solid #333; border-radius:50%;
            width:36px; height:36px; cursor:pointer; flex-shrink:0;
        `;
    }

    const canvasWrapper = document.createElement("div");
    canvasWrapper.style.cssText = `
        position:relative; background:#222; border:1px solid #333; border-radius:4px;
        overflow:hidden; display:flex; align-items:center; justify-content:center;
        background-image:
            linear-gradient(45deg,#333 25%,transparent 25%),
            linear-gradient(-45deg,#333 25%,transparent 25%),
            linear-gradient(45deg,transparent 75%,#333 75%),
            linear-gradient(-45deg,transparent 75%,#333 75%);
        background-size:16px 16px; background-position:0 0,0 8px,8px -8px,-8px 0;
    `;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "image-rendering:pixelated; display:block;";
    canvasWrapper.appendChild(canvas);

    const wasdOverlay = document.createElement("div");
    wasdOverlay.style.cssText = `
        position:absolute; bottom:8px; right:8px; display:none;
        flex-direction:column; align-items:center; gap:3px; opacity:0.85; pointer-events:none;
    `;
    const makeKey = (label) => {
        const k = document.createElement("div");
        k.textContent = label;
        k.style.cssText = `
            width:26px; height:26px; background:rgba(26,26,26,0.9); border:2px solid #555;
            border-radius:4px; display:flex; align-items:center; justify-content:center;
            font-weight:bold; font-family:monospace; font-size:12px;
        `;
        return k;
    };
    const uiKeys = { w: makeKey("W"), a: makeKey("A"), s: makeKey("S"), d: makeKey("D") };
    const wasdRow1 = document.createElement("div"); wasdRow1.style.display = "flex"; wasdRow1.appendChild(uiKeys.w);
    const wasdRow2 = document.createElement("div"); wasdRow2.style.cssText = "display:flex; gap:3px;";
    wasdRow2.append(uiKeys.a, uiKeys.s, uiKeys.d);
    wasdOverlay.append(wasdRow1, wasdRow2);
    canvasWrapper.appendChild(wasdOverlay);

    canvasRow.append(btnLeft, canvasWrapper, btnRight);
    wrap.appendChild(canvasRow);

    const stats = document.createElement("div");
    stats.style.cssText = "text-align:center; font-family:monospace; color:#4ade80; font-size:11px;";
    stats.textContent = "Dir: -- | Frame: -/-";
    wrap.appendChild(stats);

    const dirControl = document.createElement("div");
    dirControl.style.cssText = "display:flex; flex-direction:column; gap:4px;";
    const dirHeader = document.createElement("div");
    dirHeader.style.cssText = "display:flex; justify-content:space-between; color:#888; font-size:11px;";
    const dirLabelEl = document.createElement("span");
    dirLabelEl.style.color = "#fff";
    dirLabelEl.textContent = "--";
    dirHeader.innerHTML = "<span>DIRECTION</span>";
    dirHeader.appendChild(dirLabelEl);
    const dirSlider = document.createElement("input");
    dirSlider.type = "range"; dirSlider.min = 0; dirSlider.max = 0; dirSlider.value = 0;
    dirSlider.style.width = "100%";
    dirControl.append(dirHeader, dirSlider);
    wrap.appendChild(dirControl);

    const speedControl = document.createElement("div");
    speedControl.style.cssText = "display:flex; flex-direction:column; gap:4px;";
    const speedHeader = document.createElement("div");
    speedHeader.style.cssText = "display:flex; justify-content:space-between; color:#888; font-size:11px;";
    const speedLabel = document.createElement("span"); speedLabel.style.color = "#fff"; speedLabel.textContent = "12 FPS";
    speedHeader.innerHTML = "<span>SPEED</span>";
    speedHeader.appendChild(speedLabel);
    const speedSlider = document.createElement("input");
    speedSlider.type = "range"; speedSlider.min = 1; speedSlider.max = 60; speedSlider.value = 12;
    speedSlider.style.width = "100%";
    speedControl.append(speedHeader, speedSlider);
    wrap.appendChild(speedControl);

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex; gap:6px;";
    const btnInspector = document.createElement("button");
    btnInspector.textContent = "Inspector";
    const btnMinigame = document.createElement("button");
    btnMinigame.textContent = "Minigame (WASD)";
    for (const b of [btnInspector, btnMinigame]) {
        b.style.cssText = `
            flex:1; padding:8px; background:#1a1a1a; color:#fff; border:1px solid #333;
            border-radius:4px; cursor:pointer; font-weight:bold; font-size:12px;
        `;
    }
    actions.append(btnInspector, btnMinigame);
    wrap.appendChild(actions);

    // ---- State ----
    const state = {
        mode: "idle",
        framesData: {},
        availableDirs: [],
        currentDirIndex: 0,
        fps: 12,
        isMirroring: true,
        animationFrameId: null,
        lastTime: 0,
        autoRotateDir: null,
        autoRotateInterval: null,
        keys: { w: false, a: false, s: false, d: false },
        player: { x: 200, y: 150, speed: 180, currentDir: "S", isMoving: false, animTimer: 0 },
        availableWidth: 256,
        availableHeight: 256,
        lastDims: null,
    };
    ALL_DIRS_ORDERED.forEach(d => state.framesData[d] = []);

    const ctx = canvas.getContext("2d");

    async function setFrames(framesByDir, fps) {
        state.fps = fps;
        speedSlider.value = fps;
        speedLabel.textContent = `${fps} FPS`;
        // Auto-mirror: if no east frames provided, mirror from west
        const hasEast = EAST_DIRS.some(d => (framesByDir[d] || []).length > 0);
        state.isMirroring = !hasEast;

        for (const d of ALL_DIRS_ORDERED) {
            const entries = framesByDir[d] || [];
            const imgs = await Promise.all(entries.map(e => loadImage(urlFor(e))));
            state.framesData[d] = imgs.filter(x => x);
        }
        // Auto-start inspector
        startMode("inspector");
    }

    function prepareData() {
        let maxW = 0, maxH = 0;
        for (const d in state.framesData) {
            for (const img of state.framesData[d]) {
                if (img.width > maxW) maxW = img.width;
                if (img.height > maxH) maxH = img.height;
            }
        }
        if (!maxW || !maxH) return null;
        state.availableDirs = [];
        ALL_DIRS_ORDERED.forEach(dir => {
            let src = dir;
            if (state.isMirroring) {
                if (dir === "NE") src = "NW";
                if (dir === "E") src = "W";
                if (dir === "SE") src = "SW";
            }
            if ((state.framesData[src] || []).length > 0) state.availableDirs.push(dir);
        });
        return state.availableDirs.length ? { maxW, maxH } : null;
    }

    function startMode(mode) {
        const dims = prepareData();
        if (!dims) return;
        stopAutoRotate();
        if (state.animationFrameId) cancelAnimationFrame(state.animationFrameId);
        state.mode = mode;
        state.lastTime = performance.now();

        state.lastDims = dims;
        if (mode === "inspector") {
            wasdOverlay.style.display = "none";
            btnLeft.style.display = btnRight.style.display = "flex";
            dirControl.style.display = speedControl.style.display = "flex";
            canvas.width = dims.maxW;
            canvas.height = dims.maxH;
            applyLayout();
            dirSlider.max = state.availableDirs.length - 1;
            dirSlider.value = 0;
            dirSlider.disabled = false;
            state.currentDirIndex = 0;
            updateDirLabel();
        } else {
            wasdOverlay.style.display = "flex";
            btnLeft.style.display = btnRight.style.display = "none";
            applyLayout();
            state.player.x = canvas.width / 2;
            state.player.y = canvas.height / 2;
            state.player.currentDir = state.availableDirs[0];
            state.player.isMoving = false;
            state.player.animTimer = 0;
            state.keys = { w: false, a: false, s: false, d: false };
            Object.values(uiKeys).forEach(k => { k.style.background = "rgba(26,26,26,0.9)"; });
        }
        ctx.imageSmoothingEnabled = false;
        state.animationFrameId = requestAnimationFrame(renderLoop);
    }

    function applyLayout() {
        // Reserve ~180px vertical space for controls below the canvas.
        const maxW = Math.max(120, state.availableWidth - 80);  // arrows take ~80px total
        const maxH = Math.max(120, state.availableHeight - 180);

        let aspect;
        if (state.mode === "minigame") {
            aspect = 4 / 3;
        } else if (state.lastDims) {
            aspect = state.lastDims.maxW / state.lastDims.maxH;
        } else {
            aspect = 1;  // default square before frames arrive
        }

        let w = maxW, h = maxW / aspect;
        if (h > maxH) { h = maxH; w = maxH * aspect; }
        w = Math.floor(w); h = Math.floor(h);

        if (state.mode === "minigame") {
            canvas.width = w;
            canvas.height = h;
        }
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
        canvasWrapper.style.width = w + "px";
        canvasWrapper.style.height = h + "px";
    }

    function updateDirLabel() {
        if (state.availableDirs.length > 0) {
            dirLabelEl.textContent = state.availableDirs[state.currentDirIndex];
        }
    }

    function stopAutoRotate() {
        state.autoRotateDir = null;
        if (state.autoRotateInterval) { clearInterval(state.autoRotateInterval); state.autoRotateInterval = null; }
        btnLeft.style.background = btnRight.style.background = "#1a1a1a";
    }

    function toggleAutoRotate(dir) {
        if (state.autoRotateDir === dir) { stopAutoRotate(); return; }
        stopAutoRotate();
        if (state.availableDirs.length <= 1) return;
        state.autoRotateDir = dir;
        (dir === "left" ? btnLeft : btnRight).style.background = "#4ade80";
        state.autoRotateInterval = setInterval(() => {
            const n = state.availableDirs.length;
            state.currentDirIndex = dir === "left"
                ? (state.currentDirIndex - 1 + n) % n
                : (state.currentDirIndex + 1) % n;
            dirSlider.value = state.currentDirIndex;
            updateDirLabel();
        }, 1500);
    }

    function getBestDirection(desired) {
        if (state.availableDirs.includes(desired)) return desired;
        const fb = {
            NW: ["N", "W", "S", "E"], NE: ["N", "E", "S", "W"],
            SW: ["S", "W", "N", "E"], SE: ["S", "E", "N", "W"],
            N: ["NW", "NE", "W", "E", "S"], S: ["SW", "SE", "W", "E", "N"],
            W: ["NW", "SW", "N", "S", "E"], E: ["NE", "SE", "N", "S", "W"],
        };
        for (const f of (fb[desired] || [])) if (state.availableDirs.includes(f)) return f;
        return state.availableDirs[0];
    }

    function updateMinigame(dt) {
        let dx = 0, dy = 0;
        if (state.keys.w) dy -= 1;
        if (state.keys.s) dy += 1;
        if (state.keys.a) dx -= 1;
        if (state.keys.d) dx += 1;
        state.player.isMoving = dx !== 0 || dy !== 0;
        if (state.player.isMoving) {
            const len = Math.hypot(dx, dy); dx /= len; dy /= len;
            state.player.x += dx * state.player.speed * dt;
            state.player.y += dy * state.player.speed * dt;
            state.player.x = Math.max(0, Math.min(canvas.width, state.player.x));
            state.player.y = Math.max(0, Math.min(canvas.height, state.player.y));
            let desired = "";
            if (dy < 0) desired += "N"; if (dy > 0) desired += "S";
            if (dx < 0) desired += "W"; if (dx > 0) desired += "E";
            if (desired) state.player.currentDir = getBestDirection(desired);
            state.player.animTimer += dt * 1000;
        } else {
            state.player.animTimer = 0;
        }
    }

    function renderLoop(time) {
        if (state.mode === "idle") return;
        const dt = (time - state.lastTime) / 1000;
        state.lastTime = time;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        let currentDir, frameIdx = 0, frames, srcDir, flip = false;

        if (state.mode === "inspector") {
            currentDir = state.availableDirs[state.currentDirIndex];
            srcDir = currentDir;
            if (state.isMirroring) {
                if (currentDir === "NE") { srcDir = "NW"; flip = true; }
                if (currentDir === "E") { srcDir = "W"; flip = true; }
                if (currentDir === "SE") { srcDir = "SW"; flip = true; }
            }
            frames = state.framesData[srcDir] || [];
            if (frames.length) {
                const frameDuration = 1000 / state.fps;
                frameIdx = Math.floor(time / frameDuration) % frames.length;
                const img = frames[frameIdx];
                if (img) {
                    ctx.save();
                    ctx.translate(canvas.width / 2, canvas.height);
                    if (flip) ctx.scale(-1, 1);
                    ctx.drawImage(img, -img.width / 2, -img.height);
                    ctx.restore();
                }
            }
            stats.textContent = `Dir: ${currentDir} | Frame: ${frameIdx + 1}/${frames.length}`;
        } else {
            updateMinigame(dt);
            currentDir = state.player.currentDir;
            srcDir = currentDir;
            if (state.isMirroring) {
                if (currentDir === "NE") { srcDir = "NW"; flip = true; }
                if (currentDir === "E") { srcDir = "W"; flip = true; }
                if (currentDir === "SE") { srcDir = "SW"; flip = true; }
            }
            frames = state.framesData[srcDir] || [];
            if (frames.length) {
                const frameDuration = 1000 / state.fps;
                frameIdx = Math.floor(state.player.animTimer / frameDuration) % frames.length;
                const img = frames[frameIdx];
                if (img) {
                    ctx.save();
                    ctx.translate(state.player.x, state.player.y);
                    if (flip) ctx.scale(-1, 1);
                    ctx.drawImage(img, -img.width / 2, -img.height);
                    ctx.restore();
                }
            }
            stats.textContent = `WASD | Dir: ${currentDir} | Frame: ${frameIdx + 1}/${frames.length}`;
        }
        state.animationFrameId = requestAnimationFrame(renderLoop);
    }

    // ---- Events ----
    // Stop pointer events from bubbling to ComfyUI's canvas (which intercepts drags).
    const stopBubble = (el) => {
        for (const evt of ["pointerdown", "pointerup", "pointermove", "mousedown", "mouseup", "click", "wheel"]) {
            el.addEventListener(evt, (e) => e.stopPropagation());
        }
    };
    [btnLeft, btnRight, btnInspector, btnMinigame, dirSlider, speedSlider, canvas, canvasWrapper].forEach(stopBubble);

    btnLeft.addEventListener("click", (e) => { e.stopPropagation(); toggleAutoRotate("left"); });
    btnRight.addEventListener("click", (e) => { e.stopPropagation(); toggleAutoRotate("right"); });
    btnInspector.addEventListener("click", (e) => { e.stopPropagation(); startMode("inspector"); });
    btnMinigame.addEventListener("click", (e) => { e.stopPropagation(); startMode("minigame"); });
    dirSlider.addEventListener("input", (e) => {
        stopAutoRotate();
        state.currentDirIndex = parseInt(e.target.value);
        updateDirLabel();
    });
    speedSlider.addEventListener("input", (e) => {
        state.fps = parseInt(e.target.value);
        speedLabel.textContent = `${state.fps} FPS`;
    });

    // Resize observer — re-layout canvas when the node gets bigger/smaller
    const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
            const rect = entry.contentRect;
            if (rect.width > 0) state.availableWidth = rect.width;
            if (rect.height > 0) state.availableHeight = rect.height;
            applyLayout();
        }
    });
    ro.observe(wrap);
    // Initial layout
    requestAnimationFrame(() => {
        const rect = wrap.getBoundingClientRect();
        if (rect.width > 0) state.availableWidth = rect.width;
        if (rect.height > 0) state.availableHeight = rect.height;
        applyLayout();
    });

    // Keyboard for minigame — only while node is focused
    const keyHandler = (down) => (e) => {
        if (state.mode !== "minigame") return;
        if (!wrap.matches(":hover") && !document.activeElement?.isSameNode?.(wrap)) return;
        const k = e.key.toLowerCase();
        if (state.keys.hasOwnProperty(k)) {
            state.keys[k] = down;
            uiKeys[k].style.background = down ? "#4ade80" : "rgba(26,26,26,0.9)";
            uiKeys[k].style.color = down ? "#000" : "#fff";
            e.preventDefault();
        }
    };
    window.addEventListener("keydown", keyHandler(true));
    window.addEventListener("keyup", keyHandler(false));

    return { root: wrap, setFrames };
}

app.registerExtension({
    name: "loopstrip.sprite_inspector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoopStripSpriteInspector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const inspector = createInspector(this);
            this._spriteInspector = inspector;
            this.addDOMWidget("sprite_inspector", "preview", inspector.root, {
                serialize: false,
                hideOnZoom: false,
            });
            this.size = [420, 560];
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!this._spriteInspector) return;
            const frames = message?.sprite_frames?.[0];
            const fps = message?.fps?.[0] ?? 12;
            if (frames) this._spriteInspector.setFrames(frames, fps);
        };
    },
});
