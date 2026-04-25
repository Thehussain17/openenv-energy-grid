// main.js - Three.js Engine and API Controller for Energy Grid

// State
let envState = {
    obs: null,
    step: 0,
    done: false
};
let autoRunInterval = null;

// Three.js Globals
let scene, camera, renderer;
let clock = new THREE.Clock();
let windTurbines = [];
let solarPanels = [];
let sunLight;

// Colors
const COLOR_STABLE = 0x3b82f6;
const COLOR_CRITICAL = 0xef4444;
const COLOR_BG = 0x0b0f19;

// --- Initialize Scene ---
function init3D() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(COLOR_BG);
    scene.fog = new THREE.Fog(COLOR_BG, 10, 50);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 15, 20);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    sunLight = new THREE.DirectionalLight(0xfffaee, 1.2);
    sunLight.position.set(10, 20, 10);
    scene.add(sunLight);

    buildWorld();

    window.addEventListener('resize', onWindowResize);
    animate();
}

function buildWorld() {
    // Base platform
    const platformGeo = new THREE.CylinderGeometry(15, 12, 2, 64);
    const platformMat = new THREE.MeshStandardMaterial({ 
        color: 0x1e293b, 
        roughness: 0.8,
        flatShading: true
    });
    const platform = new THREE.Mesh(platformGeo, platformMat);
    platform.position.y = -1;
    scene.add(platform);

    // Battery (Center)
    const bessGeo = new THREE.CylinderGeometry(1.5, 1.5, 4, 16);
    const bessMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, metalness: 0.5 });
    const bess = new THREE.Mesh(bessGeo, bessMat);
    bess.name = 'bess';
    bess.position.set(0, 2, 0);
    scene.add(bess);

    // Wind Turbines (Left)
    for(let i=0; i<3; i++) {
        const poleGeo = new THREE.CylinderGeometry(0.2, 0.3, 6);
        const pole = new THREE.Mesh(poleGeo, new THREE.MeshStandardMaterial({color: 0x94a3b8}));
        pole.position.set(-8 + i*3, 3, -5 + i*2);
        
        const rotor = new THREE.Group();
        const hub = new THREE.Mesh(new THREE.SphereGeometry(0.4), new THREE.MeshStandardMaterial({color: 0xffffff}));
        rotor.add(hub);
        
        for(let j=0; j<3; j++) {
            const blade = new THREE.Mesh(new THREE.BoxGeometry(0.1, 4, 0.4), new THREE.MeshStandardMaterial({color: 0xffffff}));
            blade.position.y = 2;
            const pivot = new THREE.Group();
            pivot.rotation.z = (j * Math.PI * 2) / 3;
            pivot.add(blade);
            rotor.add(pivot);
        }
        rotor.position.y = 3;
        rotor.rotation.y = Math.PI / 4;
        pole.add(rotor);
        scene.add(pole);
        windTurbines.push(rotor);
    }

    // Solar Farm (Right)
    const panelGeo = new THREE.BoxGeometry(2, 0.1, 3);
    const panelMat = new THREE.MeshStandardMaterial({ color: 0x0ea5e9, metalness: 0.8, roughness: 0.2 });
    for(let i=0; i<4; i++) {
        const panel = new THREE.Mesh(panelGeo, panelMat);
        panel.position.set(8 + (i%2)*3, 0.5, 3 + Math.floor(i/2)*4);
        panel.rotation.x = -Math.PI / 6;
        scene.add(panel);
        solarPanels.push(panel);
    }

    // Sectors
    createBuilding('Hospital', 0, 1.5, 7, 0xef4444);
    createBuilding('Industrial', 7, 2, -5, 0xf59e0b);
    createBuilding('Residential', -6, 1, 5, 0x10b981, true);
}

function createBuilding(name, x, y, z, color, multiple=false) {
    const group = new THREE.Group();
    group.name = name;
    group.position.set(x, y, z);

    if (multiple) {
        for(let i=0; i<3; i++) {
            const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.5, 2, 1.5), new THREE.MeshStandardMaterial({color}));
            mesh.position.set(i*2 - 2, 0, (i%2));
            group.add(mesh);
        }
    } else {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(3, y*2, 3), new THREE.MeshStandardMaterial({color}));
        group.add(mesh);
    }
    scene.add(group);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();

    // Rotate wind turbines
    let windSpeed = envState.obs ? envState.obs.wind_norm : 0.2;
    if (envState.obs && envState.obs.wind_swan_active > 0) windSpeed = 0;
    
    windTurbines.forEach(rotor => {
        rotor.rotation.x -= windSpeed * dt * 5;
    });

    // Pulse hospital if critical
    const hosp = scene.getObjectByName('Hospital');
    if (hosp && envState.obs) {
        const mat = hosp.children[0].material;
        if (envState.obs.hosp_served_ratio < 0.95) {
            mat.emissive.setHex(COLOR_CRITICAL);
            mat.emissiveIntensity = 0.5 + 0.5 * Math.sin(Date.now() * 0.005);
        } else {
            mat.emissive.setHex(0x000000);
        }
    }

    // BESS height/color based on SoC
    const bess = scene.getObjectByName('bess');
    if (bess && envState.obs) {
        const scale = Math.max(0.1, envState.obs.battery_soc);
        bess.scale.y = THREE.MathUtils.lerp(bess.scale.y, scale, 0.1);
        bess.position.y = (scale * 4) / 2;
        
        if (envState.obs.battery_soc < 0.2) bess.material.color.setHex(COLOR_CRITICAL);
        else bess.material.color.setHex(COLOR_STABLE);
    }

    // Sun movement (Day/Night cycle)
    if (envState.obs) {
        sunLight.position.x = 20 * envState.obs.time_sin;
        sunLight.position.y = 20 * Math.max(0, envState.obs.time_cos);
        sunLight.intensity = Math.max(0.1, envState.obs.time_cos * 1.5);

        // Solar panels go dark during black swan
        solarPanels.forEach(p => {
            if (envState.obs.solar_swan_active > 0) {
                p.material.color.setHex(0x111111);
            } else {
                p.material.color.setHex(0x0ea5e9);
            }
        });
    }

    renderer.render(scene, camera);
}


// --- API Interaction ---

async function resetEnvironment() {
    try {
        const res = await fetch('/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: Math.floor(Math.random() * 1000) })
        });
        const data = await res.json();
        envState.obs = data.observation;
        envState.step = 0;
        envState.done = data.done;
        updateUI();
    } catch (e) {
        console.error("Reset failed", e);
    }
}

// Simple JS Heuristic for the UI
function getHeuristicAction(obs) {
    let bess = 0;
    let ind = 0;
    let res = 0;

    const renew = obs.solar_norm + obs.wind_norm;
    const tight = renew < 0.4;
    const any_swan = obs.solar_swan_active > 0.5 || obs.wind_swan_active > 0.5;

    if (tight && obs.battery_soc > 0.25) bess = 4; // discharge 50%
    else if (renew > 0.7 && obs.battery_soc < 0.85) bess = 2; // charge 50%

    if (any_swan && obs.battery_soc > 0.3) bess = 5; // discharge 100%

    const freq_bad = Math.abs(obs.frequency_norm - 0.5) > 0.15;
    if (any_swan || freq_bad) {
        ind = 2; // 80%
        res = 2; // 70%
    }

    return {
        bess: bess,
        hospital: 0, // Always protect
        industrial: ind,
        residential: res
    };
}

async function stepEnvironment() {
    if (envState.done) return;
    
    const action = getHeuristicAction(envState.obs);
    
    try {
        // According to OpenEnv, action payload can be nested
        const res = await fetch('/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(action)
        });
        const data = await res.json();
        envState.obs = data.observation;
        envState.step++;
        envState.done = data.done;
        updateUI();

        if (envState.done && autoRunInterval) {
            toggleAutoRun();
        }
    } catch (e) {
        console.error("Step failed", e);
    }
}

function updateUI() {
    const obs = envState.obs;
    if (!obs) return;

    // Time & Step
    document.getElementById('val-step').innerText = `${envState.step} / 24`;
    document.getElementById('val-reward').innerText = obs.reward.toFixed(2);
    document.getElementById('val-shed').innerText = (obs.cumulative_shed_ratio_norm * 20).toFixed(1) + '%';
    
    const hour = Math.atan2(obs.time_sin, obs.time_cos) * (12 / Math.PI) + 6;
    let h = Math.floor(hour); if(h < 0) h += 24; if(h >= 24) h -= 24;
    document.getElementById('val-time').innerText = `${h.toString().padStart(2, '0')}:00`;

    // Physics
    const freq = (obs.frequency_norm * 2) + 49.0;
    document.getElementById('val-freq').innerText = freq.toFixed(2) + ' Hz';
    const fPct = obs.frequency_norm * 100;
    const fBar = document.getElementById('bar-freq');
    fBar.style.width = `${fPct}%`;
    fBar.className = (freq < 49.5 || freq > 50.5) ? 'progress-bar warning' : 'progress-bar safe';

    document.getElementById('val-import').innerText = (obs.grid_import_norm * 100).toFixed(1) + '%';

    // BESS
    document.getElementById('val-soc').innerText = (obs.battery_soc * 100).toFixed(1) + '%';
    document.getElementById('bar-soc').style.width = `${obs.battery_soc * 100}%`;
    
    document.getElementById('val-health').innerText = (obs.bess_health * 100).toFixed(1) + '%';

    // Sectors
    updateSector('hosp', obs.hosp_demand_norm, obs.hosp_served_ratio);
    updateSector('ind', obs.ind_demand_norm, obs.ind_served_ratio);
    updateSector('res', obs.res_demand_norm, obs.res_served_ratio);

    // Alerts
    const alertSolar = document.getElementById('alert-solar');
    const alertWind = document.getElementById('alert-wind');
    
    if (obs.solar_swan_active > 0) alertSolar.classList.remove('hidden');
    else alertSolar.classList.add('hidden');

    if (obs.wind_swan_active > 0) alertWind.classList.remove('hidden');
    else alertWind.classList.add('hidden');
}

function updateSector(id, demandNorm, servedRatio) {
    document.getElementById(`dem-${id}`).innerText = (demandNorm * 500).toFixed(0) + ' kW';
    document.getElementById(`srv-${id}`).innerText = (servedRatio * 100).toFixed(0) + '%';
    const bar = document.getElementById(`bar-${id}`);
    bar.style.width = `${servedRatio * 100}%`;
    
    if (servedRatio < 0.95) bar.className = 'progress-bar danger';
    else if (servedRatio < 0.99) bar.className = 'progress-bar warning';
    else bar.className = 'progress-bar safe';

    const item = document.getElementById(`sector-${id}`);
    if (servedRatio < 0.95) item.classList.add('critical-alert');
    else item.classList.remove('critical-alert');
}

// --- Controls ---

document.getElementById('btn-reset').addEventListener('click', () => {
    if (autoRunInterval) toggleAutoRun();
    resetEnvironment();
});

document.getElementById('btn-step').addEventListener('click', () => {
    stepEnvironment();
});

function toggleAutoRun() {
    const btn = document.getElementById('btn-auto');
    if (autoRunInterval) {
        clearInterval(autoRunInterval);
        autoRunInterval = null;
        btn.innerText = 'Auto-Run';
        btn.classList.remove('danger');
        btn.classList.add('accent');
    } else {
        if (envState.done) resetEnvironment();
        autoRunInterval = setInterval(stepEnvironment, 800);
        btn.innerText = 'Stop';
        btn.classList.remove('accent');
        btn.classList.add('danger');
    }
}
document.getElementById('btn-auto').addEventListener('click', toggleAutoRun);

// Init
init3D();
setTimeout(resetEnvironment, 500);
