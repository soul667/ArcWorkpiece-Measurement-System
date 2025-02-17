const canvas = document.getElementById("point-cloud-canvas");
const ctx = canvas.getContext("2d");
const refreshBtn = document.getElementById("refresh-btn");
const xRegionsList = document.getElementById("x-regions-list");
const yRegionsList = document.getElementById("y-regions-list");
const submitBtn = document.getElementById("submit-btn");

let lines = [];
let x_regions = [];
let y_regions = [];
let currentView = "xy"; // 当前视图，默认是 XY
let currentMode = 0; // 当前模式，0 为垂直线 (x)，1 为水平线 (y)
let coordinateRanges = {}; // 存储从服务器获取的坐标范围

const img = new Image();
img.src = `/img/${currentView}`;

// 图片加载完成后绘制到画布
img.addEventListener("load", () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
});

// 更新视图按钮事件
document.getElementById("btn-xy").addEventListener("click", () => updateView("xy"));
document.getElementById("btn-xz").addEventListener("click", () => updateView("xz"));
document.getElementById("btn-yz").addEventListener("click", () => updateView("yz"));

// 更新视图函数
function updateView(view) {
    currentView = view;
    img.src = `/img/${view}`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    x_regions = [];
    y_regions = [];
    lines = [];
    updateRegionsDisplay();
    fetchCoordinateRanges(view); // 获取对应的坐标范围
}

// 获取坐标范围
async function fetchCoordinateRanges(view) {
    try {
        const response = await fetch(`/yml/info`);
        if (!response.ok) {
            console.error("Failed to fetch YAML data");
            return;
        }
        const text = await response.text();
        const ranges = parseYaml(text); // 解析 YAML
        coordinateRanges = ranges;
        console.log("Coordinate ranges:", coordinateRanges);
    } catch (error) {
        console.error("Error fetching YAML:", error);
    }
}

// 解析 YAML 文件
function parseYaml(yamlText) {
    const lines = yamlText.split("\n").filter(line => line.includes(":"));
    const result = {};
    lines.forEach(line => {
        const [key, value] = line.split(":").map(str => str.trim());
        result[key] = parseFloat(value);
    });
    return result;
}

// 重映射坐标
function remapCoordinates(currentView,pixel, axis) {
    const { [`${axis}_min`]: min, [`${axis}_max`]: max } = coordinateRanges;
    // var canvasSize = axis == "x" ? canvas.width : canvas.height;
    var canvasSize =canvas.width ;

    if((currentView == "xy" & axis == "y") || (currentView == "xz" & axis == "z") || (currentView == "yz" & axis == "z")){
        canvasSize= canvas.height;
    }
    // const canvasSize = axis === "x" ? canvas.width : canvas.height;
    return min + (max - min) * (pixel / canvasSize);
}

// 重绘画布
function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    x_regions.forEach(region => {
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(region[0], 0);
        ctx.lineTo(region[0], canvas.height);
        ctx.moveTo(region[1], 0);
        ctx.lineTo(region[1], canvas.height);
        ctx.stroke();
    });

    y_regions.forEach(region => {
        ctx.strokeStyle = "green";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, region[0]);
        ctx.lineTo(canvas.width, region[0]);
        ctx.moveTo(0, region[1]);
        ctx.lineTo(canvas.width, region[1]);
        ctx.stroke();
    });
}

// 更新区域显示
function updateRegionsDisplay() {
    xRegionsList.innerHTML = "";
    yRegionsList.innerHTML = "";

    x_regions.forEach((region, index) => {
        const li = document.createElement("li");
        li.innerHTML = `区域 ${index + 1}: [${region[0]}, ${region[1]}] <button onclick="deleteRegion('x', ${index})">删除</button>`;
        xRegionsList.appendChild(li);
    });

    y_regions.forEach((region, index) => {
        const li = document.createElement("li");
        li.innerHTML = `区域 ${index + 1}: [${region[0]}, ${region[1]}] <button onclick="deleteRegion('y', ${index})">删除</button>`;
        yRegionsList.appendChild(li);
    });
}

// 删除区域
function deleteRegion(type, index) {
    if (type === "x") {
        x_regions.splice(index, 1);
    } else if (type === "y") {
        y_regions.splice(index, 1);
    }
    redrawCanvas();
    updateRegionsDisplay();
}

// 鼠标事件：绘制线条
canvas.addEventListener("mouseup", (e) => {
    const line = { mode: currentMode, x: e.offsetX, y: e.offsetY };
    lines.push(line);

    ctx.strokeStyle = currentMode === 0 ? "red" : "orange";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (currentMode === 0) {
        ctx.moveTo(e.offsetX, 0);
        ctx.lineTo(e.offsetX, canvas.height);
    } else {
        ctx.moveTo(0, e.offsetY);
        ctx.lineTo(canvas.width, e.offsetY);
    }
    ctx.stroke();
});

// 按键事件：切换模式、撤销、存储
window.addEventListener("keydown", (e) => {
    if (e.key === "m" || e.key === "M") {
        currentMode = currentMode === 0 ? 1 : 0;
        console.log(`Mode switched to ${currentMode === 0 ? "Vertical (X)" : "Horizontal (Y)"}`);
    } else if (e.key === "r" || e.key === "R") {
        if (lines.length > 0) {
            lines.pop();
            redrawCanvas();
        }
    } else if (e.key === "Enter") {
        if (lines.length > 0) {
            if (currentMode === 0) {
                const xCoords = lines.map(line => line.x).sort((a, b) => a - b);
                x_regions.push([xCoords[0], xCoords[xCoords.length - 1]]);
            } else {
                const yCoords = lines.map(line => line.y).sort((a, b) => a - b);
                y_regions.push([yCoords[0], yCoords[yCoords.length - 1]]);
            }
            lines = [];
            redrawCanvas();
            updateRegionsDisplay();
        }
    }
});
submitBtn.addEventListener("click", () => {
    // 获取参数值
    const downsampleRate = parseFloat(document.getElementById("downsample-rate").value);
    const ranscDownsampleRate = parseFloat(document.getElementById("ransc-downsample-rate").value);
    const nbNeighbors = parseInt(document.getElementById("nb-neighbors").value, 10);
    const stdRatio = parseFloat(document.getElementById("std-ratio").value);
    const normalDistanceWeight = parseFloat(document.getElementById("normal-distance-weight").value);
    const maxIterations = parseInt(document.getElementById("max-iterations").value, 10);
    const distanceThreshold = parseFloat(document.getElementById("distance-threshold").value);
    const radiusMin = parseFloat(document.getElementById("radius-min").value);
    const radiusMax = parseFloat(document.getElementById("radius-max").value);

    // 根据当前视图动态生成区域数据
    const remappedRegions = {};
    if (currentView === "xy") {
        remappedRegions.x_regions = x_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "x"),
            remapCoordinates(currentView,end, "x"),
        ]);
        remappedRegions.y_regions = y_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "y"),
            remapCoordinates(currentView,end, "y"),
        ]);
    } else if (currentView === "xz") {
        remappedRegions.x_regions = x_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "x"),
            remapCoordinates(currentView,end, "x"),
        ]);
        remappedRegions.z_regions = y_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "z"),
            remapCoordinates(currentView,end, "z"),
        ]);
    } else if (currentView === "yz") {
        remappedRegions.y_regions = x_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "y"),
            remapCoordinates(currentView,end, "y"),
        ]);
        remappedRegions.z_regions = y_regions.map(([start, end]) => [
            remapCoordinates(currentView,start, "z"),
            remapCoordinates(currentView,end, "z"),
        ]);
    }

    // 构造提交数据
    const submission = {
        view: currentView,
        regions: remappedRegions,
        parameters: {
            downsample_rate: downsampleRate,
            ransc_downsample_rate: ranscDownsampleRate,
            denoising: { nb_neighbors: nbNeighbors, std_ratio: stdRatio },
            ransc: {
                normal_distance_weight: normalDistanceWeight,
                max_iterations: maxIterations,
                distance_threshold: distanceThreshold,
                radius_min: radiusMin,
                radius_max: radiusMax,
            },
        },
    };

    console.log("Formatted submission:", JSON.stringify(submission, null, 2));

    fetch("/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submission),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Submission successful:", data);
        })
        .catch(error => {
            console.error("Error during submission:", error);
        });
});
