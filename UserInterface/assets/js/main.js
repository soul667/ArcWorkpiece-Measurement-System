import * as THREE from './three.module.js';
  import { OrbitControls } from'./OrbitControls.js';
  import { PCDLoader } from './PCDLoader.js';
import { GUI } from './lil-gui.module.min.js';

// 获取渲染器的大小
var halfW = window.innerWidth / 2;
var halfH = window.innerHeight / 2;
// 创建场景
var scene = [];
for (var i = 0; i <= 0; i++) {
  scene[i] = new THREE.Scene();
}
// 创建相机camera[i].position
var camera = [];
for (var i = 0; i <= 0; i++) {
  camera[i] = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.0001, 1000);
  camera[i].position.set(50, 0, 0);
  camera[i].lookAt(100, 0, 0);
}
//三维坐标辅助线
var axesHelper=[];
for (var i = 0; i <= 0; i++) {
  axesHelper[i] = new THREE.AxesHelper(800);
}
//辅助网格
const gridHelper = [];
for (var i = 0; i <= 0; i++) {
  gridHelper[i] = new THREE.GridHelper(700, 50, 0x000000, 0xAAAAAA);
}
// 创建渲染器
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
function render() {
  // 设置四个视口的位置和大小
  var viewports = [
    { x: 0, y: 0, w: halfW*2, h: halfH*2 }, // 左上
    { x: halfW, y: halfH, w: halfW, h: halfH }, // 右上
    { x: 0, y: 0, w: halfW, h: halfH }, // 左下
    { x: halfW, y: 0, w: halfW, h: halfH } // 右下
  ];

  // 遍历四个视口
  for (var i = 0; i <= 0; i++) {
    var viewport = viewports[i];
    renderer.setScissorTest(true);
    renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h);
    renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h);
    // renderer.setClearColor(0xFFDDAA >> i * 6);
    // set color black
    renderer.setClearColor(0x000000);
    renderer.render(scene[i], camera[i]);
  }
}

// 渲染函数
function animate() {
  requestAnimationFrame(animate);
  for (var i = 0; i <= 0; i++) {
    control[i].update();
  }
  render();
}

var control = [];
for (var i = 0; i <= 0; i++) {
  control[i] = new OrbitControls(camera[i], renderer.domElement);

}

//以上为threejs三大件
//------------------------------------------------------------------
//变量声明
var gui = new GUI();
const loader = new PCDLoader();
var point = [];
var data;
var originCloud;
var minCutSegmentationData;
var cloudFilteredData;
var cloudKeyPointData;
var voxelGridData;
const parameters = {
  globalvariable: {
    rotate: true,
    PCD文件: "/assets/images/cat.pcd"
  },

  segmentationParam://分割
  {
    分割点大小: 1,
    分割点颜色: 0x888800,
    物体中心: {
      x: 69,
      y: -18,
      z: 0.57,
    },
    分割半径: 3.0433856,
    sigma分割结果的粒度和平滑度: 0.25,//图像的平滑程度、标准偏差或像素间的差异。
    源权重: 0.8,//源权重
    邻域点数: 14
  },
  filter://离群点滤波参数
  {
    滤波离群点大小: 1,
    滤波离群点颜色: 0x000000,
    邻居点范围: 40,//考虑的邻居点的范围，考虑点周围的40个邻居点。
    乘数阈值: 1.0//乘数阈值，用于确定离群点的标准偏差倍数。
  },
  keypointParam://关键点
  {
    关键点大小: 5,
    关键点颜色: 0xff0000,
    搜索半径: 6,//显著半径，值越大过滤的点越多
    非最大值抑制半径: 4,//非最大半径，值越大去除的关键点越多
    threshold21: 0.975,
    threshold32: 0.975,
    最小邻域点数: 5//为了将一个关键点保留在关键点列表中所需的最小邻居数
    
    /*一个点与最近的关键点的距离大于 Threshold32，
    *且与最近不匹配点的距离小于 Threshold21
    则认为该点为关键点
    */
  },
  voxelGridParam://体素网格参数
  {
    体素网格点大小: 10,//点大小
    体素网格点颜色: 0x00ffff,
    体素大小 : 1//体素大小，每个体素的大小是1x1x1
  }
}



// async function loadData() {
//   await PCL.init({
//     url: `/assets/js/pcl-core.wasm`
//   });
//   data = await fetch(parameters.globalvariable.PCD文件).then((res) => res.arrayBuffer());
//   originCloud = PCL.loadPCDData(data, PCL.PointXYZ);
  
//   setMinCutSegmentation();
//   setStatisticalOutlierRemoval();
//   setISSKeypoint3D();
//   setVoxelGrid();
//   loadMinCutSegmentation(minCutSegmentationData, 0)
//   loadStatisticalOutlierRemoval(cloudFilteredData, 1);
//   loadISSKeypoint3D(cloudKeyPointData, 2);
//   loadVoxelGrid(voxelGridData, 3);
// }

async function loadData() {
  await PCL.init({
    url: `/assets/js/pcl-core.wasm`
  });

  let data;
  const fileName = parameters.globalvariable.点云 ; // 确保有文件名
  const extension = fileName.slice(-3).toLowerCase();

  if (extension === 'pcd') {
    data = await fetch(parameters.globalvariable.点云).then((res) => res.arrayBuffer());
    originCloud = PCL.loadPCDData(data, PCL.PointXYZ);
  } else if (extension === 'ply') {
    data = await fetch(parameters.globalvariable.点云).then((res) => res.arrayBuffer());
    originCloud = PCL.loadPLYFile(data, PCL.PointXYZ);
  } else {
    throw new Error('Unsupported file format');
  }

  setMinCutSegmentation();
  setStatisticalOutlierRemoval();
  setISSKeypoint3D();
  setVoxelGrid();
  loadMinCutSegmentation(minCutSegmentationData, 0);
  loadStatisticalOutlierRemoval(cloudFilteredData, 1);
  loadISSKeypoint3D(cloudKeyPointData, 2);
  loadVoxelGrid(voxelGridData, 3);
}
//GUI
// async function loadGlobalGUI(){
//   var gui = new GUI().title("全局设置");
//   const CameraRotate = {bool:false}
//   gui.domElement.style.top = `${halfH-150}px`;
//   gui.domElement.style.left = `${halfW-150}px`;
//   gui.add(parameters.globalvariable, "点云", {
//     'three_car三辆车':"/assets/images/min_cut_segmentation_tutorial.pcd",
//     'horse驴': "/assets/images/horse.pcd",
//     'lioness母狮子': "/assets/images/lioness.pcd",
//     'wolf狼': "/assets/images/humaninoffice.pcd",
//     "testply":"/assets/temp/temp.pcd",
//   }).onFinishChange(function () {
//     loadData();
//   })   
//   var i=0;
//   // var newFolder1= gui.addFolder("相机是否旋转")
//   // var newFolder2= gui.addFolder("相机旋转速度")
//   // for(;i<=3;i++){
//   // newFolder1.add(control[i],'autoRotate').name("图"+(i+1)+"是否旋转");
//   // newFolder2.add(control[i],'autoRotateSpeed',0 ,5).name("图"+(i+1)+"旋转速度").step(0.1);
//   // }
// }
async function loadGlobalGUI() {
  var gui = new GUI().title("全局设置");
  const CameraRotate = { bool: false };
  gui.domElement.style.top = `${halfH - 150}px`;
  gui.domElement.style.left = `${halfW - 150}px`;

  gui.add(parameters.globalvariable, "点云", {
    'three_car三辆车': "/assets/images/min_cut_segmentation_tutorial.pcd",
    'horse驴': "/assets/images/horse.pcd",
    'lioness母狮子': "/assets/images/lioness.pcd",
    'wolf狼': "/assets/images/humaninoffice.pcd",
    "testply": "/assets/temp/temp.pcd",
  }).onFinishChange(function () {
    loadData();
  });

  // 添加上传点云的输入框
  var uploadFolder = gui.addFolder("上传点云");
  var pointCloudInput = uploadFolder.add({}, "pointCloud", "").name("上传点云").onChange(function (value) {
    uploadPointCloud(value);
  });
  pointCloudInput.domElement.innerHTML = `<input type="file" id="point-cloud" >`;

  var i = 0;
  // var newFolder1= gui.addFolder("相机是否旋转")
  // var newFolder2= gui.addFolder("相机旋转速度")
  // for(;i<=3;i++){
  // newFolder1.add(control[i],'autoRotate').name("图"+(i+1)+"是否旋转");
  // newFolder2.add(control[i],'autoRotateSpeed',0 ,5).name("图"+(i+1)+"旋转速度").step(0.1);
  // }
}

// 上传点云的函数
function uploadPointCloud(file) {
  if (file) {
    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        alert(data.message);
      })
      .catch(error => {
        console.error("Error:", error);
        alert("文件上传失败,请重试。");
      });
  }
}


//左上最小分割
//---------
//模型设置
function setMinCutSegmentation() {
  var view;
  const mcSeg = new PCL.MinCutSegmentation();
  const objectCenter = new PCL.PointXYZ(
    parameters.segmentationParam.物体中心.xyz);
  const foregroundPoints = new PCL.PointCloud();
  foregroundPoints.addPoint(objectCenter);
  mcSeg.setForegroundPoints(foregroundPoints);
  mcSeg.setInputCloud(originCloud);
  mcSeg.setRadius(parameters.segmentationParam.分割半径);
  mcSeg.setSigma(parameters.segmentationParam.sigma分割结果的粒度和平滑度);
  mcSeg.setSourceWeight(parameters.segmentationParam.源权重);
  mcSeg.setNumberOfNeighbours(parameters.segmentationParam.邻域点数);
  mcSeg.extract(); 
  const minCutPointCloud = mcSeg.getColoredCloud();
  minCutSegmentationData = PCL.savePCDDataASCII(minCutPointCloud);
  return minCutSegmentationData;  
 
}

//模型加载
async function loadMinCutSegmentation(data, index) {

  scene[index].remove(point[index]);

  point[index] = loader.parse(data, ''), function (error) {
    console.log('An error happened');
  };
  const material = new THREE.PointsMaterial({
    size: parameters.segmentationParam.分割点大小,
    color: parameters.segmentationParam.分割点颜色,
    opacity: 0.5,
    transparent: true,
    sizeAttenuation: false,
  });
  point[index].geometry.center();
  point[index].geometry.scale(1, 1, 1);
  point[index].material = material;
  scene[index].add(point[index]);
  // scene[index].add(axesHelper[index]);
  // scene[index].add(gridHelper[index]);
}
async function loadMinCutSegmentationGUI(){
//GUI
var segmentationGUI = new GUI().title("最小分割");
segmentationGUI.domElement.style.left = `0px`;;
segmentationGUI.domElement.style.top = `0px`;;

segmentationGUI.add(parameters.segmentationParam, "分割点大小", 0.1, 5, 0.1).onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});
segmentationGUI.addColor(parameters.segmentationParam, "分割点颜色").onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});

var objectCenterGUI = segmentationGUI.addFolder("物体中心");
objectCenterGUI.open();
segmentationGUI.add(parameters.segmentationParam, "分割半径", 0, 100 ).onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});
segmentationGUI.add(parameters.segmentationParam, "sigma分割结果的粒度和平滑度", 0, 1, 0.01).onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});
segmentationGUI.add(parameters.segmentationParam, "源权重", 0, 1, 0.01).onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});
segmentationGUI.add(parameters.segmentationParam, "邻域点数", 1, 50, 1).onFinishChange(function () {
  setMinCutSegmentation();
  loadMinCutSegmentation(minCutSegmentationData, 0);
});
}
//右上，统计离群点滤波
//-------------
//模型设置
function setStatisticalOutlierRemoval() {
  const sor = new PCL.StatisticalOutlierRemoval();
  sor.setInputCloud(originCloud);
  sor.setMeanK(parameters.filter.邻居点范围);
  sor.setStddevMulThresh(parameters.filter.乘数阈值);
  const cloudFiltered = sor.filter();
  cloudFilteredData = PCL.savePCDDataASCII(cloudFiltered);
  return cloudFilteredData;
}
//模型加载
async function loadStatisticalOutlierRemoval(data, index) {

  scene[index].remove(point[index]);

  point[index] = loader.parse(data, ''), function (error) {
    console.log('An error happened');
  };
  const material = new THREE.PointsMaterial({
    size: parameters.filter.滤波离群点大小,
    color: parameters.filter.滤波离群点颜色,
    opacity: 0.5,
    transparent: true,
    sizeAttenuation: false,
  });
  point[index].geometry.center();
  point[index].geometry.scale(1, 1, 1);
  point[index].material = material;
  scene[index].add(point[index]);
  //camera[index].lookAt(point[index].position);
  scene[index].add(axesHelper[index]);
  scene[index].add(gridHelper[index]);
}
async function loadFilterGUI(){
//GUI设置
var filterGUI = new GUI().title('统计离群点滤波参数');
filterGUI.add(parameters.filter, "滤波离群点大小", 0.1, 5, 0.1).onFinishChange(function () {
  cloudFilteredData = setStatisticalOutlierRemoval();
  loadStatisticalOutlierRemoval(cloudFilteredData, 1);
});
filterGUI.addColor(parameters.filter, "滤波离群点颜色").onFinishChange(function () {
  setStatisticalOutlierRemoval();
  loadStatisticalOutlierRemoval(cloudFilteredData, 1);
});

filterGUI.add(parameters.filter, "邻居点范围", 0, 100, 1).onFinishChange(function () {
  setStatisticalOutlierRemoval();
  loadStatisticalOutlierRemoval(cloudFilteredData, 1);
});
filterGUI.add(parameters.filter, "乘数阈值", 0, 10).onFinishChange(function () {
  setStatisticalOutlierRemoval();
  loadStatisticalOutlierRemoval(cloudFilteredData, 1);
});
}
//左下，ISS关键点提取
function setISSKeypoint3D() {
  const resolution = PCL.computeCloudResolution(originCloud);
  const tree = new PCL.SearchKdTree();
  const keypoints = new PCL.PointCloud();
  const iss = new PCL.ISSKeypoint3D();
  iss.setSearchMethod(tree);
  iss.setSalientRadius(parameters.keypointParam.搜索半径 * resolution);
  iss.setNonMaxRadius(parameters.keypointParam.非最大值抑制半径 * resolution);
  iss.setThreshold21(parameters.keypointParam.threshold21);
  iss.setThreshold32(parameters.keypointParam.threshold32);
  iss.setMinNeighbors(parameters.keypointParam.最小邻域点数);
  iss.setInputCloud(originCloud);
  iss.compute(keypoints);
  cloudKeyPointData = PCL.savePCDDataASCII(keypoints);
  return cloudKeyPointData;
}
//模型加载
async function loadISSKeypoint3D(data, index) {

  scene[index].remove(point[index]);

  point[index] = loader.parse(data, ''), function (error) {
    console.log('An error happened');
  };
  const material = new THREE.PointsMaterial({
    size: parameters.keypointParam.关键点大小,
    color: parameters.keypointParam.关键点颜色,
    opacity: 0.5,
    transparent: true,
    sizeAttenuation: false,
  });
  point[index].geometry.center();
  point[index].geometry.scale(1, 1, 1);
  point[index].material = material;
  scene[index].add(point[index]);
  camera[index].lookAt(point[index].position);
  scene[index].add(axesHelper[index]);
  scene[index].add(gridHelper[index]);
}
//gui
async function loadKeypointGUI(){
  var keypointGUI = new GUI().title("关键点提取");
  keypointGUI.domElement.style.left = '0px';
  keypointGUI.domElement.style.top = `${halfH}px`;;
  keypointGUI.add(parameters.keypointParam, "关键点大小", 0.1, 5, 0.1).onFinishChange(function () {
    setISSKeypoint3D();
    loadISSKeypoint3D(cloudKeyPointData, 2 );
  });
  keypointGUI.addColor(parameters.keypointParam, "关键点颜色").onFinishChange(function () {
    setISSKeypoint3D();
    loadISSKeypoint3D(cloudKeyPointData, 2);
  });
  
  keypointGUI.add(parameters.keypointParam, "搜索半径", 1, 20, 1).onFinishChange(function () {
    setISSKeypoint3D();
    loadISSKeypoint3D(cloudKeyPointData, 2);
  });
  keypointGUI.add(parameters.keypointParam, "非最大值抑制半径", 1, 20, 1).onFinishChange(function () {
    setISSKeypoint3D();
    loadISSKeypoint3D(cloudKeyPointData, 2);
  });
  keypointGUI.add(parameters.keypointParam, "最小邻域点数", 1, 20, 1).onFinishChange(function () {
    setISSKeypoint3D();
    loadISSKeypoint3D(cloudKeyPointData, 2);
  });
}
//右下体素网格滤波
//---------
//模型设置
function setVoxelGrid() {
  const sor = new PCL.VoxelGrid();
  sor.setInputCloud(originCloud);
  sor.setLeafSize(parameters.voxelGridParam.体素大小, parameters.voxelGridParam.体素大小,parameters.voxelGridParam.体素大小);
  const voxelGridCloud = sor.filter();
  voxelGridData = PCL.savePCDDataASCII(voxelGridCloud);
  return voxelGridData;  
}
//模型加载
async function loadVoxelGrid(data, index) {

  scene[index].remove(point[index]);

  point[index] = loader.parse(data, ''), function (error) {
    console.log('An error happened');
  };
  const material = new THREE.PointsMaterial({
    size: parameters.voxelGridParam.体素网格点大小,
    color: parameters.voxelGridParam.体素网格点颜色,
    opacity: 0.5,
    transparent: true,
    sizeAttenuation: false,
  });
  point[index].geometry.center();
  point[index].geometry.scale(1, 1, 1);
  point[index].material = material;
  scene[index].add(point[index]);
  scene[index].add(axesHelper[index]);
  scene[index].add(gridHelper[index]);
}
//GUI
async function loadVoxelGridGUI(){
var voxelGridGUI = new GUI().title("体素网格滤波");
voxelGridGUI.domElement.style.top = `${halfH}px`;
voxelGridGUI.add(parameters.voxelGridParam, "体素网格点大小", 1, 20, 1).onFinishChange(function () {
  setVoxelGrid();
  loadVoxelGrid(voxelGridData, 3);
});
voxelGridGUI.addColor(parameters.voxelGridParam, "体素网格点颜色").onFinishChange(function () {
  setVoxelGrid();
  loadVoxelGrid(voxelGridData, 3);
});

voxelGridGUI.add(parameters.voxelGridParam, "体素大小", 0, 5,0.01).onFinishChange(function () {
  setVoxelGrid();
  loadVoxelGrid(voxelGridData, 3);
});
}

loadGlobalGUI();//全局
loadMinCutSegmentationGUI()//最小分割
loadFilterGUI();//离群点滤波
loadKeypointGUI();//关键点
loadVoxelGridGUI();//体素网格滤波
loadData();
animate();