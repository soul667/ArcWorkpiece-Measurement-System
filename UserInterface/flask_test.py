from flask import Flask, render_template, request, jsonify,send_from_directory,send_file
import os
import open3d as o3d
import cv2
import numpy as np
from PointCouldProgress import *
# from RegionSelector import normalize_and_map #重映射图片 
app = Flask(__name__, static_folder='assets')
load_points=False
img_xy_use=None

# 全局变量，用于存储点云对象
global_source_point_cloud = None
global_source_point_cloud_down = None

import numpy as np
def normalize_and_map(x, y, image_width=1280, image_height=720):
    """归一化并映射 x 和 y 坐标到指定图像尺寸的坐标,并返回 OpenCV 图片。

    参数:
        x (numpy.ndarray): 输入的 x 坐标数组
        y (numpy.ndarray): 输入的 y 坐标数组
        image_width (int): 图像宽度,默认为 1280
        image_height (int): 图像高度,默认为 720

    返回:
        numpy.ndarray: 归一化并映射后的 OpenCV 图片
    """
    # 处理空输入数组
    if x.size == 0 or y.size == 0:
        return np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 处理无效的图像尺寸
    if image_width <= 0 or image_height <= 0:
        raise ValueError("无效的图像尺寸: width={}, height={}".format(image_width, image_height))

    # 归一化到 [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 映射到图像宽度和高度
    x_img = (x_norm * (image_width - 1)).astype(np.int32)
    y_img = (y_norm * (image_height - 1)).astype(np.int32)
    # print(x_img,y_img)

    # 创建 OpenCV 图片并绘制点
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[y_img, x_img] = (255, 255, 255)  # 白色点
    # print(image)
    # save img to file
    # cv2.imwrite('img_xy.jpg',image)
    return image

temp_path= ""
@app.route('/')
def home():
    return render_template('index1.html')

@app.route("/assets/<path:path>")
def react_app_staic(path):
#    return app.send_static_file(f"react-app/{path}")
    print(app.static_folder+f"/{path}")
    return app.send_static_file(app.static_folder+f"/{path}")

# http://127.0.0.1:5000/static/media/logo.6ce24c58023cc2f8fd88fe9d219db6c6.svg
@app.route('/demo')
def home1():
    return render_template('threejs-demo/build/index.html')

@app.route('/img/<img_name>')
def get_image(img_name):
    img_path = os.path.join(app.static_folder, 'temp', f"{img_name}.jpg")
    if os.path.exists(img_path):
        return send_from_directory(os.path.join(app.static_folder, 'temp'), f"{img_name}.jpg")
    else:
        return "Image not found", 404
    
@app.route('/yml/<yml_name>')
def get_yml(yml_name):
    yml_path = os.path.join(app.static_folder, 'temp', f"{yml_name}.yml")
    if os.path.exists(yml_path):
        return send_from_directory(os.path.join(app.static_folder, 'temp'), f"{yml_name}.yml")
    else:
        return "YAML file not found", 404
    
@app.route('/get_ply')
def get_ply():
    ply_path = './temp/temp.ply'
    return send_file(ply_path, mimetype='application/octet-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # 创建临时文件路径
        temp_dir = os.path.join(app.static_folder, 'temp')
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        source_temp_pcd_path = os.path.join(temp_dir, 'source.pcd')
        temp_pcd_path = os.path.join(temp_dir, 'temp.pcd')

        
        # 确保目录存在，如果不存在则创建
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 将文件保存到临时路径
        file.save(temp_ply_path)
        
        try:
            # 读取点云文件
            point_cloud = o3d.io.read_point_cloud(temp_ply_path)
            print("LOAD POINT CLOUD")
            app.global_source_point_cloud = point_cloud

            # 检查点云是否成功加载
            if point_cloud.is_empty():
                return jsonify({"error": "Failed to load point cloud"}), 400
            
            #因为显示性能的问题，所以我们对点云进行下采样
            point_cloud_down = point_cloud.voxel_down_sample(voxel_size=0.5)
            # 将点云保存为 PCD 格式
            o3d.io.write_point_cloud(temp_pcd_path, point_cloud_down)
            o3d.io.write_point_cloud(source_temp_pcd_path, point_cloud)
            
            points = np.array(point_cloud.points)
            img_xy = normalize_and_map(points[:, 0], points[:, 1])
            img_yz = normalize_and_map(points[:, 1], points[:, 2])
            img_xz = normalize_and_map(points[:, 0], points[:, 2])
            img_xy_use = img_xy
            cv2.imwrite(os.path.join(temp_dir, 'xy.jpg'), img_xy)
            cv2.imwrite(os.path.join(temp_dir, 'yz.jpg'), img_yz)
            cv2.imwrite(os.path.join(temp_dir, 'xz.jpg'), img_xz)
            x_min=np.min(points[:, 0])
            x_max=np.max(points[:, 0])
            y_min=np.min(points[:, 1])
            y_max=np.max(points[:, 1])
            z_min=np.min(points[:, 2])
            z_max=np.max(points[:, 2])
            # write to yml
            with open(os.path.join(temp_dir, 'info.yml'), 'w') as f:
                f.write(f"x_min: {x_min}\n")
                f.write(f"x_max: {x_max}\n")
                f.write(f"y_min: {y_min}\n")
                f.write(f"y_max: {y_max}\n")
                f.write(f"z_min: {z_min}\n")
                f.write(f"z_max: {z_max}\n")

            load_points = True

            return jsonify({"message": "File uploaded and saved as both PLY and PCD successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Invalid file"}), 400

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    # 处理 data，例如打印或保存
    print(data)
    # 调用 segmentPointCloud 并进行点云过滤
    print(app.global_source_point_cloud.points)
    data_region = data.get('regions', None)
    filtered_points = segmentPointCloud(
        # custom_down_sample((app.global_source_point_cloud.points), data['parameters']['downsample_rate']),
        app.global_source_point_cloud.points,
        data_region.get('x_regions', None),
        data_region.get('y_regions', None),
        data_region.get('z_regions', None)
    )

    # 构造一个新的点云对象
    cropped_pcd = o3d.geometry.PointCloud()
    # 设置点云的点
    cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    print("下采样完毕")
    # print("开始下采样")
    # cropped_pcd.points = o3d.utility.Vector3dVector(
    # custom_down_sample(filtered_points, data['parameters']['downsample_rate'])
    # )
    # print("下采样完毕")

    print("开始点云去噪")
    # cropped_pcd
    cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
    print("点云去噪完毕")
    
    # # cl, ind = voxel_down_pcd_for_axis1.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
    #  cropped_pcd.remove_statistical_outliers(voxel_down_pcd_for_axis1, nb_neighbors=data['parameters']['denoising']['nb_neighbors'], std_ratio=data['parameters']['denoising']['std_ratio'])

    temp_dir = os.path.join(app.static_folder, 'temp')
    temp_denoise_ply_path = os.path.join(temp_dir, 'temp_denoise.ply')
    # save to temp_denoise_ply_path

    # cropped_pcd.points = filtered_points
    o3d.io.write_point_cloud(temp_denoise_ply_path, cl)
    print("FILTERED DONE")

    return jsonify({"status": "success", "received": data}), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=12345)