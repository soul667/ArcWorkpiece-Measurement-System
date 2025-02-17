import numpy as np
import cv2

class CoordinateSelector:
    def __init__(self, points,i,j):
        """
        初始化坐标选择器。

        参数:
        - points: 点云数据，形状为 (N, 3) 的 numpy 数组，包含 (x, y, z) 坐标。
        """
        self.points = points
        self.image = None
        self.current_lines = []  # 当前正在选择的坐标线
        self.mode = "Mode 1"  # 初始模式
        self.x_regions = []  # x 坐标选择的区域
        self.y_regions = []  # y 坐标选择的区域
        self.ii=i
        self.jj=j

    def show_touying_and_choose(self):
        """
        显示点云投影，并允许通过鼠标选择多个区域。
        """
        # 提取 x 和 y 坐标，并归一化到图像范围
        x = self.points[:, self.ii]
        y = self.points[:, self.jj]

        # 归一化到 [0, 1] 再映射到 720x1280
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        x_img = (x_norm * 1279).astype(np.int32)
        y_img = (y_norm * 719).astype(np.int32)

        # 创建 720x1280 的图像并绘制点
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image[y_img, x_img] = (255, 255, 255)  # 白色点

        # 显示图像并设置回调
        cv2.imshow("Select Region", self.image)
        cv2.setMouseCallback("Select Region", self.mouse_callback)

        print("鼠标左键画线，右键撤销。完成一个区域后，按回车键确认区域。按 'm' 切换模式，按 'q' 退出。")

        while True:
            self.update_window_text()
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # 回车键确认当前模式的区域
                if len(self.current_lines) == 2:
                    new_region = sorted(self.current_lines)
                    if self.mode == "Mode 1":
                        self.x_regions.append(new_region)
                    else:
                        self.y_regions.append(new_region)
                    self.current_lines = []  # 重置当前区域
                    print(f"{self.mode} 区域完成: {new_region}")
                    self.redraw_image()
                else:
                    print("需要选择两条线才能确认一个区域！")
            elif key == ord('m'):  # 切换模式
                self.mode = "Mode 2" if self.mode == "Mode 1" else "Mode 1"
                print(f"切换到 {self.mode}")
            elif key == ord('q'):  # 退出窗口
                print("退出窗口")
                break

        cv2.destroyAllWindows()
        print(f"x 坐标区域 (Mode 1): {self.x_regions}")
        print(f"y 坐标区域 (Mode 2): {self.y_regions}")

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标事件回调函数，用于画线或撤销。

        参数:
        - event: 鼠标事件类型。
        - x, y: 鼠标点击的像素坐标。
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击画线
            if len(self.current_lines) < 2:
                if self.mode == "Mode 1":  # 选择 x 坐标
                    coord = x / 1279 * (self.points[:, self.ii].max() - self.points[:, self.ii].min()) + self.points[:, self.ii].min()
                else:  # 选择 y 坐标
                    coord = y / 719 * (self.points[:, self.jj].max() - self.points[:, self.jj].min()) + self.points[:, self.jj].min()
                self.current_lines.append(coord)

                # 在图像中画线
                pos = x if self.mode == "Mode 1" else y
                line_color = (255, 0, 0) if self.mode == "Mode 1" else (0, 255, 0)
                if self.mode == "Mode 1":
                    cv2.line(self.image, (pos, 0), (pos, 719), line_color, 1)
                else:
                    cv2.line(self.image, (0, pos), (1279, pos), line_color, 1)
                cv2.imshow("Select Region", self.image)
                print(f"已选择 {self.mode} 坐标 = {coord:.3f}")

        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击撤销
            if self.current_lines:
                removed_coord = self.current_lines.pop()  # 撤销上一步选择
                print(f"撤销选择 {self.mode} 坐标 = {removed_coord:.3f}")
                self.redraw_image()

    def redraw_image(self):
        """
        重新绘制图像，包括点云、当前选择和所有区域。
        """
        # 提取 x 和 y 坐标，并归一化到图像范围
        x = self.points[:, self.ii]
        y = self.points[:, self.jj]
        print("xmin",x.min())
        print("xmax",x.max())
        print("ymin",y.min())
        print("ymax",y.max())
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        x_img = (x_norm * 1279).astype(np.int32)
        y_img = (y_norm * 719).astype(np.int32)

        # 清空图像并重新绘制
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image[y_img, x_img] = (255, 255, 255)  # 白色点

        # 绘制 x 坐标区域（Mode 1）
        for region in self.x_regions:
            for coord in region:
                pos = int((coord - self.points[:, self.ii].min()) / (self.points[:, self.ii].max() - self.points[:, self.ii].min()) * 1279)
                cv2.line(self.image, (pos, 0), (pos, 719), (255, 0, 0), 1)

        # 绘制 y 坐标区域（Mode 2）
        for region in self.y_regions:
            for coord in region:
                pos = int((coord - self.points[:, self.jj].min()) / (self.points[:, self.jj].max() - self.points[:, self.jj].min()) * 719)
                cv2.line(self.image, (0, pos), (1279, pos), (0, 255, 0), 1)

        # 绘制当前选择的线
        line_color = (255, 0, 0) if self.mode == "Mode 1" else (0, 255, 0)
        for coord in self.current_lines:
            pos = int((coord - (self.points[:, self.ii].min() if self.mode == "Mode 1" else self.points[:, self.jj].min())) /
                      ((self.points[:, self.ii].max() - self.points[:, self.ii].min()) if self.mode == "Mode 1" else
                       (self.points[:, self.jj].max() - self.points[:, self.jj].min())) *
                      (1279 if self.mode == "Mode 1" else 719))
            if self.mode == "Mode 1":
                cv2.line(self.image, (pos, 0), (pos, 719), line_color, 1)
            else:
                cv2.line(self.image, (0, pos), (1279, pos), line_color, 1)

        cv2.imshow("Select Region", self.image)

    def update_window_text(self):
        """
        在窗口上显示当前模式和选定的区域信息。
        """
        display_image = self.image.copy()
        text = f"Mode: {self.mode} | X Regions: {len(self.x_regions)} | Y Regions: {len(self.y_regions)}"
        cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if self.current_lines:
            current_text = f"Current Lines: {', '.join([f'{x:.3f}' for x in self.current_lines])}"
            cv2.putText(display_image, current_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

        cv2.imshow("Select Region", display_image)


# 示例使用
if __name__ == "__main__":
    num_points = 1000
    points = np.random.rand(num_points, 3) * 100

    selector = CoordinateSelector(points,1,2)
    selector.show_touying_and_choose()
